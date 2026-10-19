"""Train one model, select thresholds on validation, evaluate on every held-out set."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fieldclf import __version__
from fieldclf.dataset import label_matrix, load
from fieldclf.metrics import decide, per_label, summary, tune_thresholds
from fieldclf.models import Model
from fieldclf.predictor import Classifier

# Held-out evaluation sets. "shift" is Horizon Europe (2021-2027): a later
# programme, never used for training or any selection.
EVAL_SETS = {
    "test": lambda d: d["split"] == "test",
    "shift": lambda d: d["split"] == "shift",
    "shift_human_validated": lambda d: (d["split"] == "shift") & d["human_validated"],
}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(
    model: Model,
    data_dir: Path,
    reports_dir: Path,
    artifacts_dir: Path | None = None,
    train_limit: int | None = None,
) -> dict[str, Any]:
    df, vocab = load(data_dir)
    train = df[df["split"] == "train"]
    if train_limit:
        train = train.sample(n=min(train_limit, len(train)), random_state=0)
    val = df[df["split"] == "val"]
    y_train, y_val = label_matrix(train["labels"], vocab), label_matrix(val["labels"], vocab)

    t0 = time.perf_counter()
    model.fit(train["text"].tolist(), y_train, val["text"].tolist(), y_val)
    fit_seconds = time.perf_counter() - t0

    val_proba = model.predict_proba(val["text"].tolist())
    thresholds = tune_thresholds(y_val, val_proba)

    results: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, select in EVAL_SETS.items():
        part = df[select(df)]
        if part.empty:
            continue
        y = label_matrix(part["labels"], vocab)
        t0 = time.perf_counter()
        proba = model.predict_proba(part["text"].tolist())
        ms_per_doc = 1000 * (time.perf_counter() - t0) / max(len(part), 1)
        pred = decide(proba, thresholds)
        results[name] = summary(y, proba, pred) | {"ms_per_doc": round(ms_per_doc, 3)}
        if name != "shift_human_validated":
            results[name + "_per_label"] = per_label(y, pred, proba, vocab)
        predictions[name] = proba
        predictions[name + "_ids"] = part["id"].to_numpy()

    report = {
        "model": model.name,
        "params": model.params,
        "train_size": len(train),
        "fit_seconds": round(fit_seconds, 1),
        "thresholds": dict(zip(vocab, thresholds.round(2).tolist(), strict=True)),
        "results": results,
        "env": {
            "package": __version__,
            "commit": _git_commit(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
    }

    slug = model.name.replace(":", "__")
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics" / f"{slug}.json").write_text(json.dumps(report, indent=2) + "\n")
    if artifacts_dir:
        Classifier(
            model,
            vocab,
            thresholds,
            meta={k: report[k] for k in ("model", "params")}
            | {"test": results["test"], "shift": results["shift"]},
        ).save(artifacts_dir / slug)
        np.savez_compressed(artifacts_dir / slug / "predictions.npz", **predictions)  # type: ignore[arg-type]
    return report


def leaderboard(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((reports_dir / "metrics").glob("*.json")):
        r = json.loads(f.read_text())
        row: dict[str, object] = {"model": r["model"], "fit_s": r["fit_seconds"]}
        for split in ("test", "shift", "shift_human_validated"):
            m = r["results"].get(split)
            if m is None:
                continue
            row |= {
                f"{split}:micro_f1": round(m["micro_f1"], 3),
                f"{split}:macro_f1": round(m["macro_f1"], 3),
                f"{split}:macro_pr_auc": round(m["macro_pr_auc"], 3),
            }
        row["test:ece"] = round(r["results"]["test"]["ece"], 3)
        row["ms/doc"] = r["results"]["test"]["ms_per_doc"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("test:micro_f1", ascending=False)
