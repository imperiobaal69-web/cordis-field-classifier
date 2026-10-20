"""Figures and error analysis from reports/metrics and artifacts/*/predictions.npz.

Every number in README.md and reports/error_analysis.md is written by this script
or by scripts/run_experiments.py.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fieldclf.dataset import label_matrix, load
from fieldclf.experiment import leaderboard
from fieldclf.metrics import (
    bootstrap_f1,
    decide,
    interval,
    reliability_curve,
    summary,
    tune_thresholds,
)

REPORTS, ARTIFACTS, DATA = Path("reports"), Path("artifacts"), Path("data/processed")
FIG = REPORTS / "figures"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": ["Helvetica Neue", "DejaVu Sans"],
        "font.size": 10,
        "font.weight": 500,
        "axes.titleweight": 600,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
    }
)


def short(label: str) -> str:
    return label.split("/")[-1]


def best_model(board: pd.DataFrame) -> str:
    return str(board.iloc[0]["model"])


def fig_leaderboard(board: pd.DataFrame) -> None:
    board = board.sort_values("test:micro_f1")
    y = np.arange(len(board))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(board) + 1.6))
    ax.barh(y + h / 2, board["test:micro_f1"], h - 0.04, color=BLUE, label="H2020 test")
    ax.barh(
        y - h / 2, board["shift:micro_f1"], h - 0.04, color=ORANGE, label="Horizon Europe (shift)"
    )
    for i, (t, s) in enumerate(zip(board["test:micro_f1"], board["shift:micro_f1"], strict=True)):
        ax.text(t + 0.01, i + h / 2, f"{t:.3f}", va="center", fontsize=9, color=INK_2)
        ax.text(s + 0.01, i - h / 2, f"{s:.3f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, board["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("micro-F1 (thresholds tuned on validation)")
    ax.grid(axis="y", visible=False)
    ax.set_title("micro-F1: H2020 test vs. Horizon Europe", pad=28)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.2,
    )
    fig.tight_layout()
    fig.savefig(FIG / "leaderboard.png", dpi=160)
    plt.close(fig)


def fig_support(per_label: pd.DataFrame, train_support: pd.Series, model: str) -> None:
    df = per_label.assign(train=per_label["label"].map(train_support))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(df["train"], df["f1"], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([100, 300, 1000, 3000, 10000])
    picks = pd.concat([df.nsmallest(2, "f1"), df.nlargest(2, "f1")])
    offsets = [(8, 6), (8, -12), (-10, 8), (-10, -14)]
    for (_, r), off in zip(picks.iterrows(), offsets, strict=True):
        ax.annotate(
            short(r["label"]),
            (r["train"], r["f1"]),
            xytext=off,
            textcoords="offset points",
            fontsize=8.5,
            color=INK_2,
            ha="left" if off[0] > 0 else "right",
        )
    ax.set_xlabel("training projects with this field (log scale)")
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-field F1 vs. training support - {model}")
    fig.tight_layout()
    fig.savefig(FIG / "f1_vs_support.png", dpi=160)
    plt.close(fig)


def fig_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label="perfect")
    for (name, (conf, acc)), color in zip(curves.items(), (BLUE, ORANGE, AQUA), strict=False):
        ax.plot(conf, acc, color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration on H2020 test")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=160)
    plt.close(fig)


def fig_shift(test: pd.DataFrame, shift: pd.DataFrame, model: str) -> None:
    df = test[["label", "f1"]].merge(shift[["label", "f1"]], on="label", suffixes=("_t", "_s"))
    df["drop"] = df["f1_t"] - df["f1_s"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.scatter(df["f1_t"], df["f1_s"], s=36, color=ORANGE, edgecolor=SURFACE, linewidth=1.5)
    for _, r in df.nlargest(4, "drop").iterrows():
        ax.annotate(
            short(r["label"]),
            (r["f1_t"], r["f1_s"]),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_xlabel("F1 on H2020 test")
    ax.set_ylabel("F1 on Horizon Europe")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Which fields degrade under shift - {model}")
    fig.tight_layout()
    fig.savefig(FIG / "shift_per_field.png", dpi=160)
    plt.close(fig)


def error_analysis(model_slug: str, df: pd.DataFrame, vocab: list[str]) -> str:
    report = json.loads((REPORTS / "metrics" / f"{model_slug}.json").read_text())
    thresholds = np.array([report["thresholds"][v] for v in vocab])
    preds = np.load(ARTIFACTS / model_slug / "predictions.npz", allow_pickle=True)
    lines = [f"# Error analysis - `{report['model']}`", ""]
    lines += [
        "Generated by `scripts/make_report.py` from the saved test-set predictions.",
        "",
    ]

    for split in ("test", "shift"):
        ids = preds[split + "_ids"]
        part = df.set_index("id").loc[ids]
        y = label_matrix(part["labels"], vocab)
        proba = preds[split]
        pred = decide(proba, thresholds)
        r = report["results"][split]
        lines += [
            f"## {'H2020 test' if split == 'test' else 'Horizon Europe (shift)'}",
            "",
            f"- projects: {len(y):,}; true fields per project {r['labels_per_project_true']:.2f}, "
            f"predicted {r['labels_per_project_pred']:.2f}",
            f"- micro precision {r['micro_precision']:.3f}, micro recall {r['micro_recall']:.3f}",
            f"- exact set match: {(pred == y).all(axis=1).mean():.1%}; "
            f"at least one correct field: {((pred & y).sum(axis=1) > 0).mean():.1%}",
            "",
        ]
        if split == "shift":
            continue

        # Which wrong field is predicted in place of which missed field.
        swaps: Counter[tuple[str, str]] = Counter()
        for yt, yp in zip(y, pred, strict=True):
            missed = np.flatnonzero(yt & (1 - yp))
            extra = np.flatnonzero(yp & (1 - yt))
            for m in missed:
                for e in extra:
                    swaps[(vocab[m], vocab[e])] += 1
        lines += [
            "### Most frequent substitutions (missed field -> predicted instead)",
            "",
            "| missed | predicted instead | projects |",
            "| --- | --- | ---: |",
        ]
        lines += [f"| {short(a)} | {short(b)} | {n} |" for (a, b), n in swaps.most_common(10)]
        lines.append("")

        per = pd.DataFrame(report["results"]["test_per_label"])
        lines += [
            "### Weakest fields",
            "",
            "| field | test support | precision | recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for _, p in per.nsmallest(8, "f1").iterrows():
            lines.append(
                f"| {p['label']} | {p['support']} | {p['precision']:.2f} | "
                f"{p['recall']:.2f} | {p['f1']:.2f} |"
            )
        lines.append("")

        rng = np.random.default_rng(7)
        wrong = np.flatnonzero((pred != y).any(axis=1))
        lines += ["### Random sample of errors (fixed seed)", ""]
        for i in rng.choice(wrong, size=min(5, len(wrong)), replace=False):
            row = part.iloc[i]
            t = [short(vocab[j]) for j in np.flatnonzero(y[i])]
            p = [short(vocab[j]) for j in np.flatnonzero(pred[i])]
            title = row["text"].split(". ")[0][:110]
            lines += [f"- **{title}** - true: {', '.join(t)}; predicted: {', '.join(p)}"]
        lines.append("")
    return "\n".join(lines) + "\n"


def adaptation(df: pd.DataFrame, vocab: list[str], models: list[str]) -> pd.DataFrame:
    """Re-tune only the thresholds on a small labelled Horizon sample; no retraining.

    Horizon projects are split by a hash of their id into an adaptation pool and a
    disjoint evaluation set; growing prefixes of the pool show how many labelled
    examples the recalibration needs.
    """
    rows = []
    for name in models:
        slug = name.replace(":", "__")
        report = json.loads((REPORTS / "metrics" / f"{slug}.json").read_text())
        base_t = np.array([report["thresholds"][v] for v in vocab])
        p = np.load(ARTIFACTS / slug / "predictions.npz", allow_pickle=True)
        ids, proba = p["shift_ids"], p["shift"]
        y = label_matrix(df.set_index("id").loc[ids]["labels"], vocab)
        bucket = np.array([int(hashlib.sha1(i.encode()).hexdigest()[:8], 16) % 100 for i in ids])
        pool, held = np.flatnonzero(bucket < 20), np.flatnonzero(bucket >= 20)
        pool = pool[np.argsort([hashlib.sha1(ids[i].encode()).hexdigest() for i in pool])]
        before = summary(y[held], proba[held], decide(proba[held], base_t))
        rows.append(
            {
                "model": name,
                "labelled Horizon projects": 0,
                "micro_f1": before["micro_f1"],
                "macro_f1": before["macro_f1"],
                "precision": before["micro_precision"],
                "recall": before["micro_recall"],
            }
        )
        for n in (100, 250, 500, 1000, len(pool)):
            idx = pool[:n]
            t = tune_thresholds(y[idx], proba[idx], min_positives=20)
            m = summary(y[held], proba[held], decide(proba[held], t))
            rows.append(
                {
                    "model": name,
                    "labelled Horizon projects": n,
                    "micro_f1": m["micro_f1"],
                    "macro_f1": m["macro_f1"],
                    "precision": m["micro_precision"],
                    "recall": m["micro_recall"],
                }
            )
    out = pd.DataFrame(rows).round(3)
    out.attrs["held_out"] = len(held)
    return out


def uncertainty(df: pd.DataFrame, vocab: list[str], models: list[str]) -> str:
    """95% bootstrap intervals, paired model differences and shift drops."""
    boots: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    points: dict[tuple[str, str], dict[str, float]] = {}
    for name in models:
        slug = name.replace(":", "__")
        report = json.loads((REPORTS / "metrics" / f"{slug}.json").read_text())
        thresholds = np.array([report["thresholds"][v] for v in vocab])
        p = np.load(ARTIFACTS / slug / "predictions.npz", allow_pickle=True)
        for split, seed in (("test", 101), ("shift", 202)):
            y = label_matrix(df.set_index("id").loc[p[split + "_ids"]]["labels"], vocab)
            pred = decide(p[split], thresholds)
            # Same seed per split for every model -> paired replicates.
            boots[(name, split)] = bootstrap_f1(y, pred, n_boot=1000, seed=seed)
            points[(name, split)] = report["results"][split]

    def fmt(point: float, samples: np.ndarray) -> str:
        lo, hi = interval(samples)
        return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = [
        "# Uncertainty",
        "",
        "95% percentile intervals from 1,000 bootstrap resamples of projects.",
        "Within a split every model is resampled with the same project draws, so",
        "differences between models are paired.",
        "",
        "| model | split | micro-F1 | macro-F1 |",
        "| --- | --- | --- | --- |",
    ]
    for name in models:
        for split in ("test", "shift"):
            b, pt = boots[(name, split)], points[(name, split)]
            lines.append(
                f"| {name} | {split} | {fmt(pt['micro_f1'], b['micro_f1'])} | "
                f"{fmt(pt['macro_f1'], b['macro_f1'])} |"
            )
    lines += [
        "",
        "## Paired differences (row model minus column model)",
        "",
        "| comparison | split | micro-F1 difference | macro-F1 difference |",
        "| --- | --- | --- | --- |",
    ]
    best = models[0]
    for other in models[1:]:
        for split in ("test", "shift"):
            a, b = boots[(best, split)], boots[(other, split)]
            pa, pb = points[(best, split)], points[(other, split)]
            lines.append(
                f"| {best} - {other} | {split} | "
                f"{fmt(pa['micro_f1'] - pb['micro_f1'], a['micro_f1'] - b['micro_f1'])} | "
                f"{fmt(pa['macro_f1'] - pb['macro_f1'], a['macro_f1'] - b['macro_f1'])} |"
            )
    lines += [
        "",
        "## Shift drop (Horizon Europe minus H2020 test, independent samples)",
        "",
        "| model | micro-F1 change | macro-F1 change |",
        "| --- | --- | --- |",
    ]
    for name in models:
        t, s = boots[(name, "test")], boots[(name, "shift")]
        pt, ps = points[(name, "test")], points[(name, "shift")]
        lines.append(
            f"| {name} | {fmt(ps['micro_f1'] - pt['micro_f1'], s['micro_f1'] - t['micro_f1'])} | "
            f"{fmt(ps['macro_f1'] - pt['macro_f1'], s['macro_f1'] - t['macro_f1'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    board = leaderboard(REPORTS)
    (REPORTS / "leaderboard.md").write_text(board.to_markdown(index=False) + "\n")
    fig_leaderboard(board[board["model"] != "prior"])

    df, vocab = load(DATA)
    train_support = df[df["split"] == "train"]["labels"].explode().value_counts()
    best = best_model(board)
    slug = best.replace(":", "__")
    rep = json.loads((REPORTS / "metrics" / f"{slug}.json").read_text())
    test_pl = pd.DataFrame(rep["results"]["test_per_label"])
    shift_pl = pd.DataFrame(rep["results"]["shift_per_label"])
    fig_support(test_pl, train_support, best)
    fig_shift(test_pl, shift_pl, best)

    curves = {}
    test_ids = None
    for name in board["model"]:
        if name == "prior":
            continue
        s = name.replace(":", "__")
        p = np.load(ARTIFACTS / s / "predictions.npz", allow_pickle=True)
        test_ids = p["test_ids"]
        y = label_matrix(df.set_index("id").loc[test_ids]["labels"], vocab)
        conf, acc, _ = reliability_curve(y, p["test"])
        curves[name] = (conf, acc)
    fig_reliability(curves)

    (REPORTS / "error_analysis.md").write_text(error_analysis(slug, df, vocab))

    models = [m for m in board["model"] if m != "prior"]
    (REPORTS / "uncertainty.md").write_text(uncertainty(df, vocab, models))
    adapt = adaptation(df, vocab, models)
    (REPORTS / "adaptation.md").write_text(
        "# Threshold re-tuning on a small labelled Horizon Europe sample\n\n"
        f"Evaluated on {adapt.attrs['held_out']:,} Horizon projects never used for "
        "tuning. Row 0 uses thresholds tuned on H2020 validation.\n\n"
        + adapt.to_markdown(index=False)
        + "\n"
    )
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
