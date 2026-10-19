"""Build the labelled dataset: one row per project, a list of fields per row.

Design decisions (see README, "Data"):

* **Label = second level of EuroSciVoc** ("natural sciences/chemical sciences").
  Level 1 has six classes and is too coarse to be useful; level 3+ has hundreds of
  long-tail classes with a few dozen examples each.
* **Leakage-safe split.** Projects whose normalised text is identical (resubmitted
  fellowships, follow-up grants) share a *group*; groups, not rows, are assigned to
  train/validation/test. The assignment is a hash of the group key, so it is stable
  when the export is refreshed and does not depend on row order.
* **Temporal shift set.** Horizon Europe (2021-2027) projects are never used for
  training or model selection. Any Horizon project whose text also appears in H2020
  is removed from it so the shift test measures generalisation, not memory.
* **Label vocabulary from training data only**: a field is kept if it has at least
  ``min_support`` training projects.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fieldclf.cordis import Archive, read_field_labels, read_projects

SPLITS = ("train", "val", "test", "shift")
MIN_TEXT_CHARS = 200


def field_of(path: str, level: int = 2) -> str | None:
    """Truncate a taxonomy path: level 2 of 'natural sciences/physical sciences/optics'
    is 'natural sciences/physical sciences'."""
    parts = [p for p in path.strip("/").split("/") if p]
    return "/".join(parts[:level]) if len(parts) >= level else None


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def group_key(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()


def assign_split(key: str, val_pct: int = 10, test_pct: int = 10) -> str:
    bucket = int(key[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def _projects_with_fields(archive: Archive, level: int) -> pd.DataFrame:
    projects = read_projects(archive)
    labels = read_field_labels(archive)
    labels["field"] = labels["path"].map(lambda p: field_of(p, level))
    fields = (
        labels.dropna(subset=["field"])
        .groupby("project_id")["field"]
        .agg(lambda s: sorted(set(s)))
        .rename("fields")
    )
    df = projects.merge(fields, left_on="id", right_index=True, how="inner")
    df["text"] = (df["title"].fillna("") + ". " + df["objective"].fillna("")).str.strip()
    return pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "programme": archive.programme,
            "start_year": pd.to_numeric(df["startDate"].str[:4], errors="coerce").astype("Int64"),
            "human_validated": df["Human-validated"].eq("true"),
            "text": df["text"].astype(str),
            "fields": df["fields"],
        }
    )


@dataclass
class BuildReport:
    sources: dict[str, str]
    level: int
    min_support: int
    counts: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)


def build(
    train_archive: Archive,
    shift_archive: Archive,
    level: int = 2,
    min_support: int = 100,
) -> tuple[pd.DataFrame, BuildReport]:
    report = BuildReport(
        sources={a.programme: a.sha256() for a in (train_archive, shift_archive)},
        level=level,
        min_support=min_support,
    )

    base = _projects_with_fields(train_archive, level)
    shift = _projects_with_fields(shift_archive, level)

    for name, df in (("base", base), ("shift", shift)):
        short = df["text"].str.len() < MIN_TEXT_CHARS
        report.dropped[f"{name}_short_text"] = int(short.sum())
    base = base[base["text"].str.len() >= MIN_TEXT_CHARS].copy()
    shift = shift[shift["text"].str.len() >= MIN_TEXT_CHARS].copy()

    base["group"] = base["text"].map(group_key)
    shift["group"] = shift["text"].map(group_key)
    report.counts["base_duplicate_rows"] = int(base["group"].duplicated(keep=False).sum())

    seen = set(base["group"])
    overlap = shift["group"].isin(seen)
    report.dropped["shift_text_seen_in_base"] = int(overlap.sum())
    shift = shift[~overlap].copy()

    base["split"] = base["group"].map(assign_split)
    shift["split"] = "shift"

    support = base.loc[base["split"] == "train", "fields"].explode().value_counts()
    vocab = sorted(support[support >= min_support].index)
    report.labels = vocab

    df = pd.concat([base, shift], ignore_index=True)
    df["labels"] = df["fields"].map(lambda fs: [f for f in fs if f in vocab])
    no_label = df["labels"].map(len) == 0
    report.dropped["no_label_in_vocab"] = int(no_label.sum())
    df = df[~no_label].drop(columns=["fields"]).reset_index(drop=True)

    for s in SPLITS:
        report.counts[s] = int((df["split"] == s).sum())
    report.counts["human_validated_test"] = int(
        (df["human_validated"] & df["split"].isin(["test", "shift"])).sum()
    )
    return df, report


def label_matrix(labels: pd.Series, vocab: list[str]) -> np.ndarray:
    index = {f: i for i, f in enumerate(vocab)}
    y = np.zeros((len(labels), len(vocab)), dtype=np.int8)
    for row, fs in enumerate(labels):
        for f in fs:
            y[row, index[f]] = 1
    return y


def save(df: pd.DataFrame, report: BuildReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "dataset.parquet", index=False)
    (out_dir / "dataset.json").write_text(json.dumps(asdict(report), indent=2) + "\n")


def load(out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(out_dir / "dataset.parquet")
    df["labels"] = df["labels"].map(list)
    vocab = json.loads((out_dir / "dataset.json").read_text())["labels"]
    return df, vocab
