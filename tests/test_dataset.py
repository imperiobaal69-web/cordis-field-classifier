from collections import Counter

import numpy as np

from fieldclf.cordis import Archive
from fieldclf.dataset import assign_split, build, field_of, group_key, label_matrix, load, save
from tests.conftest import project_row, write_archive


def test_field_of_levels():
    path = "natural sciences/physical sciences/optics/laser physics"
    assert field_of(path) == "natural sciences/physical sciences"
    assert field_of(path, level=1) == "natural sciences"
    assert field_of("natural sciences") is None


def test_group_key_ignores_case_and_punctuation():
    assert group_key("Hello, World!") == group_key("hello world")
    assert group_key("hello world") != group_key("hello there")


def test_split_is_deterministic_and_roughly_80_10_10():
    keys = [group_key(str(i)) for i in range(20_000)]
    first = [assign_split(k) for k in keys]
    assert first == [assign_split(k) for k in keys]
    share = {k: v / len(keys) for k, v in Counter(first).items()}
    assert abs(share["train"] - 0.8) < 0.02
    assert abs(share["val"] - 0.1) < 0.02


def test_build_counts_and_vocab(archives, tmp_path):
    df, report = build(Archive(archives[0]), Archive(archives[1]), min_support=20)
    assert set(df["split"]) == {"train", "val", "test", "shift"}
    assert report.counts["shift"] == 200
    assert len(report.labels) == 3
    assert all(len(ls) > 0 for ls in df["labels"])
    save(df, report, tmp_path / "out")
    df2, vocab = load(tmp_path / "out")
    assert vocab == report.labels and len(df2) == len(df)


def test_duplicate_texts_never_straddle_splits(tmp_path):
    text = "protein cell gene enzyme " * 20
    rows = [project_row(str(i), "Same", text, 2016) for i in range(1, 30)]
    labels = [(str(i), "natural sciences/biological sciences/x") for i in range(1, 30)]
    base = write_archive(tmp_path / "H2020.zip", rows, labels)
    shift = write_archive(
        tmp_path / "HORIZON.zip",
        [project_row("900", "Same", text, 2023)],
        [("900", "natural sciences/biological sciences/x")],
    )
    df, report = build(Archive(base), Archive(shift), min_support=1)
    assert df.loc[df["programme"] == "H2020", "split"].nunique() == 1
    # The Horizon copy of a training text is excluded from the shift set.
    assert report.dropped["shift_text_seen_in_base"] == 1
    assert (df["split"] == "shift").sum() == 0


def test_label_matrix():
    y = label_matrix([["b"], ["a", "c"]], ["a", "b", "c"])
    assert y.tolist() == [[0, 1, 0], [1, 0, 1]]
    assert y.dtype == np.int8
