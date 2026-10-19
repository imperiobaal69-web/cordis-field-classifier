import io
import zipfile

import pandas as pd
import pytest

from fieldclf.cordis import (
    Archive,
    CordisFormatError,
    parse_table,
    read_field_labels,
    read_projects,
)


def test_reads_every_record_including_malformed_quotes(archives):
    base, _ = archives
    projects = read_projects(Archive(base))
    assert len(projects) == 601
    quirk = projects.set_index("id").loc["999999"]
    assert quirk["objective"].startswith('"Quoted start" then protein; cell')
    assert quirk["Human-validated"] == "true"


def test_standard_csv_parser_loses_the_malformed_record(archives):
    """Documents why a custom reader exists: pandas skips the row (real export: 285 rows)."""
    base, _ = archives
    raw = zipfile.ZipFile(base).read("project.csv").decode()
    df = pd.read_csv(io.StringIO(raw), sep=";", dtype=str, on_bad_lines="skip")
    assert "999999" not in set(df["id"])


def test_field_labels(archives):
    base, _ = archives
    labels = read_field_labels(Archive(base))
    assert set(labels.columns) == {"project_id", "path"}
    assert labels["path"].str.contains("/").all()


def test_wrong_field_count_fails_loudly():
    text = '"id";"title"\n"1";"a";"b"\n'
    with pytest.raises(CordisFormatError, match="fields"):
        parse_table(text)


def test_duplicate_ids_fail(tmp_path):
    from tests.conftest import project_row, write_archive

    rows = [project_row("1", "a", "x", 2015), project_row("1", "b", "y", 2015)]
    path = write_archive(tmp_path / "dup.zip", rows, [])
    with pytest.raises(CordisFormatError, match="duplicate"):
        read_projects(Archive(path))
