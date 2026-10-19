"""Readers for the CORDIS bulk CSV exports (data.europa.eu).

The exports are semicolon-separated with every field quoted, but quotes inside
fields are not escaped consistently: an objective that begins with a quoted phrase
is written with two leading quote characters where RFC 4180 needs three. A
standard CSV parser therefore mis-splits those rows. On the H2020 export, pandas
with ``on_bad_lines="skip"`` drops 285 of 35,389 projects and returns a truncated
or shifted objective for 960 more, without raising.

Every record starts at the beginning of a line with a quoted first column of a
fixed shape (a numeric id in project.csv), and every field boundary is the
three-character sequence quote-semicolon-quote. Splitting on those two invariants
recovers all records; every record's field count is checked and the reader fails
loudly if the invariant ever breaks.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Each table's records begin with a quoted first column of a known shape.
PROJECT_RECORD = re.compile(r'\r?\n(?="\d+";")')
EUROSCIVOC_RECORD = re.compile(r'\r?\n(?="/\d)')
_SEP = '";"'


class CordisFormatError(ValueError):
    """Raised when an export no longer matches the structure this reader relies on."""


@dataclass(frozen=True)
class Archive:
    """One CORDIS programme export (a zip of CSV tables)."""

    path: Path

    @property
    def programme(self) -> str:
        return self.path.stem

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def read_text(self, member: str) -> str:
        with zipfile.ZipFile(self.path) as z:
            return z.read(member).decode("utf-8")


def _split_records(text: str, record_start: re.Pattern[str]) -> tuple[list[str], list[list[str]]]:
    header, _, body = text.partition("\n")
    columns = [c.strip('"') for c in header.strip().split(";")]
    rows: list[list[str]] = []
    for i, record in enumerate(record_start.split(body.strip())):
        if not (record.startswith('"') and record.endswith('"')):
            raise CordisFormatError(f"record {i} is not fully quoted: {record[:80]!r}")
        fields = record[1:-1].split(_SEP)
        if len(fields) != len(columns):
            raise CordisFormatError(
                f"record {i} has {len(fields)} fields, expected {len(columns)}: {record[:80]!r}"
            )
        rows.append([f.replace('""', '"').strip() for f in fields])
    return columns, rows


def parse_table(text: str, record_start: re.Pattern[str] = PROJECT_RECORD) -> pd.DataFrame:
    """Parse one CORDIS CSV table into a string-typed DataFrame."""
    columns, rows = _split_records(text, record_start)
    return pd.DataFrame(rows, columns=columns, dtype="string")


def read_projects(archive: Archive) -> pd.DataFrame:
    df = parse_table(archive.read_text("project.csv"))
    if df["id"].duplicated().any():
        raise CordisFormatError("duplicate project ids")
    return df


def read_field_labels(archive: Archive) -> pd.DataFrame:
    """EuroSciVoc labels: one row per (project, taxonomy path)."""
    df = parse_table(archive.read_text("euroSciVoc.csv"), EUROSCIVOC_RECORD)
    return df.rename(columns={"projectID": "project_id", "euroSciVocPath": "path"})[
        ["project_id", "path"]
    ]
