"""Synthetic CORDIS archives that reproduce the real export format, quirks included."""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import pytest

PROJECT_COLUMNS = [
    "id", "acronym", "status", "title", "startDate", "endDate", "totalCost",
    "ecMaxContribution", "topics", "ecSignatureDate", "frameworkProgramme", "masterCall",
    "subCall", "fundingScheme", "nature", "objective", "contentUpdateDate", "rcn", "grantDoi",
    "keywords", "Human-validated", "legalBasis",
]  # fmt: skip

VOCAB = {
    "natural sciences/physical sciences": ["laser", "photon", "quantum", "optics", "plasma"],
    "natural sciences/biological sciences": ["protein", "cell", "gene", "enzyme", "genome"],
    "social sciences/economics and business": ["market", "firm", "trade", "price", "finance"],
}
FILLER = [
    *("the", "project", "will", "develop", "study"),
    *("new", "approach", "results", "europe", "research"),
]


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def project_row(pid: str, title: str, objective: str, year: int, hv: str = "false") -> str:
    values = dict.fromkeys(PROJECT_COLUMNS, "")
    values |= {
        "id": pid,
        "title": title,
        "objective": objective,
        "startDate": f"{year}-01-01",
        "Human-validated": hv,
    }
    return ";".join(_quote(values[c]) for c in PROJECT_COLUMNS)


def write_archive(path: Path, rows: list[str], labels: list[tuple[str, str]]) -> Path:
    header = ";".join(_quote(c) for c in PROJECT_COLUMNS)
    esv_header = (
        '"euroSciVocCode";"euroSciVocPath";"euroSciVocTitle";"euroSciVocDescription";"projectID"'
    )
    esv = [f'"/1/2/3";"{p}";"x";"";"{pid}"' for pid, p in labels]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("project.csv", "\n".join([header, *rows]) + "\n")
        z.writestr("euroSciVoc.csv", "\n".join([esv_header, *esv]) + "\n")
    return path


def synthetic_projects(n: int, seed: int, id_offset: int, year: int):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    rows, labels = [], []
    fields = list(VOCAB)
    for i in range(n):
        pid = str(id_offset + i)
        chosen = rng.sample(fields, k=rng.choice([1, 1, 2]))
        words = [w for f in chosen for w in rng.choices(VOCAB[f], k=12)]
        words += rng.choices(FILLER, k=40)
        rng.shuffle(words)
        objective = " ".join(words) + " " + f"(ref {pid})."
        rows.append(project_row(pid, f"Project {pid}", objective, year))
        labels += [(pid, f + "/subfield") for f in chosen]
    return rows, labels


@pytest.fixture
def archives(tmp_path: Path) -> tuple[Path, Path]:
    rows, labels = synthetic_projects(600, seed=1, id_offset=100_000, year=2016)
    # A real-world quirk: an objective that starts with a quoted phrase, exported
    # with only two leading quote characters.
    rows.append(
        '"999999";"";"";"Quirk";"2017-01-01";"";"";"";"";"";"";"";"";"";"";'
        '""Quoted start"" then protein; cell gene enzyme genome protein cell gene enzyme genome '
        "protein cell gene enzyme genome protein cell gene enzyme genome protein cell gene enzyme "
        'genome protein cell gene enzyme genome study new approach results europe research";'
        '"";"";"";"";"true";""'
    )
    labels.append(("999999", "natural sciences/biological sciences/x"))
    base = write_archive(tmp_path / "H2020.zip", rows, labels)
    rows2, labels2 = synthetic_projects(200, seed=2, id_offset=500_000, year=2023)
    shift = write_archive(tmp_path / "HORIZON.zip", rows2, labels2)
    return base, shift
