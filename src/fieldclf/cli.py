"""Command line: fieldclf download | build | train | leaderboard | predict | serve."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
ARTIFACTS = Path("artifacts")

# Official bulk exports published on data.europa.eu (European Commission reuse notice).
SOURCES = {
    "H2020": "https://cordis.europa.eu/data/cordis-h2020projects-csv.zip",
    "HORIZON": "https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip",
}


@app.command()
def download(force: bool = False) -> None:
    """Fetch the CORDIS exports and record their checksums."""
    RAW.mkdir(parents=True, exist_ok=True)
    for programme, url in SOURCES.items():
        target = RAW / f"{programme}.zip"
        if target.exists() and not force:
            typer.echo(f"{target} exists (use --force to refresh)")
            continue
        with urllib.request.urlopen(url, timeout=300) as r:
            target.write_bytes(r.read())
        meta: dict[str, Any] = {
            "programme": programme,
            "download_url": url,
            "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
        (RAW / f"{programme}.metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        typer.echo(f"{programme}: {meta['bytes']:,} bytes sha256={meta['sha256'][:12]}")


@app.command()
def build(min_support: int = 100) -> None:
    """Parse the exports and write the labelled, split dataset."""
    from fieldclf.cordis import Archive
    from fieldclf.dataset import build as build_dataset
    from fieldclf.dataset import save

    df, report = build_dataset(
        Archive(RAW / "H2020.zip"), Archive(RAW / "HORIZON.zip"), min_support=min_support
    )
    save(df, report, PROCESSED)
    typer.echo(
        json.dumps(
            {"counts": report.counts, "dropped": report.dropped, "labels": len(report.labels)},
            indent=2,
        )
    )


@app.command()
def train(
    model: Annotated[str, typer.Argument(help="prior | tfidf | embed | finetune")],
) -> None:
    """Train one model, tune thresholds on validation, evaluate on held-out sets."""
    from fieldclf.experiment import run
    from fieldclf.models import EmbeddingLinear, Model, PriorModel, TfidfLinear

    m: Model
    if model == "finetune":
        from fieldclf.finetune import FineTuned

        m = FineTuned()
    elif model == "embed":
        m = EmbeddingLinear(cache_dir=Path("data/cache"))
    elif model == "tfidf":
        m = TfidfLinear()
    elif model == "prior":
        m = PriorModel()
    else:
        raise typer.BadParameter(f"unknown model {model!r}")
    report = run(m, PROCESSED, REPORTS, ARTIFACTS)
    typer.echo(json.dumps(report["results"]["test"], indent=2))


@app.command()
def leaderboard() -> None:
    """Print the comparison table from reports/metrics/*.json."""
    from fieldclf.experiment import leaderboard as lb

    typer.echo(lb(REPORTS).to_markdown(index=False))


@app.command()
def predict(
    text: str,
    model_dir: Path = ARTIFACTS / "tfidf_logreg",
    top_k: int = 5,
) -> None:
    """Classify one text with a trained model."""
    from fieldclf.predictor import Classifier

    typer.echo(json.dumps(Classifier.load(model_dir).predict([text], top_k)[0], indent=2))


@app.command()
def serve(model_dir: Path = ARTIFACTS / "tfidf_logreg", port: int = 8000) -> None:
    """Run the HTTP API."""
    import os

    import uvicorn

    os.environ["FIELDCLF_MODEL_DIR"] = str(model_dir)
    uvicorn.run("fieldclf.serve:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
