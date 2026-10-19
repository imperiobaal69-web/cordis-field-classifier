"""Train a tiny model on synthetic CORDIS-format data (used by CI to build the image)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldclf.cordis import Archive
from fieldclf.dataset import build, save
from fieldclf.experiment import run
from fieldclf.models import TfidfLinear
from tests.conftest import synthetic_projects, write_archive

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        base = write_archive(t / "H2020.zip", *synthetic_projects(600, 1, 100_000, 2016))
        shift = write_archive(t / "HORIZON.zip", *synthetic_projects(200, 2, 500_000, 2023))
        df, report = build(Archive(base), Archive(shift), min_support=20)
        save(df, report, t / "data")
        run(TfidfLinear(c_grid=(4.0,)), t / "data", t / "reports", out)
    print(f"smoke model written to {out / 'tfidf_logreg'}")
