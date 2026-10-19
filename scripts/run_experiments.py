"""Reproduce every row of the README leaderboard (about 2 hours on an Apple M4, most of it fine-tuning)."""

import sys
from pathlib import Path

from fieldclf.experiment import leaderboard, run
from fieldclf.finetune import FineTuned
from fieldclf.models import EmbeddingLinear, PriorModel, TfidfLinear

DATA, REPORTS, ARTIFACTS, CACHE = (
    Path(p) for p in ("data/processed", "reports", "artifacts", "data/cache")
)

MODELS = {
    "prior": PriorModel,
    "tfidf": TfidfLinear,
    "embed": lambda: EmbeddingLinear("BAAI/bge-small-en-v1.5", CACHE),
    "finetune": lambda: FineTuned("BAAI/bge-small-en-v1.5"),
}

if __name__ == "__main__":
    for key in sys.argv[1:] or list(MODELS):
        report = run(MODELS[key](), DATA, REPORTS, ARTIFACTS)
        print(report["model"], report["results"]["test"], flush=True)
    print(leaderboard(REPORTS).to_string(index=False))
