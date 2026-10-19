"""End-to-end on synthetic data: build -> train -> save -> load -> serve."""

import json

import pytest
from fastapi.testclient import TestClient

from fieldclf.cordis import Archive
from fieldclf.dataset import build, save
from fieldclf.experiment import leaderboard, run
from fieldclf.models import PriorModel, TfidfLinear
from fieldclf.predictor import Classifier
from fieldclf.serve import MAX_BATCH, MAX_CHARS, create_app


@pytest.fixture
def trained(archives, tmp_path):
    df, report = build(Archive(archives[0]), Archive(archives[1]), min_support=20)
    save(df, report, tmp_path / "data")
    for model in (PriorModel(), TfidfLinear(c_grid=(4.0,))):
        run(model, tmp_path / "data", tmp_path / "reports", tmp_path / "artifacts")
    return tmp_path


def test_learns_the_synthetic_signal_and_beats_the_prior(trained):
    board = leaderboard(trained / "reports").set_index("model")
    assert board.loc["tfidf_logreg", "test:micro_f1"] > 0.9
    assert board.loc["tfidf_logreg", "shift:micro_f1"] > 0.9
    assert board.loc["prior", "test:micro_f1"] < board.loc["tfidf_logreg", "test:micro_f1"]
    report = json.loads((trained / "reports/metrics/tfidf_logreg.json").read_text())
    assert set(report["results"]) >= {"test", "shift", "test_per_label"}


def test_saved_classifier_round_trips(trained):
    clf = Classifier.load(trained / "artifacts/tfidf_logreg")
    out = clf.predict(["quantum photon laser optics plasma " * 10])[0]
    assert out["fields"] == ["natural sciences/physical sciences"]
    assert out["scores"][0]["score"] >= out["scores"][1]["score"]


@pytest.fixture
def client(trained):
    clf = Classifier.load(trained / "artifacts/tfidf_logreg")
    with TestClient(create_app(clf)) as c:
        yield c


def test_api_health_and_labels(client):
    assert client.get("/health").json() == {"status": "ok", "model": "tfidf_logreg", "labels": 3}
    assert len(client.get("/labels").json()) == 3


def test_api_predict(client):
    r = client.post(
        "/predict", json={"texts": ["market firm trade price finance " * 10], "top_k": 2}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"][0]["fields"] == ["social sciences/economics and business"]
    assert len(body["predictions"][0]["scores"]) == 2
    assert "Server-Timing" in r.headers


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"texts": []}, 422),
        ({"texts": ["x"] * (MAX_BATCH + 1)}, 422),
        ({"texts": ["   "]}, 422),
        ({"texts": ["a" * (MAX_CHARS + 1)]}, 413),
        ({"texts": ["ok"], "top_k": 0}, 422),
    ],
)
def test_api_rejects_bad_requests(client, payload, status):
    assert client.post("/predict", json=payload).status_code == status
