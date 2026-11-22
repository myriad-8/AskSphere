"""End-to-end checks using tiny random-weight checkpoints (no downloads, no training)."""

import pytest
from fastapi.testclient import TestClient

from asksphere.smoke import build_smoke_checkpoints

CONTEXT = (
    "Albert Einstein was a German-born theoretical physicist who developed the theory of "
    "relativity. He received the Nobel Prize in Physics in 1921."
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    ckpt = tmp_path_factory.mktemp("ckpt")
    ext, gen = build_smoke_checkpoints(ckpt)
    mp = pytest.MonkeyPatch()
    mp.setenv("ASKSPHERE_EXTRACTIVE_DIR", str(ext))
    mp.setenv("ASKSPHERE_GENERATIVE_DIR", str(gen))
    mp.setenv("ASKSPHERE_PRETRAINED_MODEL", "")  # no network in tests
    from asksphere.api import app

    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health_reports_both_models(client):
    body = client.get("/api/health").json()
    assert body["models"]["extractive"]["loaded"] is True
    assert body["models"]["generative"]["loaded"] is True
    assert body["models"]["generative"]["variant"] == "hybrid"


def test_extractive_predict_shape(client):
    r = client.post("/predict", json={"model_type": "extractive", "question": "Who was Einstein?", "context": CONTEXT})
    assert r.status_code == 200
    body = r.json()
    assert {"answer", "null_score", "predicted_no_answer"} <= body.keys()


def test_extractive_handles_long_context_with_multiple_windows(client):
    long_context = " ".join([CONTEXT] * 40)
    r = client.post(
        "/predict",
        json={"model_type": "extractive", "question": "When did he win the prize?", "context": long_context, "max_length": 64, "doc_stride": 16},
    )
    assert r.status_code == 200


def test_generative_predict_with_and_without_gate(client):
    base = {"model_type": "generative", "question": "Who was Einstein?", "context": CONTEXT, "max_new_tokens": 8, "beam_size": 2}
    gated = client.post("/predict", json=base).json()
    assert "gate" in gated and isinstance(gated["answer"], str)
    ungated = client.post("/predict", json={**base, "enable_no_answer_gate": False}).json()
    assert "gate" not in ungated


@pytest.mark.parametrize(
    "payload",
    [
        {"model_type": "bogus", "question": "q", "context": "c"},
        {"model_type": "extractive", "question": "", "context": "c"},
        {"model_type": "extractive", "question": "   ", "context": "c"},
        {"model_type": "generative", "question": "q", "context": "c", "beam_size": 99},
    ],
)
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code in (400, 422)


def test_missing_models_give_clear_503_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("ASKSPHERE_EXTRACTIVE_DIR", str(tmp_path / "nope1"))
    monkeypatch.setenv("ASKSPHERE_GENERATIVE_DIR", str(tmp_path / "nope2"))
    monkeypatch.setenv("ASKSPHERE_PRETRAINED_MODEL", "")
    from asksphere.api import app

    with TestClient(app) as c:
        health = c.get("/api/health").json()
        assert health["models"]["extractive"]["loaded"] is False
        r = c.post("/predict", json={"model_type": "extractive", "question": "q", "context": "c"})
        assert r.status_code == 503 and "not loaded" in r.json()["detail"]


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AskSphere" in r.text


def test_pretrained_mode_plumbing_with_tiny_local_hf_model(tmp_path, monkeypatch):
    """Exercises the Hugging Face fallback path offline using a tiny random BERT QA model."""
    from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering

    ext, _ = build_smoke_checkpoints(tmp_path / "ckpt")
    tok = AutoTokenizer.from_pretrained(str(ext))
    hf_dir = tmp_path / "hf_model"
    BertForQuestionAnswering(
        BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                   intermediate_size=64, max_position_embeddings=512)
    ).save_pretrained(str(hf_dir))
    tok.save_pretrained(str(hf_dir))

    monkeypatch.setenv("ASKSPHERE_EXTRACTIVE_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("ASKSPHERE_GENERATIVE_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("ASKSPHERE_PRETRAINED_MODEL", str(hf_dir))
    from asksphere.api import app

    with TestClient(app) as c:
        assert c.get("/api/health").json()["models"]["pretrained"]["loaded"] is True
        r = c.post("/predict", json={"model_type": "pretrained", "question": "Who was Einstein?", "context": CONTEXT * 30})
        assert r.status_code == 200 and "answer" in r.json()
