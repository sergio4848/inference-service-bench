import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from infer_service.api import create_app
from infer_service.batcher import MicroBatcher
from infer_service.config import Settings
from infer_service.preprocess import image_to_tensor
from tests.conftest import N_CLASSES


@pytest.fixture(scope="module")
def client(synthetic_model):
    model_path, labels_path = synthetic_model
    app = create_app(Settings(model_path=model_path, labels_path=labels_path, batch_max=4, batch_wait_ms=5))
    with TestClient(app) as c:
        yield c


def test_preprocess_shapes_and_normalisation(jpeg_bytes):
    tensor = image_to_tensor(jpeg_bytes, 224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert -3 < float(tensor.mean()) < 3


def test_ready_and_model_info(client):
    assert client.get("/readyz").status_code == 200
    info = client.get("/v1/model").json()
    assert info["input_shape"] == [None, 3, 224, 224]
    assert info["labels"] == N_CLASSES
    assert info["batching"] == {"max_batch": 4, "max_wait_ms": 5.0}


def test_classify_returns_top_k_probabilities(client, jpeg_bytes):
    r = client.post("/v1/classify", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == 5
    assert preds[0]["probability"] >= preds[-1]["probability"]
    assert all(p["label"].startswith("class-") for p in preds)
    assert r.json()["timings_ms"]["queue_and_infer"] >= 0


def test_classify_rejects_garbage(client):
    r = client.post("/v1/classify", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_metrics_exposed(client, jpeg_bytes):
    client.post("/v1/classify", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")})
    text = client.get("/metrics").text
    assert "infer_forward_latency_seconds" in text
    assert "infer_batch_size" in text
    assert "infer_model_loaded 1.0" in text


def test_missing_model_is_not_ready(tmp_path):
    app = create_app(Settings(model_path=str(tmp_path / "missing.onnx")))
    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 503
        r = c.post("/v1/classify", files={"file": ("x.jpg", b"123", "image/jpeg")})
        assert r.status_code == 503


def test_micro_batcher_coalesces_concurrent_requests():
    sizes: list[int] = []

    def predict(batch: np.ndarray) -> np.ndarray:
        sizes.append(len(batch))
        return np.tile(np.arange(N_CLASSES, dtype=np.float32), (len(batch), 1))

    async def scenario():
        batcher = MicroBatcher(predict, max_batch=8, max_wait_ms=20)
        await batcher.start()
        tensors = [np.full((3, 4, 4), i, dtype=np.float32) for i in range(6)]
        rows = await asyncio.gather(*[batcher.submit(t) for t in tensors])
        await batcher.stop()
        return rows

    rows = asyncio.run(scenario())
    assert len(rows) == 6 and all(r.shape == (N_CLASSES,) for r in rows)
    assert max(sizes) >= 2, f"expected coalesced batches, got sizes {sizes}"
    assert sum(sizes) == 6


def test_llm_gateway_unconfigured_returns_503(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client.app.state.llm._key = None
    r = client.post("/v1/llm/complete", json={"prompt": "hello"})
    assert r.status_code == 503
