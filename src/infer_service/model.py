"""ONNX Runtime classifier wrapper."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from infer_service.preprocess import softmax


class OnnxClassifier:
    def __init__(self, model_path: str | Path, labels: list[str], *, intra_op_threads: int = 0,
                 providers: list[str] | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), options,
                                            providers=providers or ["CPUExecutionProvider"])
        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_shape = [d if isinstance(d, int) else None for d in meta.shape]
        self.output_name = self.session.get_outputs()[0].name
        self.labels = labels
        self.model_path = str(model_path)
        self.providers = self.session.get_providers()
        self.last_infer_ms: float = 0.0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Run one forward pass on an (N, 3, H, W) float32 batch; returns (N, classes) probabilities."""
        started = time.perf_counter()
        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        self.last_infer_ms = (time.perf_counter() - started) * 1000
        return softmax(np.asarray(logits, dtype=np.float32))

    def top_k(self, probs: np.ndarray, k: int = 5) -> list[dict]:
        k = min(k, probs.shape[-1])
        idx = np.argpartition(-probs, k - 1)[:k]
        idx = idx[np.argsort(-probs[idx])]
        return [{"index": int(i), "label": self.labels[i] if i < len(self.labels) else str(i),
                 "probability": round(float(probs[i]), 6)} for i in idx]

    def warmup(self, rounds: int = 3) -> None:
        shape = [d or 1 for d in self.input_shape]
        dummy = np.zeros(shape, dtype=np.float32)
        for _ in range(rounds):
            self.predict(dummy)


def load_labels(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text("utf-8").splitlines() if ln.strip()]
    # synset.txt lines look like "n01440764 tench, Tinca tinca": keep the human-readable part.
    return [ln.split(" ", 1)[1] if ln.startswith("n0") and " " in ln else ln for ln in lines]
