"""Prometheus metrics and logging."""

import logging
import sys

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

REQUESTS = Counter("infer_requests_total", "HTTP requests", ["route", "status"])
REQUEST_LATENCY = Histogram("infer_request_latency_seconds", "End-to-end request latency", ["route"],
                            buckets=LATENCY_BUCKETS)
INFER_LATENCY = Histogram("infer_forward_latency_seconds", "Latency of one forward pass (per batch)",
                          buckets=LATENCY_BUCKETS)
PREPROCESS_LATENCY = Histogram("infer_preprocess_latency_seconds", "Image decode and preprocessing",
                               buckets=LATENCY_BUCKETS)
QUEUE_WAIT = Histogram("infer_queue_wait_seconds", "Time a request waited for a batch",
                       buckets=LATENCY_BUCKETS)
BATCH_SIZE = Histogram("infer_batch_size", "Images per forward pass", buckets=(1, 2, 4, 8, 16, 32))
MODEL_LOADED = Gauge("infer_model_loaded", "1 when the ONNX model is loaded")
LLM_TTFT = Histogram("infer_llm_ttft_seconds", "Time to first token from the LLM endpoint",
                     buckets=LATENCY_BUCKETS)
LLM_TOKENS_PER_S = Histogram("infer_llm_tokens_per_second", "Generation speed per completion",
                             buckets=(1, 5, 10, 20, 40, 80, 160, 320))


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "ts=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
