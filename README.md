# Inference Service Bench

A production-shaped inference service and the load generator that benchmarks it.

- **Vision:** an ONNX Runtime image classifier (reference model: MobileNetV2 from the ONNX Model
  Zoo) behind FastAPI, with **micro-batching** that coalesces concurrent requests into one forward
  pass, readiness that reflects whether the model is loaded, and Prometheus histograms for
  preprocessing, queue wait, forward-pass latency and batch size.
- **LLM:** a gateway to any OpenAI-compatible chat-completions endpoint that streams from upstream
  and reports **time to first token**, total latency and tokens per second per request.
- **Bench:** `infer-bench`, a closed-loop load generator that sweeps concurrency levels, reports RPS
  and p50/p95/p99 latency, reads the service's own batch-size metrics, and writes JSON + Markdown.
- **Delivery:** non-root multi-stage Docker image, Kubernetes manifests with an init container that
  fetches the model into an `emptyDir` (the image never contains weights), probes, resource limits,
  HPA, and CI that runs the tests against a synthetic ONNX model so no download is needed.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/get_model.py                        # MobileNetV2 (~14 MB) + ImageNet labels into ./models
uvicorn infer_service.api:app
curl -s -F file=@samples/bench.jpg localhost:8000/v1/classify | python -m json.tool
```

Actual response for the bundled synthetic sample image (shapes on a gradient, so the label is
meaningless; the timings are real):

```json
{
  "predictions": [
    {"index": 722, "label": "ping-pong ball", "probability": 0.673526},
    {"index": 714, "label": "pick, plectrum, plectron", "probability": 0.043486},
    {"index": 551, "label": "face powder", "probability": 0.02361}
  ],
  "timings_ms": {"preprocess": 13.63, "queue_and_infer": 3.84}
}
```

`GET /v1/model` shows the loaded model, input shape, execution providers, batching settings and the
number of batches run so far. `GET /readyz` returns 503 until the model is loaded, so Kubernetes
never routes traffic to a pod without weights.

### LLM gateway

```bash
export OPENAI_API_KEY=sk-...          # or INFER_LLM_API_KEY; INFER_LLM_BASE_URL for other providers
curl -s localhost:8000/v1/llm/complete -H 'content-type: application/json' \
  -d '{"prompt":"Explain micro-batching in two sentences.","max_tokens":80}'
# -> {"text": "...", "completion_tokens": 62, "ttft_ms": 412.3, "total_ms": 1830.9, "tokens_per_second": 43.7}
```

## Micro-batching

Single-image requests are put on a queue with a future. A worker takes the first item, keeps
collecting until `INFER_BATCH_MAX` items or `INFER_BATCH_WAIT_MS` have passed, runs **one**
`session.run` in a thread (ONNX Runtime releases the GIL) and resolves each future with its row.
A lone request therefore waits at most `INFER_BATCH_WAIT_MS` (default 4 ms); under concurrency the
service trades a few milliseconds of queueing for fewer, larger forward passes. Whether that trade
pays off depends on the hardware, which is why it is measured below rather than assumed: on this
CPU it does not, so batching ships **off by default** and is a one-variable switch for accelerators.

## Benchmark

```bash
uvicorn infer_service.api:app --workers 1 &
infer-bench --image samples/bench.jpg --concurrency 1,4,16,32 --duration 10 --label "laptop CPU, defaults"
INFER_BATCH_MAX=8 uvicorn infer_service.api:app --port 8001 &      # same sweep with batching on
infer-bench --url http://localhost:8001 --image samples/bench.jpg --concurrency 16 --duration 10 --label "batch 8"
```

The report contains, per concurrency level: requests, RPS, p50/p95/p99 latency, the average batch
size the service actually formed, the average forward-pass time per batch, and the error count.
Results are written to `bench/results/<mode>-<timestamp>.json` and printed as a Markdown table.

### Results (MobileNetV2-12, CPU)

Machine: Intel Core i7-13620H (10 cores / 16 threads), 64 GB, Windows 11, Python 3.11, ONNX Runtime
1.30 `CPUExecutionProvider`, one uvicorn worker. Sample: `samples/bench.jpg` (640×480 JPEG). The
load generator ran on the same machine, so it competes for CPU: treat the numbers as relative.
Raw reports are in `bench/results/`.

**Recommended defaults** (batching off, ONNX Runtime default threading, JPEG draft decoding):

| concurrency | req (10 s) | RPS | p50 ms | p95 ms | p99 ms | fwd ms/batch | errors |
|---|---|---|---|---|---|---|---|
| 1 | 889 | 88.8 | 10.7 | 16.3 | 19.3 | 3.1 | 0 |
| 4 | 1372 | 137.0 | 27.2 | 43.9 | 53.4 | 4.5 | 0 |
| 16 | 1141 | 113.5 | 91.0 | 444.0 | 813.0 | 3.9 | 0 |
| 32 | 1026 | 100.0 | 311.6 | 375.8 | 471.7 | 4.4 | 0 |

**Micro-batching on vs off**, same model, concurrency 16:

| configuration | RPS | p50 ms | p95 ms | avg batch | fwd ms/batch | fwd ms/image |
|---|---|---|---|---|---|---|
| ORT default threads, `batch_max=8` | 104.6 | 120.3 | 348.2 | 1.87 | 7.5 | 4.0 |
| ORT default threads, `batch_max=1` | 113.2 | 101.6 | 404.0 | 1.0 | 5.4 | 5.4 |
| `intra_op_threads=1`, `batch_max=8` | 98.9 | 154.2 | 223.0 | 6.54 | 60.7 | 9.3 |
| `intra_op_threads=1`, `batch_max=1` | 129.4 | 105.7 | 272.0 | 1.0 | 6.1 | 6.1 |

What the numbers say:

1. **The forward pass is not the bottleneck.** MobileNetV2 runs in 3–4 ms per image on this CPU;
   a single request spent about 10 ms of its 14 ms in JPEG decoding and resizing. Switching the
   decoder to JPEG draft mode (decode at 1/2 scale, still above the 224 px target) cut p50 at
   concurrency 1 from 14.1 ms to 10.7 ms and raised single-client throughput from 70 to 89 RPS.
2. **Micro-batching does not pay on CPU for this model.** With default threading the batcher only
   formed batches of about 2 under load and throughput fell (105 vs 113 RPS) while p50 rose by
   19 ms. Pinned to one intra-op thread, the shape of a 1-CPU pod, it formed batches of 6.5, but
   one batch took 61 ms against 6 ms for a single image: per-image cost went up and throughput
   dropped 24 %. ONNX Runtime already parallelises inside one forward pass on CPU and there is no
   large fixed per-call cost to amortise, so batching only adds queue wait. It is therefore **off
   by default** (`INFER_BATCH_MAX=1`) and kept for GPUs and NPUs, where the per-call overhead is
   what batching removes; the harness is there to check that claim on real hardware before turning
   it on.
3. **Saturation is at roughly 110–140 RPS per worker.** Past concurrency 16 latency grows with
   queue depth while errors stay at zero. The fix is more replicas (the HPA in `deploy/k8s`), not a
   deeper queue.

The sample image is synthetic (a red disc and two shapes on a gradient), so the model's top guess,
"ping-pong ball" at 0.67, says nothing about accuracy. Accuracy is a property of the model, which
this project deploys and measures but does not train.

## Tests

```bash
ruff check . && pytest
```

The test-suite builds a tiny ONNX classifier at test time (`tests/conftest.py`), so it exercises the
real serving path (decode, preprocess, batcher, ONNX Runtime session, top-k, metrics, readiness)
without downloading anything. It also checks that six concurrent requests are coalesced into
batches and that a missing model turns readiness into a 503.

## Docker and Kubernetes

```bash
docker build -t inference-service-bench:local .
docker run --rm -p 8000:8000 -v "$PWD/models:/models:ro" inference-service-bench:local

docker tag inference-service-bench:local <your-registry>/inference-service-bench:0.1.0 && docker push <your-registry>/inference-service-bench:0.1.0
# point deploy/k8s/deployment.yaml `image:` at that tag, then:
kubectl apply -f deploy/k8s/deployment.yaml
kubectl port-forward svc/infer-service 8000:80
```

The Deployment runs as a non-root user with a read-only root filesystem, fetches the model in an
init container, exposes startup/readiness/liveness probes, sets CPU and memory requests and limits
(`INFER_INTRA_OP_THREADS` is aligned with the CPU limit), and scales from 2 to 8 replicas on CPU.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `INFER_MODEL_PATH` | `models/mobilenetv2-12.onnx` | ONNX model file |
| `INFER_LABELS_PATH` | `models/synset.txt` | one label per line (synset format accepted) |
| `INFER_INPUT_SIZE` | `224` | square input size |
| `INFER_INTRA_OP_THREADS` | `0` | ONNX Runtime intra-op threads (0 = default) |
| `INFER_BATCH_MAX` / `INFER_BATCH_WAIT_MS` | `1` / `4` | micro-batch limits (`1` = batching off) |
| `INFER_LLM_BASE_URL` / `INFER_LLM_MODEL` | OpenAI / `gpt-4.1-mini` | LLM gateway target |
| `INFER_LLM_API_KEY` | – | falls back to `OPENAI_API_KEY` |

## License

MIT. The reference model and labels are downloaded from the ONNX Model Zoo (Apache-2.0) and are not
part of this repository.
