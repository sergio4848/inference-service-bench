"""Closed-loop load generator.

    infer-bench --url http://localhost:8000 --image samples/dog.jpg --concurrency 1,4,16 --duration 20

For each concurrency level it runs `concurrency` workers that send requests back to back for
`duration` seconds, then reports throughput, latency percentiles and error rate, and the service's
own batch-size distribution read from /metrics. Results are written to JSON and printed as a
Markdown table so they can be pasted into a README or diffed between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


async def _worker(client: httpx.AsyncClient, url: str, files: dict | None, json_body: dict | None,
                  stop_at: float, latencies: list[float], errors: list[str]) -> None:
    while time.perf_counter() < stop_at:
        started = time.perf_counter()
        try:
            resp = await client.post(url, files=files, json=json_body)
            if resp.status_code != 200:
                errors.append(f"HTTP {resp.status_code}")
            else:
                latencies.append((time.perf_counter() - started) * 1000)
        except httpx.HTTPError as exc:
            errors.append(exc.__class__.__name__)


async def _read_batch_metrics(client: httpx.AsyncClient, base: str) -> dict:
    try:
        text = (await client.get(f"{base}/metrics")).text
    except httpx.HTTPError:
        return {}
    out = {}
    for key in ("infer_batch_size_sum", "infer_batch_size_count", "infer_forward_latency_seconds_sum",
                "infer_forward_latency_seconds_count"):
        m = re.search(rf"^{key} (\S+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


async def run_level(base: str, route: str, concurrency: int, duration: float, image: Path | None,
                    prompt: str | None) -> dict:
    latencies: list[float] = []
    errors: list[str] = []
    files = {"file": (image.name, image.read_bytes(), "image/jpeg")} if image else None
    json_body = {"prompt": prompt, "max_tokens": 64} if prompt else None
    async with httpx.AsyncClient(timeout=120) as client:
        before = await _read_batch_metrics(client, base)
        stop_at = time.perf_counter() + duration
        started = time.perf_counter()
        await asyncio.gather(*[_worker(client, f"{base}{route}", files, json_body, stop_at, latencies, errors)
                               for _ in range(concurrency)])
        elapsed = time.perf_counter() - started
        after = await _read_batch_metrics(client, base)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    batches = delta.get("infer_batch_size_count", 0)
    return {
        "concurrency": concurrency,
        "duration_s": round(elapsed, 1),
        "requests": len(latencies),
        "errors": len(errors),
        "rps": round(len(latencies) / elapsed, 1) if elapsed else 0,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0,
        },
        "avg_batch_size": round(delta.get("infer_batch_size_sum", 0) / batches, 2) if batches else None,
        "avg_forward_ms": round(1000 * delta.get("infer_forward_latency_seconds_sum", 0)
                                / max(delta.get("infer_forward_latency_seconds_count", 0), 1), 2)
        if batches else None,
        "error_kinds": sorted(set(errors)),
    }


def markdown(report: dict) -> str:
    lines = ["| concurrency | req | RPS | p50 ms | p95 ms | p99 ms | avg batch | fwd ms | errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in report["levels"]:
        lines.append(f"| {r['concurrency']} | {r['requests']} | {r['rps']} | {r['latency_ms']['p50']} | "
                     f"{r['latency_ms']['p95']} | {r['latency_ms']['p99']} | {r['avg_batch_size'] or '-'} | "
                     f"{r['avg_forward_ms'] or '-'} | {r['errors']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["vision", "llm"], default="vision")
    parser.add_argument("--image", type=Path, help="JPEG/PNG sent on every request (vision mode)")
    parser.add_argument("--prompt", default="Explain micro-batching in two sentences.",
                        help="LLM mode prompt")
    parser.add_argument("--concurrency", default="1,4,16", help="comma-separated levels")
    parser.add_argument("--duration", type=float, default=15.0, help="seconds per level")
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--label", default="", help="free text stored with the report (machine, model)")
    args = parser.parse_args(argv)

    if args.mode == "vision" and not args.image:
        parser.error("--image is required in vision mode")
    route = "/v1/classify" if args.mode == "vision" else "/v1/llm/complete"
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    results = []
    for level in levels:
        result = asyncio.run(run_level(args.url, route, level, args.duration,
                                       args.image if args.mode == "vision" else None,
                                       args.prompt if args.mode == "llm" else None))
        results.append(result)
        lat = result["latency_ms"]
        print(f"concurrency={level} rps={result['rps']} p50={lat['p50']}ms p95={lat['p95']}ms "
              f"batch={result['avg_batch_size']} errors={result['errors']}")

    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": args.mode, "url": args.url, "label": args.label,
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpu": platform.processor()},
        "levels": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"{args.mode}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2), "utf-8")
    print()
    print(markdown(report))
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
