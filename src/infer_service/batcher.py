"""Micro-batching: coalesce concurrent single-image requests into one forward pass.

A request is enqueued with a future. The worker takes the first item, then keeps collecting items
until `max_batch` is reached or `max_wait_ms` has elapsed since the first item, runs one prediction
in a thread (the ONNX session releases the GIL) and resolves every future with its own row.
Throughput goes up under concurrency; a single request waits at most `max_wait_ms`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import numpy as np

from infer_service.observability import BATCH_SIZE, INFER_LATENCY, QUEUE_WAIT


class MicroBatcher:
    def __init__(self, predict: Callable[[np.ndarray], np.ndarray], *, max_batch: int = 8,
                 max_wait_ms: float = 4.0):
        self._predict = predict
        self._max_batch = max(1, max_batch)
        self._max_wait = max_wait_ms / 1000
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future, float]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.batches_run = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="micro-batcher")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, tensor: np.ndarray) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((tensor, fut, time.perf_counter()))
        return await fut

    async def _collect(self) -> list[tuple[np.ndarray, asyncio.Future, float]]:
        first = await self._queue.get()
        items = [first]
        deadline = time.perf_counter() + self._max_wait
        while len(items) < self._max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                items.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return items

    async def _loop(self) -> None:
        while True:
            items = await self._collect()
            now = time.perf_counter()
            for _, _, enqueued in items:
                QUEUE_WAIT.observe(now - enqueued)
            batch = np.stack([t for t, _, _ in items]).astype(np.float32, copy=False)
            BATCH_SIZE.observe(len(items))
            try:
                started = time.perf_counter()
                probs = await asyncio.to_thread(self._predict, batch)
                INFER_LATENCY.observe(time.perf_counter() - started)
                self.batches_run += 1
                for row, (_, fut, _) in zip(probs, items, strict=True):
                    if not fut.done():
                        fut.set_result(row)
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter
                for _, fut, _ in items:
                    if not fut.done():
                        fut.set_exception(exc)
