import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI

from apps.backend.interfaces.api.routers.tasks import build_router
from apps.backend.interfaces.api.task_registry import (
    TaskEntry,
    get_task,
    register_task,
    tasks,
    tasks_lock,
)


def _clear_tasks() -> None:
    with tasks_lock:
        tasks.clear()


def _parse_sse(body: str) -> list[tuple[int, dict]]:
    events: list[tuple[int, dict]] = []
    for block in body.split("\n\n"):
        b = block.strip()
        if not b:
            continue
        event_id: int | None = None
        payload: dict | None = None
        for line in b.splitlines():
            if line.startswith("id:"):
                event_id = int(line.split(":", 1)[1].strip())
            if line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
        if event_id is None:
            raise AssertionError(f"missing id line in block: {b!r}")
        if payload is None or not isinstance(payload, dict):
            raise AssertionError(f"missing/invalid data payload in block: {b!r}")
        events.append((event_id, payload))
    return events


@dataclass(slots=True)
class _DummyBackendState:
    def stop_generating(self) -> None:  # pragma: no cover
        return


def test_task_sse_includes_ids_and_terminal_events():
    async def main() -> None:
        _clear_tasks()

        app = FastAPI()
        app.include_router(build_router(codex_root=Path("."), backend_state=_DummyBackendState()))

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop, max_buffered_events=10, max_buffered_bytes=1024 * 1024)
        task_id = "task(test-sse-ids)"
        register_task(task_id, entry)

        entry.push_event({"type": "status", "stage": "queued"})
        entry.push_event({"type": "status", "stage": "running"})
        entry.result = {"status": "completed", "result": {"images": [], "info": {"ok": True}}}
        entry.mark_finished(success=True)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res_all = await client.get(f"/api/tasks/{task_id}/events?after=0")
            assert res_all.status_code == 200
            events_all = _parse_sse(res_all.text)

            res_after = await client.get(f"/api/tasks/{task_id}/events?after=2")
            assert res_after.status_code == 200
            events_after = _parse_sse(res_after.text)

        assert [eid for eid, _ in events_all] == [1, 2, 3, 4]
        assert [p.get("type") for _, p in events_all] == ["status", "status", "result", "end"]

        assert [eid for eid, _ in events_after] == [3, 4]
        assert [p.get("type") for _, p in events_after] == ["result", "end"]

        # Ensure the task was registered for the duration of the stream.
        assert get_task(task_id) is not None
        _clear_tasks()

    asyncio.run(main())


def test_task_sse_gap_event_on_truncated_history():
    async def main() -> None:
        _clear_tasks()

        app = FastAPI()
        app.include_router(build_router(codex_root=Path("."), backend_state=_DummyBackendState()))

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop, max_buffered_events=3, max_buffered_bytes=1024 * 1024)
        task_id = "task(test-sse-gap)"
        register_task(task_id, entry)

        # Push 5 events; buffer keeps only the newest 3.
        for _ in range(5):
            entry.push_event({"type": "status", "stage": "running"})
        entry.result = {"status": "completed", "result": {"images": [], "info": {}}}
        entry.mark_finished(success=True)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get(f"/api/tasks/{task_id}/events?after=0")
            assert res.status_code == 200
            events = _parse_sse(res.text)

        # gap id == oldest-1 (oldest is 3 when only events 3..5 remain)
        assert events[0][0] == 2
        assert events[0][1]["type"] == "gap"
        assert events[0][1]["oldest_event_id"] == 3
        assert events[0][1]["newest_event_id"] == 5

        # Then we replay buffered events and finish with result/end.
        assert [eid for eid, _ in events[1:]] == [3, 4, 5, 6, 7]
        assert [p.get("type") for _, p in events[1:]] == ["status", "status", "status", "result", "end"]

        _clear_tasks()

    asyncio.run(main())


def test_task_entry_wait_for_event_or_done_wakes():
    async def main() -> None:
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop, max_buffered_events=10, max_buffered_bytes=1024 * 1024)

        waiter = asyncio.create_task(entry.wait_for_event_or_done(after_event_id=0))
        await asyncio.sleep(0)
        assert not waiter.done()

        entry.push_event({"type": "status", "stage": "queued"})
        await asyncio.wait_for(waiter, timeout=1.0)

    asyncio.run(main())


def test_task_entry_schedule_cleanup_unregisters_task():
    async def main() -> None:
        _clear_tasks()
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop, max_buffered_events=1, max_buffered_bytes=1024)
        task_id = "task(test-cleanup)"
        register_task(task_id, entry)
        entry.mark_finished(success=True)
        entry.schedule_cleanup(task_id, delay=0.01)
        await asyncio.sleep(0.05)
        assert get_task(task_id) is None

    asyncio.run(main())

