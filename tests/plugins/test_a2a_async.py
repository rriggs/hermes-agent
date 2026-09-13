"""Tests for A2A async long-running task flow.

Covers:
  - Inbound non-blocking SendMessage accept (returnImmediately / push config)
  - Background finalization of async-accepted tasks
  - Outbound async tools: a2a_submit, a2a_get_task, a2a_await, a2a_cancel
  - SQLite task-store durability across restarts
  - Watchdog exemption for tasks with live waiters
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.platforms.a2a import adapter as adapter_mod
from plugins.platforms.a2a import protocol, tools
from plugins.platforms.a2a.adapter import A2AAdapter


def _bare_adapter(**kwargs) -> A2AAdapter:
    from gateway.config import PlatformConfig
    return A2AAdapter(PlatformConfig(enabled=True), **kwargs)


def _fake_dispatch(monkeypatch) -> None:
    """Mock run_coroutine_threadsafe so dispatch succeeds without a real loop.

    Closes the (never-awaited) handle_message coroutine to avoid resource
    warnings; the reply future is resolved directly by tests instead.
    """
    def _noop_dispatch(coro, loop):
        try:
            coro.close()
        except Exception:
            pass

    monkeypatch.setattr(adapter_mod.asyncio, "run_coroutine_threadsafe", _noop_dispatch)


def _gateway_ready(adapter: A2AAdapter) -> None:
    """Make the adapter look like its gateway loop + handler are wired up.

    _prepare_task short-circuits to a terminal FAILED task when these are
    None; tests that want a real WORKING dispatch must set them.
    """
    adapter._loop = object()          # type: ignore[attr-defined]
    adapter._message_handler = object()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Phase 1: inbound non-blocking accept
# --------------------------------------------------------------------------

class TestReturnImmediatelyDetection:
    def test_configuration_flag(self):
        assert adapter_mod._return_immediately(
            {"configuration": {"returnImmediately": True}}) is True

    def test_no_flag(self):
        assert adapter_mod._return_immediately({}) is False
        assert adapter_mod._return_immediately(
            {"configuration": {"returnImmediately": False}}) is False

    def test_push_config_implies_async(self):
        assert adapter_mod._return_immediately(
            {"configuration": {"taskPushNotificationConfig": {"url": "http://cb/x"}}}) is True

    def test_empty_push_config_does_not(self):
        assert adapter_mod._return_immediately(
            {"configuration": {"taskPushNotificationConfig": {}}}) is False


class TestAsyncAccept:
    def _dispatch_send(self, adapter, params):
        return adapter._rpc_message_send("req-1", params, "peer-alice")

    def test_return_immediately_returns_working_task_id(self, monkeypatch):
        """The RPC must come back with WORKING + taskId before the agent replies."""
        adapter = _bare_adapter()
        _gateway_ready(adapter)
        _fake_dispatch(monkeypatch)

        captured = {}

        real_prepare = adapter._prepare_task

        def fake_prepare(params, peer, agent=None):
            terminal, pending = real_prepare(params, peer, agent=agent)
            if pending is not None:
                captured["task_id"] = pending["task_id"]
            return terminal, pending

        monkeypatch.setattr(adapter, "_prepare_task", fake_prepare)

        resp = self._dispatch_send(adapter, {
            "message": protocol.text_message(protocol.ROLE_USER, "long job"),
            "configuration": {"returnImmediately": True},
        })
        result = json.loads(json.dumps(resp["result"]))
        status = (result.get("status") or {})
        assert status.get("state") == protocol.STATE_WORKING
        assert result.get("id") == captured["task_id"]
        # Task is registered and queryable while still working.
        rec = adapter.tasks.get(captured["task_id"])
        assert rec and rec["state"] == protocol.STATE_WORKING

    def test_blocking_path_unchanged_without_flag(self, monkeypatch):
        """No returnImmediately => old blocking behavior (future awaited)."""
        adapter = _bare_adapter()
        _gateway_ready(adapter)
        _fake_dispatch(monkeypatch)
        fut_holder = {}
        real_add = adapter._add_pending

        def fake_add(task_id, context_id):
            fut = real_add(task_id, context_id)
            fut_holder["fut"] = fut
            return fut

        monkeypatch.setattr(adapter, "_add_pending", fake_add)

        def resolve_later():
            deadline = time.time() + 5
            while time.time() < deadline and not fut_holder.get("fut"):
                time.sleep(0.01)
            if "fut" in fut_holder:
                fut_holder["fut"].set_result((protocol.STATE_COMPLETED, "done!"))

        threading.Thread(target=resolve_later, daemon=True).start()
        resp = self._dispatch_send(adapter, {
            "message": protocol.text_message(protocol.ROLE_USER, "quick job"),
        })
        state = resp["result"]["status"]["state"]
        assert state == protocol.STATE_COMPLETED

    def test_background_finalizer_completes_task_and_pushes(self, monkeypatch):
        """Async-accepted task completes when the agent's reply arrives; push fires."""
        adapter = _bare_adapter()
        _gateway_ready(adapter)
        _fake_dispatch(monkeypatch)
        pushed = threading.Event()

        def fake_push(task_id, context_id, reply, state):
            pushed.set()

        monkeypatch.setattr(adapter, "_send_push_notification", fake_push)

        resp = self._dispatch_send(adapter, {
            "message": protocol.text_message(protocol.ROLE_USER, "job"),
            "configuration": {"returnImmediately": True},
        })
        assert resp["result"]["status"]["state"] == protocol.STATE_WORKING
        task_id = resp["result"]["id"]

        # Simulate the agent finishing (as on_processing_complete would).
        adapter._resolve_task(task_id, protocol.STATE_COMPLETED, "job done")

        # The background finalizer should record the terminal state.
        deadline = time.time() + 5
        rec = None
        while time.time() < deadline:
            rec = adapter.tasks.get(task_id)
            if rec and rec["state"] in protocol.TERMINAL_STATES:
                break
            time.sleep(0.05)
        assert rec is not None and rec["state"] == protocol.STATE_COMPLETED
        assert rec["reply"] == "job done"
        assert pushed.wait(timeout=5), "push notification was not sent"

    def test_async_timeout_env(self, monkeypatch):
        monkeypatch.setenv("A2A_ASYNC_TIMEOUT", "7200")
        assert adapter_mod._async_timeout() == 7200.0
        monkeypatch.setenv("A2A_ASYNC_TIMEOUT", "bogus")
        assert adapter_mod._async_timeout() == 28800.0


# --------------------------------------------------------------------------
# Phase 3a: durable task store
# --------------------------------------------------------------------------

class TestDurableTaskStore:
    def test_terminal_tasks_survive_restart(self, tmp_path):
        db = str(tmp_path / "tasks.db")
        store1 = protocol.TaskStore(db_path=db)
        store1.create("t-keep", "ctx-1", "peer")
        store1.complete("t-keep", protocol.STATE_COMPLETED, "the answer")

        store2 = protocol.TaskStore(db_path=db)
        rec = store2.get("t-keep")
        assert rec is not None
        assert rec["state"] == protocol.STATE_COMPLETED
        assert rec["reply"] == "the answer"

    def test_nonterminal_tasks_failed_on_restart(self, tmp_path):
        db = str(tmp_path / "tasks.db")
        store1 = protocol.TaskStore(db_path=db)
        store1.create("t-live", "ctx-2", "peer")
        store1.set_state("t-live", protocol.STATE_WORKING)

        store2 = protocol.TaskStore(db_path=db)
        rec = store2.get("t-live")
        assert rec is not None
        assert rec["state"] == protocol.STATE_FAILED
        assert "restarted" in rec["reply"]

    def test_wal_mode_schema_version_and_task_metadata(self, tmp_path):
        db = str(tmp_path / "tasks.db")
        store = protocol.TaskStore(db_path=db)
        store.create("t-x", "ctx-3", "peer", direction="outbound", ownership="remote")
        con = sqlite3.connect(db)
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        version = con.execute("PRAGMA user_version").fetchone()[0]
        cols = {r[1] for r in con.execute("PRAGMA table_info(a2a_tasks)").fetchall()}
        row = con.execute("SELECT direction, ownership FROM a2a_tasks WHERE task_id = 't-x'").fetchone()
        con.close()
        assert mode == "wal"
        assert version == protocol.A2A_TASK_SCHEMA_VERSION
        assert {"task_id", "context_id", "state", "reply", "direction", "ownership", "remote_state"} <= cols
        assert row == ("outbound", "remote")

    def test_outbound_nonterminal_tasks_survive_restart(self, tmp_path):
        db = str(tmp_path / "tasks.db")
        store1 = protocol.TaskStore(db_path=db)
        store1.create("t-remote", "ctx-remote", "peer", direction="outbound", ownership="remote")
        store1.set_state("t-remote", protocol.STATE_WORKING)

        store2 = protocol.TaskStore(db_path=db)
        rec = store2.get("t-remote")
        assert rec is not None
        assert rec["state"] == protocol.STATE_WORKING
        assert rec["direction"] == "outbound"
        assert rec["ownership"] == "remote"

    def test_outbound_terminal_state_updates_remote_state(self, tmp_path):
        db = str(tmp_path / "tasks.db")
        store = protocol.TaskStore(db_path=db)
        store.create("t-terminal", "ctx", "peer", direction="outbound", ownership="remote")
        store.set_state("t-terminal", protocol.STATE_WORKING)
        store.complete("t-terminal", protocol.STATE_COMPLETED, "done")
        rec = store.get("t-terminal")
        assert rec is not None
        assert rec["remote_state"] == protocol.STATE_COMPLETED

    def test_trim_removes_from_db(self, tmp_path):
        db = str(tmp_path / "tasks.db")
        store = protocol.TaskStore(db_path=db)
        for i in range(store._MAX_TERMINAL + 5):
            tid = f"t-{i}"
            store.create(tid, "ctx", "peer")
            store.complete(tid, protocol.STATE_COMPLETED, "")
        con = sqlite3.connect(db)
        n = con.execute("SELECT COUNT(*) FROM a2a_tasks").fetchone()[0]
        con.close()
        assert n <= store._MAX_TERMINAL

    def test_corrupt_db_is_not_fatal(self, tmp_path):
        db = tmp_path / "tasks.db"
        db.write_text("this is not a database")
        store = protocol.TaskStore(db_path=str(db))  # must not raise
        store.create("t-y", "ctx", "peer")           # in-memory still works
        assert store.get("t-y") is not None


# --------------------------------------------------------------------------
# Phase 3b: watchdog exemption
# --------------------------------------------------------------------------

class TestWatchdogExemption:
    def test_protected_task_not_reaped(self):
        store = protocol.TaskStore(db_path="")  # empty path disables persistence? no — falls back to default
        # Use an explicit temp db instead.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            store = protocol.TaskStore(db_path=os.path.join(td, "t.db"))
            store.create("t-waited", "ctx", "peer")
            failed = store.fail_orphans(0, protected={"t-waited"})
            assert failed == []
            assert store.get("t-waited")["state"] != protocol.STATE_FAILED

    def test_unprotected_task_reaped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            store = protocol.TaskStore(db_path=os.path.join(td, "t.db"))
            store.create("t-alone", "ctx", "peer")
            failed = store.fail_orphans(0)
            assert failed == ["t-alone"]
            assert store.get("t-alone")["state"] == protocol.STATE_FAILED

    def test_watchdog_loop_respects_pending_futures(self, monkeypatch):
        adapter = _bare_adapter()
        fut = adapter._add_pending("t-pending", "ctx-pending")
        reaped = []

        def fake_fail_orphans(timeout, protected=None):
            reaped.append(set(protected or set()))
            return []

        monkeypatch.setattr(adapter.tasks, "fail_orphans", fake_fail_orphans)
        monkeypatch.setattr(adapter_mod, "_WATCHDOG_INTERVAL", 0.01)
        stop = threading.Event()
        monkeypatch.setattr(adapter, "_watchdog_stop", stop)
        thread = threading.Thread(target=adapter._watchdog_loop, daemon=True)
        thread.start()
        # Wait for at least one watchdog pass.
        deadline = time.time() + 5
        while not reaped and time.time() < deadline:
            time.sleep(0.05)
        stop.set()
        thread.join(timeout=5)
        assert reaped and "t-pending" in reaped[0]


# --------------------------------------------------------------------------
# Phase 2: outbound async tools
# --------------------------------------------------------------------------

class _PeerServer(BaseHTTPRequestHandler):
    """Scriptable fake peer: responds per-method from the class-level table."""
    responses = {}   # method -> result payload (or callable(params) -> result)
    seen = []        # list of (method, params)

    def log_message(self, *a):  # noqa: N802
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({"name": "fake-peer"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        method = req.get("method", "")
        params = req.get("params", {})
        type(self).seen.append((method, params))
        handler = type(self).responses.get(method)
        result = handler(params) if callable(handler) else handler
        if isinstance(result, dict) and "error" in result:
            body = json.dumps({"jsonrpc": "2.0", "id": req.get("id"),
                               "error": result["error"]}).encode()
        else:
            body = json.dumps({"jsonrpc": "2.0", "id": req.get("id"),
                               "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def peer_server(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), _PeerServer)
    port = httpd.server_address[1]
    url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(tools, "_load_config",
                        lambda: {"a2a_agents": {"peer": {"url": url}}})
    yield url
    httpd.shutdown()
    httpd.server_close()


def _working_task(task_id="task-async-1", ctx="ctx-a"):
    return protocol.build_task(task_id, ctx, protocol.STATE_WORKING)


def _done_task(task_id="task-async-1", ctx="ctx-a", text="finished work"):
    return protocol.build_task(task_id, ctx, protocol.STATE_COMPLETED, text)


class TestA2ASubmit:
    def test_submit_returns_task_id(self, peer_server):
        _PeerServer.responses.clear(); _PeerServer.seen.clear()
        _PeerServer.responses["SendMessage"] = lambda p: _working_task()
        out = tools.a2a_submit({"agent": "peer", "message": "build it"})
        assert "task-async-1" in out
        assert "a2a_get_task" in out
        method, params = _PeerServer.seen[0]
        assert method == "SendMessage"
        assert params["configuration"]["returnImmediately"] is True

    def test_submit_with_synchronous_peer_returns_reply(self, peer_server):
        """Peer ignores returnImmediately and answers completed — show reply."""
        _PeerServer.responses.clear()
        _PeerServer.responses["SendMessage"] = lambda p: _done_task(text="instant answer")
        out = tools.a2a_submit({"agent": "peer", "message": "hi"})
        assert "instant answer" in out

    def test_submit_error_passthrough(self, peer_server, monkeypatch):
        _PeerServer.responses.clear()
        _PeerServer.responses["SendMessage"] = {"error": {"code": -32000, "message": "nope"}}
        out = tools.a2a_submit({"agent": "peer", "message": "hi"})
        assert "Error" in out and "nope" in out


class TestA2AGetTaskAndAwait:
    def test_get_task_snapshot(self, peer_server):
        _PeerServer.responses.clear(); _PeerServer.seen.clear()
        _PeerServer.responses["GetTask"] = lambda p: _done_task(text="all done")
        out = tools.a2a_get_task({"agent": "peer", "task_id": "task-async-1"})
        assert "completed" in out
        assert "all done" in out
        method, params = _PeerServer.seen[0]
        assert method == "GetTask"

    def test_await_polls_until_terminal(self, peer_server, monkeypatch):
        _PeerServer.responses.clear()
        calls = {"n": 0}

        def get_task(params):
            calls["n"] += 1
            if calls["n"] < 3:
                return _working_task()
            return _done_task(text="poll result")

        _PeerServer.responses["GetTask"] = get_task
        monkeypatch.setattr(tools.time, "sleep", lambda s: None)
        out = tools.a2a_await({"agent": "peer", "task_id": "task-async-1",
                               "timeout": 30, "interval": 1})
        assert calls["n"] >= 3
        assert "poll result" in out

    def test_await_times_out_still_working(self, peer_server, monkeypatch):
        _PeerServer.responses.clear()
        _PeerServer.responses["GetTask"] = lambda p: _working_task()
        monkeypatch.setattr(tools.time, "sleep", lambda s: None)
        # Advance the clock past the deadline on the first poll so the loop
        # terminates rather than spinning forever.
        start = time.time()
        clock = {"now": start}

        def fake_time():
            clock["now"] += 60
            return clock["now"]

        monkeypatch.setattr(tools.time, "time", fake_time)
        out = tools.a2a_await({"agent": "peer", "task_id": "task-async-1", "timeout": 10})
        assert "still working" in out

    def test_cancel(self, peer_server):
        _PeerServer.responses.clear(); _PeerServer.seen.clear()
        _PeerServer.responses["CancelTask"] = lambda p: _done_task(
            task_id="task-async-1", text="")
        # build_task with canceled state
        _PeerServer.responses["CancelTask"] = lambda p: protocol.build_task(
            "task-async-1", "ctx-a", protocol.STATE_CANCELED)
        out = tools.a2a_cancel({"agent": "peer", "task_id": "task-async-1"})
        assert "canceled" in out
        method, params = _PeerServer.seen[0]
        assert method == "CancelTask"


class TestToolRegistration:
    def test_new_tools_registered_in_schemas(self):
        from plugins.platforms.a2a.tools import _SCHEMAS, _HANDLERS
        for name in ("a2a_submit", "a2a_get_task", "a2a_await", "a2a_cancel"):
            assert name in _SCHEMAS
            assert name in _HANDLERS
            fn = _SCHEMAS[name]["function"]
            assert fn["name"] == name
            assert "required" in fn["parameters"]
