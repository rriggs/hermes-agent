"""A plugin completion suppressed by ``background_process_notifications: off`` is still settled.

Suppressing the wake is a delivery decision the user made; leaving the durable row pending turns it
into a replay that delivers later anyway. The TUI must claim and acknowledge exactly like the
gateway and CLI drains do, and its periodic recovery must sweep the plugin ledger as well as the
delegation ledger.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from agent import plugin_completions as pc
from gateway.session_context import clear_session_vars, set_session_vars
from tui_gateway import server

PLUGIN = "a2a"
SESSION_KEY = "agent:main:tui:dm:1"
SESSION_ID = "sess-tui"


@pytest.fixture
def bound(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: off\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = set_session_vars(platform="tui", source="tui", chat_id="1", chat_type="dm",
                              session_key=SESSION_KEY, session_id=SESSION_ID)
    try:
        yield tmp_path
    finally:
        clear_session_vars(tokens)


def _publish_event(content: str = "peer finished the task") -> dict:
    from tools.process_registry import process_registry
    target = pc.capture_completion_target(PLUGIN, session_id=SESSION_ID)
    pc.publish_completion(PLUGIN, target=target, event_id="task-1:completed",
                          content=content)
    return process_registry.completion_queue.get_nowait()


def _session(profile_home=None) -> dict:
    return {"history_lock": threading.RLock(), "running": False, "history": [], "agent": None,
            "profile_home": str(profile_home) if profile_home else None}


def test_off_acknowledges_the_durable_row_and_starts_no_turn(bound, monkeypatch):
    evt = _publish_event()
    submits: list = []
    monkeypatch.setattr(server, "_session_owns_notification_event",
                        lambda sid, session, event: True)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: submits.append(a))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    registry = SimpleNamespace(completion_queue=queue.Queue(),
                               is_completion_consumed=lambda session_id: False)
    handled = server._notif_handle_event(
        "sid", _session(bound), evt, set(), registry, lambda e: "notification text", None)

    assert handled is True
    assert submits == [], "notifications are off: no turn may start"
    row = pc._existing_completion(PLUGIN, SESSION_ID, "task-1:completed")
    assert row is not None
    assert row["delivery_state"] == "delivered", (
        "a suppressed completion left pending replays later and delivers anyway")


def test_the_recovery_hook_sweeps_the_plugin_ledger(bound, monkeypatch):
    calls: list = []
    monkeypatch.setattr(pc, "sweep_stale_plugin_completions",
                        lambda queue_, **kw: calls.append(queue_) or 1)

    target_queue = queue.Queue()
    server._notif_sweep_plugin_completions(target_queue)

    assert calls == [target_queue], (
        "the TUI's periodic recovery must re-offer stale plugin completions, not only delegations")
