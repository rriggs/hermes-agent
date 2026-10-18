"""Gateway delivery of a plugin completion (``agent/plugin_completions.py``).

A plugin publishes a durable completion for the messaging route that invoked it. The gateway drain
is the only surface that can turn that into a real turn for the chat, so its settlement rules are a
contract, not an implementation detail:

* admitted once -- the row is claimed before injection and acknowledged after the adapter accepted
  the synthetic message, so a second drain never delivers it again;
* refused -- the claim is deferred (no delivery attempt spent) and the event is re-offered;
* notifications off -- drained and acknowledged, never injected, so the row does not stay pending

Uses a real ``GatewayRunner`` with a fake transport, against the hermetic temp ``HERMES_HOME``.
"""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import plugin_completions as pc
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session_context import clear_session_vars, set_session_vars

SESSION_KEY = "agent:main:telegram:dm:42"
SESSION_ID = "sess-1"
PLUGIN = "a2a"


class _AdmittingHandler(AsyncMock):
    """Fake transport whose successful insertion issues the production receipt."""

    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


@pytest.fixture
def bound_session():
    tokens = set_session_vars(
        platform="telegram", source="telegram", chat_id="42", chat_type="dm",
        session_key=SESSION_KEY, session_id=SESSION_ID,
    )
    try:
        yield tokens
    finally:
        clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _fresh_completion_state():
    from queue import Empty

    from tools.process_registry import process_registry
    pc._offered.clear()
    while True:
        try:
            process_registry.completion_queue.get_nowait()
        except Empty:
            break
    yield
    pc._offered.clear()


def _build_runner(monkeypatch, tmp_path, mode: str = "concise") -> GatewayRunner:
    (tmp_path / "config.yaml").write_text(
        f"display:\n  background_process_notifications: {mode}\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(
        send=AsyncMock(), handle_message=_AdmittingHandler())
    return runner


def _published_event(bound_session) -> dict:
    """Capture + publish, then hand back the queued copy (as a restarted process would see it)."""
    from queue import Empty

    from tools.process_registry import process_registry

    target = pc.capture_completion_target(PLUGIN, session_id=SESSION_ID)
    pc.publish_completion(
        PLUGIN, target=target, event_id="task-7:completed", content="Peer finished task 7.",
        metadata={"task_id": "7", "state": "completed"})
    try:
        return process_registry.completion_queue.get_nowait()
    except Empty:  # pragma: no cover - publish always offers
        raise AssertionError("publish did not offer an event on the completion queue")


def _row(event_id: str = "task-7:completed") -> dict:
    row = pc._existing_completion(PLUGIN, SESSION_KEY, event_id)
    assert row is not None, "the completion ledger row must exist"
    return row


@pytest.mark.asyncio
async def test_the_gateway_delivers_a_completion_once_and_acknowledges_the_row(
    monkeypatch, tmp_path, bound_session,
):
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.TELEGRAM]
    events = queue.Queue()
    events.put(_published_event(bound_session))
    assert _row()["delivery_state"] == "pending"

    await runner._drain_watch_notifications(events)

    adapter.handle_message.assert_awaited_once()
    synth = adapter.handle_message.await_args.args[0]
    assert "Peer finished task 7." in synth.text
    assert "plugin a2a" in synth.text
    assert _row()["delivery_state"] == "delivered"

    # A replayed copy (restart, sweep) must not start a second turn for the same event.
    from tools.async_delegation import restore_undelivered_completions
    from tools.process_registry import process_registry

    pc._offered.clear()
    assert restore_undelivered_completions(process_registry.completion_queue) == 0
    await runner._drain_watch_notifications(events)
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_a_refused_delivery_is_deferred_without_spending_an_attempt(
    monkeypatch, tmp_path, bound_session,
):
    runner = _build_runner(monkeypatch, tmp_path)
    # A handler that never issues the admission receipt is a retryable refusal.
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    events = queue.Queue()
    events.put(_published_event(bound_session))

    await runner._drain_watch_notifications(events)

    row = _row()
    assert row["delivery_state"] == "pending"
    assert row["delivery_attempts"] == 0, "a refusal must not spend a delivery attempt"
    # No immediate requeue: a deterministic refusal would spin this drain every tick. The stale
    # sweep re-offers the row once its claim lease expires.
    assert events.qsize() == 0

    from agent.plugin_completions import sweep_stale_plugin_completions
    reoffered = sweep_stale_plugin_completions(events, now=time.time() + 120.0)
    assert reoffered == 1 and events.qsize() == 1


def test_a_plugin_completion_has_a_local_admission_identity_and_card(bound_session):
    """The gateway-hosted CLI/Desktop path needs both, or local delivery is refused as invalid_params."""
    from gateway.run_notifications import _local_completion_card

    event = _published_event(bound_session)
    runner = object.__new__(GatewayRunner)
    identity = runner._completion_delivery_identity(event)
    assert identity and identity[0] == "plugin_completion", (
        "the local admission path rejects an event without a producer identity")
    from tools.process_registry_notifications import PROCESS_COMPLETE_DISPLAY_KIND
    card = _local_completion_card(event)
    assert card["display_kind"] == PROCESS_COMPLETE_DISPLAY_KIND
    assert "plugin a2a" in card["display_text"]
    assert "Peer finished task 7." in card["display_text"]


@pytest.mark.asyncio
async def test_notifications_off_acknowledges_without_injecting(monkeypatch, tmp_path, bound_session):
    runner = _build_runner(monkeypatch, tmp_path, mode="off")
    adapter = runner.adapters[Platform.TELEGRAM]
    events = queue.Queue()
    events.put(_published_event(bound_session))

    await runner._drain_watch_notifications(events)

    adapter.handle_message.assert_not_awaited()
    assert _row()["delivery_state"] == "delivered", (
        "an explicitly suppressed completion is consumed, not left pending forever")


@pytest.mark.asyncio
async def test_a_claim_held_by_another_consumer_is_skipped(monkeypatch, tmp_path, bound_session):
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.TELEGRAM]
    events = queue.Queue()
    evt = _published_event(bound_session)
    assert pc.claim_plugin_completion(evt["completion_id"], "tui:poller") is True
    events.put(evt)

    await runner._drain_watch_notifications(events)

    adapter.handle_message.assert_not_awaited()
    assert _row()["delivery_state"] == "pending"
    # The owning consumer still settles it.
    assert pc.complete_plugin_completion(evt["completion_id"], "tui:poller") is True
    assert _row()["delivery_state"] == "delivered"
