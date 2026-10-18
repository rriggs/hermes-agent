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

import contextlib
import queue
import sqlite3
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
    row = pc._existing_completion(PLUGIN, SESSION_ID, event_id)
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

@pytest.mark.asyncio
async def test_claim_inject_and_settle_all_run_in_the_owning_profile_scope(
    monkeypatch, tmp_path, bound_session,
):
    """A secondary profile's row lives in ITS state.db; the launch scope would strand it."""
    runner = _build_runner(monkeypatch, tmp_path)
    events = queue.Queue()
    events.put(_published_event(bound_session))
    import gateway.run_notifications as rn

    scope = {"active": False, "claim_in_scope": None, "settle_in_scope": None}

    @contextlib.asynccontextmanager
    async def recording_scope(evt):
        scope["active"] = True
        try:
            yield
        finally:
            scope["active"] = False

    monkeypatch.setattr(runner, "_completion_event_scope", recording_scope, raising=False)
    real_claim, real_settle = rn._plugin_completion_claim, rn._settle_plugin_completion

    def claim(evt, consumer):
        scope["claim_in_scope"] = scope["active"]
        return real_claim(evt, consumer)

    def settle(evt, claim_id, operation):
        scope["settle_in_scope"] = scope["active"]
        return real_settle(evt, claim_id, operation)

    monkeypatch.setattr(rn, "_plugin_completion_claim", claim)
    monkeypatch.setattr(rn, "_settle_plugin_completion", settle)

    await runner._drain_watch_notifications(events)

    assert scope["claim_in_scope"] is True, (
        "the claim ran outside the owning profile scope: it would hit the launch profile's ledger")
    assert scope["settle_in_scope"] is True, (
        "settlement ran outside the owning profile scope: the real row would stay pending")
    assert _row()["delivery_state"] == "delivered"


def test_a_stale_claim_is_reoffered_without_a_restart(bound_session):
    """A consumer that dies while holding the claim must not strand the row until restart."""
    events = queue.Queue()
    evt = _published_event(bound_session)
    assert pc.claim_plugin_completion(evt["completion_id"], "dead:consumer") is True

    # Past the claim lease (300s) and the sweep's stale window (60s), same process, same _offered set.
    reoffered = pc.sweep_stale_plugin_completions(events, now=time.time() + 400.0)

    assert reoffered == 1, "the process-local offer marker suppressed the stale-claim sweep"
    assert events.qsize() == 1
    assert _row()["delivery_state"] == "pending"


def test_a_delivered_event_stays_a_duplicate_while_its_target_is_publishable(
        bound_session, monkeypatch):
    """Dedup rows are the "never re-notify" contract: they must outlive the target."""
    target = pc.capture_completion_target(PLUGIN, session_id=SESSION_ID)
    pc.publish_completion(PLUGIN, target=target, event_id="task-9:done", content="done")
    row = _row("task-9:done")
    assert pc.claim_plugin_completion(row["completion_id"], "test:consumer") is True
    assert pc.complete_plugin_completion(row["completion_id"], "test:consumer") is True

    real = time.time()
    monkeypatch.setattr(pc, "time", SimpleNamespace(time=lambda: real + 8 * 86400))
    pc._prune()

    again = pc.publish_completion(PLUGIN, target=target, event_id="task-9:done", content="done")
    assert again["status"] == "duplicate", (
        "the dedup tombstone was pruned while the target could still publish")


def test_a_target_is_publishable_for_thirty_days_and_pruned_by_expiry(bound_session, monkeypatch):
    target = pc.capture_completion_target(PLUGIN, session_id=SESSION_ID)
    assert target.expires_at is not None
    assert target.expires_at > time.time() + 29 * 86400

    base = time.time()

    def at(offset: float) -> None:
        monkeypatch.setattr(pc, "time", SimpleNamespace(time=lambda: base + offset))

    def target_rows() -> int:
        with sqlite3.connect(pc._db_path()) as con:
            return con.execute(
                "SELECT COUNT(*) FROM plugin_completion_targets WHERE target_id=?",
                (target.target_id,)).fetchone()[0]

    # Day 8 is past the ledger's 7-day row retention, but the target was captured for a long watch.
    at(8 * 86400)
    pc._prune()
    assert target_rows() == 1, "a live target was pruned by capture age"

    at(31 * 86400)
    with pytest.raises(pc.CompletionTargetError):
        pc.publish_completion(PLUGIN, target=target, event_id="late", content="too late")
    pc._prune()
    assert target_rows() == 1, "an expired target is kept through its retention grace window"

    # Expiry plus the ledger's retention grace: long enough to explain the failure, then gone.
    at(38 * 86400)
    pc._prune()
    assert target_rows() == 0, "an expired target outlived its declared expiry"

def test_a_new_conversation_is_not_a_duplicate(monkeypatch, tmp_path):
    """/new keeps the chat's session_key; the completion must still reach the new conversation."""
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: concise\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    def publish_in(session_id: str, content: str) -> dict:
        tokens = set_session_vars(
            platform="telegram", source="telegram", chat_id="42", chat_type="dm",
            session_key=SESSION_KEY, session_id=session_id)
        try:
            target = pc.capture_completion_target(PLUGIN, session_id=session_id)
            return pc.publish_completion(
                PLUGIN, target=target, event_id="task-7:completed", content=content)
        finally:
            clear_session_vars(tokens)

    first = publish_in("sess-A", "finished in A")
    second = publish_in("sess-B", "finished in B")

    assert first["status"] == "queued"
    assert second["status"] == "queued", (
        "a publish into a different conversation was reported as a duplicate and dropped")
    # ... while a real duplicate in the SAME conversation still deduplicates.
    again = publish_in("sess-B", "finished in B twice")
    assert again["status"] == "duplicate"
    assert again["completion_id"] == second["completion_id"]


def test_the_sweep_does_not_multiply_a_live_unclaimed_offer(bound_session):
    """A copy that is still queued must not be duplicated by every sweep pass."""
    events = queue.Queue()
    events.put(_published_event(bound_session))  # the live, never-claimed copy

    for _ in range(3):
        assert pc.sweep_stale_plugin_completions(events, now=time.time() + 120.0) == 0

    assert events.qsize() == 1
    assert _row()["delivery_state"] == "pending"


def test_a_deferred_row_is_reoffered_once_and_stops(bound_session):
    events = queue.Queue()
    evt = _published_event(bound_session)
    assert pc.claim_plugin_completion(evt["completion_id"], "refusing:consumer") is True
    assert pc.defer_plugin_completion(evt["completion_id"], "refusing:consumer") is True

    assert pc.sweep_stale_plugin_completions(events, now=time.time() + 120.0) == 1
    assert events.qsize() == 1
    # The re-offer clears the deferred marker, so the next sweep adds no second copy.
    assert pc.sweep_stale_plugin_completions(events, now=time.time() + 130.0) == 0
    assert events.qsize() == 1


def test_an_abandoned_claim_is_reoffered_once_even_with_the_offer_marker_set(bound_session):
    events = queue.Queue()
    evt = _published_event(bound_session)
    assert pc.claim_plugin_completion(evt["completion_id"], "dead:consumer") is True

    assert pc.sweep_stale_plugin_completions(events, now=time.time() + 400.0) == 1
    assert events.qsize() == 1
    # The re-offer cleared the dead claim: later sweeps must not add more copies of one event.
    assert pc.sweep_stale_plugin_completions(events, now=time.time() + 410.0) == 0
    assert events.qsize() == 1


def test_a_discarded_startup_offer_is_eligible_for_the_sweep(bound_session):
    """A TUI that cannot prove ownership drops the copy; the row must still be recoverable."""
    events = queue.Queue()
    evt = _published_event(bound_session)
    pc.return_plugin_completion_offer(evt)  # the queue copy was discarded

    assert pc.sweep_stale_plugin_completions(events, now=time.time() + 120.0) == 1
    assert events.qsize() == 1
