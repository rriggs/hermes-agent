"""Behaviour contracts for the public plugin completion publisher (``agent/plugin_completions.py``).

The API is the supported way for a plugin to deliver an async result into the session that invoked
it: capture a target during the tool call, publish later, and the durable row is claimed and
acknowledged by whichever surface (CLI, TUI, gateway) drains it. These tests exercise the real
state.db, the real queue and the real drain/format paths against the hermetic temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty

import pytest

from agent import plugin_completions as pc
from gateway.session_context import clear_session_vars, declare_stateless_channel, set_session_vars

SESSION_KEY = "agent:main:telegram:dm:42"
SESSION_ID = "sess-1"


@pytest.fixture
def bound_session():
    """A gateway turn's session context, as a tool handler would see it."""
    tokens = set_session_vars(
        platform="telegram", source="telegram", chat_id="42", chat_type="dm",
        session_key=SESSION_KEY, session_id=SESSION_ID, profile="default",
    )
    try:
        yield tokens
    finally:
        clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _fresh_completion_state():
    """The offer set and the queue are process-global: each test starts with neither."""
    from tools.process_registry import process_registry
    pc._offered.clear()
    while True:
        try:
            process_registry.completion_queue.get_nowait()
        except Empty:
            break
    yield
    pc._offered.clear()
    while True:
        try:
            process_registry.completion_queue.get_nowait()
        except Empty:
            break


@pytest.fixture(autouse=True)
def _no_pm_boot_probe(monkeypatch):
    """``import hermes_bootstrap`` selects the dependency tree at process boot, which probes for a
    sealed payload manifest beside the checkout. That is boot plumbing, not the behaviour under
    test here (and a checkout that lives inside the real ``~/.hermes`` would have the probe read a
    home path, tripping the test I/O guard)."""
    import pm.environments as pe
    monkeypatch.setattr(pe, "activate_dependencies", lambda _root: None)


def _queue_events() -> list[dict]:
    from tools.process_registry import process_registry
    events = []
    while True:
        try:
            events.append(process_registry.completion_queue.get_nowait())
        except Empty:
            return events


# ── capture ─────────────────────────────────────────────────────────────────

def test_capture_binds_the_invoking_session_and_survives_serialization(bound_session):
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)

    assert target.plugin_id == "a2a"
    assert target.session_key == SESSION_KEY
    assert target.session_id == SESSION_ID
    assert target.platform == "telegram"
    assert target.chat_id == "42"
    assert target.profile == "default"

    # A plugin persists the mapping and uses it after a restart: the round trip must be lossless.
    restored = pc.CompletionTarget.from_dict(json.loads(json.dumps(target.to_dict())))
    assert restored == target


def test_capture_refuses_a_session_id_that_is_not_the_bound_one(bound_session):
    with pytest.raises(pc.CompletionTargetError):
        pc.capture_completion_target("a2a", session_id="some-other-session")


def test_capture_refuses_a_surface_that_can_never_deliver(bound_session):
    """A one-shot/stateless session learns at capture time, instead of persisting a dead watch."""
    declare_stateless_channel()
    with pytest.raises(pc.CompletionDeliveryUnsupported):
        pc.capture_completion_target("a2a", session_id=SESSION_ID)


def test_capture_without_a_bound_session_is_refused():
    with pytest.raises(pc.CompletionTargetError):
        pc.capture_completion_target("a2a")


# ── publish ─────────────────────────────────────────────────────────────────

def test_publish_persists_then_queues_and_is_idempotent(bound_session):
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)

    receipt = pc.publish_completion(
        "a2a", target=target.to_dict(), event_id="task-1:completed",
        content="Remote task 1 completed.", metadata={"task_id": "task-1", "state": "completed"})

    assert receipt["status"] == "queued"
    assert receipt["delivery_state"] == "pending"
    events = _queue_events()
    assert len(events) == 1
    assert events[0]["type"] == "plugin_completion"
    assert events[0]["session_key"] == SESSION_KEY
    assert events[0]["content"] == "Remote task 1 completed."

    # A restarted watcher re-observing the same terminal state must not notify twice.
    duplicate = pc.publish_completion("a2a", target=target, event_id="task-1:completed", content="again")
    assert duplicate["status"] == "duplicate"
    assert duplicate["completion_id"] == receipt["completion_id"]
    assert _queue_events() == []


def test_publish_refuses_a_target_captured_by_another_plugin(bound_session):
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    with pytest.raises(pc.CompletionTargetError):
        pc.publish_completion("someone-else", target=target, event_id="e1", content="not yours")


def test_publish_refuses_unknown_and_malformed_targets(bound_session):
    captured = pc.capture_completion_target("a2a", session_id=SESSION_ID).to_dict()
    with pytest.raises(pc.CompletionTargetError):
        pc.publish_completion("a2a", target={**captured, "target_id": "pct_missing"},
                              event_id="e1", content="x")
    with pytest.raises(pc.CompletionTargetError):
        pc.publish_completion("a2a", target={**captured, "contract_version": 999},
                              event_id="e2", content="x")
    with pytest.raises(pc.CompletionTargetError):
        pc.publish_completion("a2a", target={**captured, "surprise": "field"},
                              event_id="e3", content="x")


def test_publish_rejects_an_over_budget_request(bound_session):
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    with pytest.raises(pc.CompletionRequestError):
        pc.publish_completion("a2a", target=target, event_id="", content="x")
    with pytest.raises(pc.CompletionRequestError):
        pc.publish_completion("a2a", target=target, event_id="e1", content="")
    with pytest.raises(pc.CompletionRequestError):
        pc.publish_completion("a2a", target=target, event_id="e2", content="x" * (pc._MAX_CONTENT_CHARS + 1))
    with pytest.raises(pc.CompletionRequestError):
        pc.publish_completion(
            "a2a", target=target, event_id="e3", content="x",
            metadata={"blob": "y" * (pc._MAX_METADATA_BYTES + 1)})
    assert _queue_events() == []


# ── delivery claims ─────────────────────────────────────────────────────────

def test_one_consumer_owns_a_completion_until_it_settles(bound_session):
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="e1", content="done")
    (evt,) = _queue_events()

    assert pc.claim_plugin_completion(evt["completion_id"], "cli:1") is True
    assert pc.claim_plugin_completion(evt["completion_id"], "tui:1") is False
    assert pc.complete_plugin_completion(evt["completion_id"], "cli:1") is True
    # Delivered is terminal: a retry can never notify the session a second time.
    assert pc.claim_plugin_completion(evt["completion_id"], "cli:2") is False
    assert pc.complete_plugin_completion(evt["completion_id"], "cli:1") is False


def test_a_released_claim_is_retryable_and_a_deferred_one_spends_no_attempt(bound_session):
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="e1", content="done")
    (evt,) = _queue_events()

    assert pc.claim_plugin_completion(evt["completion_id"], "first") is True
    assert pc.defer_plugin_completion(evt["completion_id"], "first") is True
    assert pc.claim_plugin_completion(evt["completion_id"], "second") is True
    assert pc.release_plugin_completion(evt["completion_id"], "second") is True
    assert pc.claim_plugin_completion(evt["completion_id"], "third") is True


# ── restart replay ──────────────────────────────────────────────────────────

def test_a_repeatedly_abandoned_claim_is_dropped_at_the_attempt_cap(bound_session, monkeypatch):
    """A consumer that dies holding its claim never settles; the sweep must still cap it."""
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="task-9:done", content="x")
    row = pc._existing_completion("a2a", SESSION_ID, "task-9:done")
    from queue import Queue

    monkeypatch.setattr(pc, "_CLAIM_LEASE_S", 0.0)
    queue = Queue()
    for _ in range(pc._MAX_DELIVERY_ATTEMPTS + 1):
        if not pc.claim_plugin_completion(row["completion_id"], "dead-consumer"):
            break
        if pc.sweep_stale_plugin_completions(queue, now=pc.time.time() + 1) == 0:
            break
        queue.get_nowait()  # re-offered copy

    with pc._DB_LOCK, pc._transaction() as conn:
        state = conn.execute(
            "SELECT delivery_state FROM plugin_completions WHERE completion_id=?",
            (row["completion_id"],)).fetchone()[0]
    assert state == "dropped", "an event no consumer can settle must not re-offer forever"


def test_metadata_keys_are_redacted_too(bound_session):
    secret = "ghp_" + "G" * 29
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="task-10:done", content="done",
                          metadata={secret: "value"})

    event = _queue_events()[-1]
    assert secret not in json.dumps(event["metadata"]), (
        "keys are rendered as key=value too")


def test_metadata_is_redacted_before_it_reaches_the_model(bound_session):
    """Metadata is rendered into the notification, so it gets the same redaction as content."""
    secret = "ghp_" + "F" * 29
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    receipt = pc.publish_completion(
        "a2a", target=target, event_id="task-8:done", content="done",
        metadata={"task_id": "task-8", "note": f"token {secret}", "nested": {"t": secret}})

    event = _queue_events()[-1]
    rendered = json.dumps(event["metadata"])

    assert secret not in rendered
    assert receipt["status"] == "queued"
    row = pc._existing_completion("a2a", SESSION_ID, "task-8:done")
    assert row is not None


def test_a_prune_failure_does_not_leave_an_accepted_publication_unoffered(
    bound_session, monkeypatch,
):
    """Housekeeping after an accepted insert must never raise before the copy is queued."""
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    monkeypatch.setattr(pc, "_prune", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))

    receipt = pc.publish_completion("a2a", target=target, event_id="task-6:done", content="x")

    assert receipt["status"] == "queued"
    assert [e["event_id"] for e in _queue_events()] == ["task-6:done"]


def test_the_pending_cap_refuses_new_work_instead_of_deleting_accepted_work(
    bound_session, monkeypatch,
):
    monkeypatch.setattr(pc, "_MAX_PENDING_ROWS", 2)
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    for i in range(2):
        pc.publish_completion("a2a", target=target, event_id=f"task-7:{i}", content="x")

    with pytest.raises(pc.CompletionError):
        pc.publish_completion("a2a", target=target, event_id="task-7:2", content="x")

    events = _queue_events()
    assert len(events) == 2, "the accepted copies must still be on the queue"
    assert all(pc.claim_plugin_completion(e["completion_id"], "c") for e in events), (
        "evicting an accepted row would leave its queue copy unclaimable")


def test_a_sweep_never_clears_a_claim_that_changed_after_its_selection(bound_session):
    """Selection and recovery are separate statements; the recovery is fenced on the claim."""
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="task-1:done", content="x")
    row = pc._existing_completion("a2a", SESSION_ID, "task-1:done")

    with pc._DB_LOCK, pc._transaction() as conn:
        conn.execute(
            "UPDATE plugin_completions SET delivery_claim='stale', delivery_claimed_at=0 "
            "WHERE completion_id=?", (row["completion_id"],))
        # ... another process re-claims the row after this sweep selected the stale claim.
        conn.execute(
            "UPDATE plugin_completions SET delivery_claim='live', delivery_claimed_at=? "
            "WHERE completion_id=?", (pc.time.time(), row["completion_id"]))
        assert pc._clear_stale_claim(conn, row["completion_id"], "stale") is False

    with pc._DB_LOCK, pc._transaction() as conn:
        claim = conn.execute(
            "SELECT delivery_claim FROM plugin_completions WHERE completion_id=?",
            (row["completion_id"],)).fetchone()[0]
    assert claim == "live", "the sweep must not erase a claim another consumer just took"


def test_a_row_whose_fence_lost_is_not_reoffered_or_dropped(bound_session, monkeypatch):
    """A failed claim-clear means another consumer owns the row: leave every column alone."""
    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="task-4:done", content="x")
    row = pc._existing_completion("a2a", SESSION_ID, "task-4:done")
    old = pc.time.time() - 49 * 3600
    with pc._DB_LOCK, pc._transaction() as conn:
        conn.execute(
            "UPDATE plugin_completions SET delivery_claim='live', delivery_claimed_at=0, "
            "created_at=? WHERE completion_id=?", (old, row["completion_id"]))

    monkeypatch.setattr(pc, "_clear_stale_claim", lambda *args: False)
    from queue import Queue

    assert pc.sweep_stale_plugin_completions(Queue(), now=pc.time.time()) == 0
    with pc._DB_LOCK, pc._transaction() as conn:
        state, claim = conn.execute(
            "SELECT delivery_state, delivery_claim FROM plugin_completions WHERE completion_id=?",
            (row["completion_id"],)).fetchone()
    assert (state, claim) == ("pending", "live"), "the sweep touched a row it lost the fence on"


def test_returning_an_offer_writes_the_owners_ledger_not_the_callers(tmp_path, bound_session):
    """The queue is process-wide, the ledger is per profile: return under the owner's scope."""
    owner_home = tmp_path / "owner"
    owner_home.mkdir()
    with pc._profile_scope(str(owner_home)):
        target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
        pc.publish_completion("a2a", target=target, event_id="task-2:done", content="x")

    event = _queue_events()[-1]
    assert event["profile_home"] == str(owner_home)

    pc.return_plugin_completion_offer(event)  # caller runs in the test profile, not the owner's

    with pc._profile_scope(str(owner_home)), pc._DB_LOCK, pc._transaction() as conn:
        deferred = conn.execute(
            "SELECT delivery_deferred_at FROM plugin_completions WHERE completion_id=?",
            (event["completion_id"],)).fetchone()
    assert deferred is not None and deferred[0] is not None, (
        "the owner's row must become sweep-visible, or it is stranded until a restart")


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def __getattr__(self, name):
        import time as _time
        return getattr(_time, name)


def test_a_tombstone_outlives_a_late_target_in_its_conversation(bound_session, monkeypatch):
    """Dedup is conversation-scoped, so a tombstone must live as long as any target can publish."""
    clock = _Clock(1_800_000_000.0)
    monkeypatch.setattr(pc, "time", clock)

    first = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=first, event_id="task-3:done", content="x")
    row = pc._existing_completion("a2a", SESSION_ID, "task-3:done")
    assert pc.claim_plugin_completion(row["completion_id"], "consumer")
    assert pc.complete_plugin_completion(row["completion_id"], "consumer")

    clock.now += 29 * 86400  # same conversation, a fresh target, still inside its 30-day life
    late = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    clock.now += 9 * 86400   # day 38: past the tombstone's own retention, target still live
    pc._prune()

    receipt = pc.publish_completion("a2a", target=late, event_id="task-3:done", content="x")
    assert receipt["status"] == "duplicate", "a pruned tombstone would deliver the same event twice"


def test_a_pending_completion_replays_after_a_restart(bound_session):
    """The in-memory queue dies with the process; the durable row must re-offer on the next boot."""
    from tools.async_delegation import restore_undelivered_completions
    from tools.process_registry import process_registry

    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="e1", content="survived")
    _queue_events()
    pc._offered.clear()  # a new process has no in-memory offers

    assert restore_undelivered_completions(process_registry.completion_queue) == 1
    (evt,) = _queue_events()
    assert evt["event_id"] == "e1"
    assert evt["content"] == "survived"
    assert evt["restored"] is True and evt["startup_replay"] is True

    # The offer set keeps a second replay pass in the same process from duplicating the turn.
    assert restore_undelivered_completions(process_registry.completion_queue) == 0
    assert _queue_events() == []


def test_a_delivered_completion_never_replays(bound_session):
    from tools.async_delegation import restore_undelivered_completions
    from tools.process_registry import process_registry

    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="e1", content="done")
    (evt,) = _queue_events()
    assert pc.claim_plugin_completion(evt["completion_id"], "cli:1")
    assert pc.complete_plugin_completion(evt["completion_id"], "cli:1")

    assert restore_undelivered_completions(process_registry.completion_queue) == 0
    assert _queue_events() == []


# ── surfaces ────────────────────────────────────────────────────────────────

def test_the_cli_drain_delivers_the_text_and_acknowledges_the_row(bound_session):
    """End to end on the CLI path: the real drain routes by session, formats the event, and the
    completion row settles delivered only through the claim/ack pair the drain calls."""
    from tools.async_delegation import claim_event_delivery, complete_event_delivery
    from tools.process_registry import process_registry

    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion(
        "a2a", target=target, event_id="e1", content="Peer finished task 7.",
        metadata={"task_id": "7", "state": "completed"})

    def owns(evt):
        return evt.get("session_key") == SESSION_KEY

    drained = process_registry.drain_notifications(session_key=SESSION_KEY, owns_event=owns)
    assert len(drained) == 1
    evt, text = drained[0]
    assert "Peer finished task 7." in text
    assert "plugin a2a" in text

    claim = claim_event_delivery(evt, "cli:drain")
    assert claim
    complete_event_delivery(evt, claim)
    assert pc.claim_plugin_completion(evt["completion_id"], "second") is False


def test_a_foreign_session_cannot_drain_someone_elses_completion(bound_session):
    from tools.process_registry import process_registry

    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="e1", content="mine")

    drained = process_registry.drain_notifications(
        session_key="a-different-session", owns_event=lambda evt: False)
    assert drained == []
    # Not consumed: the owner's next drain still finds it.
    refetched = process_registry.drain_notifications(
        session_key=SESSION_KEY, owns_event=lambda evt: evt.get("session_key") == SESSION_KEY)
    assert [text for _evt, text in refetched] and "mine" in refetched[0][1]


def test_the_gateway_takes_messaging_routes_and_leaves_local_ones(bound_session):
    from gateway.run import _format_gateway_process_notification
    from gateway.run_notifications import _drain_gateway_watch_events
    from tools.process_registry import process_registry

    target = pc.capture_completion_target("a2a", session_id=SESSION_ID)
    pc.publish_completion("a2a", target=target, event_id="e1", content="gateway result")

    watch_events = _drain_gateway_watch_events(process_registry.completion_queue)
    assert [evt["event_id"] for evt in watch_events] == ["e1"]
    assert "gateway result" in _format_gateway_process_notification(watch_events[0])


def test_a_local_route_is_requeued_for_its_own_surface(bound_session):
    """A CLI/TUI session's completion stays on the queue: the gateway must not consume or drop it."""
    clear_session_vars([])  # a CLI turn binds only the session id
    from gateway.run_notifications import _drain_gateway_watch_events
    from tools.process_registry import process_registry

    target = pc.capture_completion_target("a2a", session_id="cli-session-id")
    pc.publish_completion("a2a", target=target, event_id="e1", content="local result")

    assert _drain_gateway_watch_events(process_registry.completion_queue) == []
    assert [evt["event_id"] for evt in _queue_events()] == ["e1"]


# ── external plugin compatibility ───────────────────────────────────────────

def test_a_standalone_plugin_publishes_through_ctx(tmp_path):
    """The frozen external-plugin contract: a plugin loaded by real discovery publishes with the
    public ``ctx`` surface only, and a messaging-style drain delivers the result."""
    import hermes_yaml as yaml
    from hermes_cli import plugins as plugins_mod
    from tools.registry import registry

    hermes_home = Path(os.environ["HERMES_HOME"])
    plugin_dir = hermes_home / "plugins" / "completion_publisher"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: completion_publisher\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "import json\n"
        "\n"
        "def register(ctx):\n"
        "    def publish(args, session_id=None):\n"
        "        target = ctx.capture_completion_target(session_id=session_id)\n"
        "        receipt = ctx.publish_completion(\n"
        "            target=target, event_id=args['event_id'], content=args['content'])\n"
        "        return json.dumps(receipt)\n"
        "    ctx.register_tool(\n"
        "        name='completion_publisher_demo', toolset='demo',\n"
        "        schema={'name': 'completion_publisher_demo', 'description': 'demo',\n"
        "                'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=publish)\n",
        encoding="utf-8")
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["completion_publisher"]}}), encoding="utf-8")

    previous = getattr(plugins_mod, "_plugin_manager", None)
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    try:
        plugins_mod.discover_plugins()
        tokens = set_session_vars(
            platform="telegram", source="telegram", chat_id="42", chat_type="dm",
            session_key=SESSION_KEY, session_id=SESSION_ID)
        try:
            raw = registry.dispatch("completion_publisher_demo", {
                "event_id": "peer-task-9", "content": "Peer finished task 9."}, session_id=SESSION_ID)
        finally:
            clear_session_vars(tokens)
    finally:
        if previous is None:
            plugins_mod._plugin_manager = plugins_mod.PluginManager()
        else:
            plugins_mod._plugin_manager = previous

    receipt = json.loads(raw)
    assert receipt["status"] == "queued"
    (evt,) = _queue_events()
    assert evt["plugin_id"] == "completion_publisher"
    assert evt["session_key"] == SESSION_KEY
    assert evt["content"] == "Peer finished task 9."
