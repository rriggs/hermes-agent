"""Public, plugin-safe completion publisher: ``PluginContext.capture_completion_target`` /
``PluginContext.publish_completion``.

A plugin can run work that outlives the tool call (an A2A task watcher, a long poll). Until now it
had no supported way to deliver the result back into the conversation that asked for it: the plugin
event bus is plugin-local and the durable completion queue is a Hermes internal.

The contract this module implements:

* **Capture, then publish.** ``capture_completion_target`` runs during the tool invocation and binds
  an opaque target to the CURRENT tool invocation's profile, session and route (it reads the
  dispatcher-bound session context, never a later guess). The target is a plain, JSON-safe dict the
  plugin may persist in its own store; it stays valid across a process restart, which is what a
  durable background watch needs.
* **Persist before queueing.** ``publish_completion`` writes a durable row to the profile's
  ``state.db`` and only then offers an event on ``process_registry.completion_queue`` -- the same
  queue the CLI, TUI and gateway already drain. The result starts a NEW turn at an idle boundary,
  never mid-turn. Deduplication is by ``(plugin, session, event_id)`` so a restarted watcher that
  re-observes the same terminal state cannot notify twice.
* **Claims, not guesses.** Every consumer claims a row before injecting it and acks after the
  surface accepted the turn; a refused or interrupted delivery releases the claim for another
  attempt, bounded by ``_MAX_DELIVERY_ATTEMPTS``. Pending rows replay after a restart through the
  same policy.
* **Honest receipts.** A receipt says ``queued`` -- queue admission, not delivery. ``status`` never
  claims the model processed the content or that a user-facing reply was sent.

Ownership: a target row records the plugin that captured it; publishing against another plugin's
target, an unknown target or an expired one raises. A plugin cannot publish into an arbitrary
session, and nothing here lets a plugin read or write another plugin's rows.

This module never imports ``tools.process_registry`` at module scope (the registry imports tool
modules during discovery); the queue is resolved lazily at publish time.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Mapping, Optional

from hermes_constants import (
    get_hermes_home, get_hermes_home_override, hermes_home_key,
    reset_hermes_home_override, set_hermes_home_override,
)

logger = logging.getLogger(__name__)

# The plugin-facing contract version. Bumped only for a breaking change to the target/receipt shape;
# additive fields are not a version bump (plugins/AGENTS.md compat contract).
PUBLIC_CONTRACT_VERSION = 1

EVENT_TYPE = "plugin_completion"

# A completion's content is a message the agent will read: cap it so a chatty peer cannot inject a
# whole transcript as one turn, and cap metadata so the receipt stays a receipt.
_MAX_CONTENT_CHARS = 32_000
_MAX_METADATA_BYTES = 8_192
_MAX_EVENT_ID_CHARS = 200
_MAX_PLUGIN_ID_CHARS = 128
# Cap retried deliveries so an unroutable row converges to terminal 'dropped' instead of spinning.
_MAX_DELIVERY_ATTEMPTS = 8
# A claim older than this is abandoned (its consumer died mid-injection) and may be re-claimed.
_CLAIM_LEASE_S = 300.0
# Pending completions older than this are dropped on replay instead of starting a turn nobody waits on.
_MAX_REPLAY_AGE_S = 48 * 3600.0
_RETENTION_SECONDS = 7 * 24 * 60 * 60
# How long a captured target stays publishable. A plugin may legitimately hold a target across
# hours of background work, so the default is generous; it is a real, enforced expiry (publish
# raises ``CompletionTargetError`` after it) rather than a pruning accident.
_DEFAULT_TARGET_TTL_S = 30 * 24 * 60 * 60
_MAX_PENDING_ROWS = 2_000
# Idle floor before the sweep re-offers a row whose claim lease expired.
_SWEEP_STALE_S = 60.0

_DB_LOCK = threading.RLock()
_offer_lock = threading.Lock()
# (home key, completion_id) copies this process put on the queue and has not handed back: a sweep
# must not offer the same row twice while the first copy is still live. The delivery claim, not this
# set, is what settles the row.
_offered: set[tuple[str, str]] = set()

TARGET_FIELDS = (
    "contract_version", "target_id", "plugin_id", "session_key", "session_id", "platform",
    "chat_id", "chat_type", "thread_id", "scope_id", "user_id", "user_name", "ui_session_id",
    "profile", "profile_home", "created_at", "expires_at",
)
# Target fields copied onto the queued event: the drain paths route on exactly these.
_ROUTE_FIELDS = (
    "session_key", "session_id", "platform", "chat_id", "chat_type", "thread_id", "scope_id",
    "user_id", "user_name", "ui_session_id", "profile",
)


class CompletionError(ValueError):
    """A completion request the public API cannot safely accept."""


class CompletionTargetError(CompletionError):
    """Unknown, foreign, expired or malformed completion target."""


class CompletionRequestError(CompletionError):
    """Malformed event id, content or metadata."""


class CompletionDeliveryUnsupported(RuntimeError):
    """This session cannot receive a turn later (one-shot, cron, Kanban, worker, stateless API).

    Raised at capture time so a plugin learns immediately that nothing will deliver the result,
    instead of persisting a watch whose completions would be dropped.
    """


@dataclasses.dataclass(frozen=True)
class CompletionTarget:
    """Opaque, serializable delivery target captured from one tool invocation."""

    contract_version: int
    target_id: str
    plugin_id: str
    session_key: str
    session_id: str
    platform: str
    chat_id: str
    chat_type: str
    thread_id: str
    scope_id: str
    user_id: str
    user_name: str
    ui_session_id: str
    profile: str
    profile_home: str
    created_at: float
    expires_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompletionTarget:
        if not isinstance(value, Mapping):
            raise CompletionTargetError("completion target must be a mapping.")
        unknown = set(value) - set(TARGET_FIELDS)
        if unknown:
            raise CompletionTargetError(f"unknown completion target fields: {', '.join(sorted(unknown))}.")
        try:
            target = cls(**{key: value.get(key) for key in TARGET_FIELDS})  # type: ignore[arg-type]
        except TypeError as exc:
            raise CompletionTargetError(f"malformed completion target: {exc}") from exc
        target.validate()
        return target

    def validate(self) -> None:
        """Field-shape check applied to a (possibly deserialized) target before trusting it."""
        if type(self.contract_version) is not int or self.contract_version != PUBLIC_CONTRACT_VERSION:
            raise CompletionTargetError("unsupported completion target contract version.")
        for name in ("target_id", "plugin_id", "session_key", "session_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise CompletionTargetError(f"completion target {name} must be a string.")
            if name in ("target_id", "plugin_id") and not value:
                raise CompletionTargetError(f"completion target {name} must not be empty.")
        for name in _ROUTE_FIELDS:
            if not isinstance(getattr(self, name), str):
                raise CompletionTargetError(f"completion target {name} must be a string.")
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, (int, float)):
            raise CompletionTargetError("completion target created_at must be a number.")
        if self.expires_at is not None and (
            isinstance(self.expires_at, bool) or not isinstance(self.expires_at, (int, float))
        ):
            raise CompletionTargetError("completion target expires_at must be a number or null.")


# ── capture ─────────────────────────────────────────────────────────────────

def capture_completion_target(
    plugin_id: str, *, session_id: Optional[str] = None,
) -> CompletionTarget:
    """Bind a delivery target to the current tool invocation.

    ``session_id`` is the dispatcher-provided session id the tool call arrived with. When the
    context has a bound session id too, the two must agree: a plugin cannot capture a target for a
    session other than the one it is running in.
    """
    plugin_id = _clean_plugin_id(plugin_id)
    _require_async_delivery()
    bound_id = _bound_session_id()
    requested = _clean_session_id(session_id)
    if requested and bound_id and requested != bound_id:
        raise CompletionTargetError(
            "session_id does not match the session that invoked this plugin; capture the target "
            "from a tool call or omit session_id.")
    route = _current_route(bound_id or requested)
    if not (route["session_key"] or route["session_id"]):
        raise CompletionTargetError(
            "no session is bound to this call: capture_completion_target() must run inside a tool "
            "invocation, not from a background thread.")
    now = time.time()
    target = CompletionTarget(
        contract_version=PUBLIC_CONTRACT_VERSION,
        target_id=f"pct_{uuid.uuid4().hex}",
        plugin_id=plugin_id,
        profile_home=str(get_hermes_home().resolve()),
        profile=_active_profile_name(),
        created_at=now,
        expires_at=now + _DEFAULT_TARGET_TTL_S,
        **_route_kwargs(route),
    )
    with _profile_scope(target.profile_home):
        _insert_target(target)
    logger.debug("Plugin %s captured completion target %s (session_key=%s)",
                 plugin_id, target.target_id, target.session_key or target.session_id)
    return target


# ── publish ─────────────────────────────────────────────────────────────────

def publish_completion(
    plugin_id: str, *, target: Any, event_id: str, content: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Persist a completion for ``target`` and offer it for delivery; return a receipt.

    Idempotent per ``(plugin, target session, event_id)``: a repeated publish returns the existing
    row's receipt with ``status='duplicate'`` and queues nothing. A row already delivered stays
    delivered -- a retry never re-notifies the session.
    """
    plugin_id = _clean_plugin_id(plugin_id)
    tgt = _coerce_target(target)
    if tgt.plugin_id != plugin_id:
        raise CompletionTargetError("completion target belongs to another plugin.")
    event_id = _clean_event_id(event_id)
    content = _clean_content(content)
    meta = _clean_metadata(metadata)
    now = time.time()
    with _profile_scope(tgt.profile_home):
        row = _load_target(tgt.target_id)
        if row is None:
            raise CompletionTargetError(f"unknown completion target {tgt.target_id!r}.")
        if row.plugin_id != plugin_id:
            raise CompletionTargetError("completion target belongs to another plugin.")
        if row.expires_at is not None and row.expires_at <= now:
            raise CompletionTargetError(f"completion target {tgt.target_id!r} has expired.")
        completion_id = f"pcp_{uuid.uuid4().hex}"
        event = {
            "type": EVENT_TYPE,
            "completion_id": completion_id,
            "plugin_id": plugin_id,
            "event_id": event_id,
            "target_id": tgt.target_id,
            "content": content,
            "metadata": meta,
            "created_at": now,
            **{name: getattr(row, name) for name in _ROUTE_FIELDS},
        }
        inserted = _insert_completion(completion_id, plugin_id, row, event_id, event, now)
        if not inserted:
            existing = _existing_completion(plugin_id, _conversation_id(tgt), event_id)
            # A concurrent publisher may have raced us to the row; report it, never queue twice.
            if existing is None:  # pragma: no cover - only a torn unique index could reach this
                raise CompletionError("duplicate completion row could not be read back.")
            logger.debug("Plugin %s completion %s for event %s already exists (%s)",
                         plugin_id, existing["completion_id"], event_id, existing["delivery_state"])
            return _receipt(existing, status="duplicate")
        _prune()
    _offer(event, tgt.profile_home)
    logger.info("Plugin %s published completion %s (event=%s, session_key=%s)",
                plugin_id, completion_id, event_id, row.session_key or row.session_id)
    return {
        "status": "queued",
        "completion_id": completion_id,
        "event_id": event_id,
        "target_id": tgt.target_id,
        "delivery_state": "pending",
        "accepted_at": now,
        "detail": (
            "Queued for delivery to the originating session at its next idle boundary. Queued is "
            "not delivered: the session may be busy or gone, and acceptance by the queue is not "
            "proof the model ran on it."),
    }


# ── delivery claims (consumers: CLI / TUI / gateway drains) ─────────────────

def claim_plugin_completion(completion_id: str, claim_id: str) -> bool:
    """Claim a pending row for one consumer; False when another consumer owns it."""
    if not completion_id or not claim_id:
        return False
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return False
        cur = conn.execute(
            """UPDATE plugin_completions
                  SET delivery_claim=?, delivery_claimed_at=?, delivery_attempts=delivery_attempts+1,
                      delivery_deferred_at=NULL, updated_at=?
                WHERE completion_id=? AND delivery_state='pending'
                  AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, completion_id, now - _CLAIM_LEASE_S))
        return cur.rowcount == 1


def complete_plugin_completion(completion_id: str, claim_id: str) -> bool:
    """Acknowledge that the surface accepted the turn carrying this completion."""
    if not completion_id or not claim_id:
        return False
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return False
        cur = conn.execute(
            """UPDATE plugin_completions
                  SET delivery_state='delivered', delivered_at=?, updated_at=?,
                      delivery_claim=NULL, delivery_claimed_at=NULL, delivery_deferred_at=NULL
                WHERE completion_id=? AND delivery_state='pending' AND delivery_claim=?""",
            (now, now, completion_id, claim_id))
        acked = cur.rowcount == 1
    if acked:
        _forget_offer(completion_id)
    return acked


def release_plugin_completion(completion_id: str, claim_id: str) -> bool:
    """Release a failed delivery; the row converges to ``dropped`` once its attempts are spent."""
    if not completion_id or not claim_id:
        return False
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return False
        capped = conn.execute(
            """UPDATE plugin_completions
                  SET delivery_state='dropped', updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
                WHERE completion_id=? AND delivery_state='pending'
                  AND delivery_claim=? AND delivery_attempts>=?""",
            (now, completion_id, claim_id, _MAX_DELIVERY_ATTEMPTS))
        if capped.rowcount == 1:
            logger.warning("Plugin completion %s exhausted its %d delivery attempts; "
                           "terminally dropping it.", completion_id, _MAX_DELIVERY_ATTEMPTS)
            _forget_offer(completion_id)
            return True
        cur = conn.execute(
            """UPDATE plugin_completions
                  SET delivery_claim=NULL, delivery_claimed_at=NULL, delivery_deferred_at=?,
                      updated_at=?
                WHERE completion_id=? AND delivery_state='pending' AND delivery_claim=?""",
            (now, now, completion_id, claim_id))
        released = cur.rowcount == 1
    if released:
        # The in-memory copy is gone and the row is pending: let a later sweep offer it again.
        _forget_offer(completion_id)
    return released


def defer_plugin_completion(completion_id: str, claim_id: str) -> bool:
    """Return an unadmitted completion to pending without spending a delivery attempt."""
    if not completion_id or not claim_id:
        return False
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return False
        cur = conn.execute(
            """UPDATE plugin_completions
                  SET delivery_claim=NULL, delivery_claimed_at=NULL, delivery_deferred_at=?,
                      delivery_attempts=MAX(0, delivery_attempts-1), updated_at=?
                WHERE completion_id=? AND delivery_state='pending' AND delivery_claim=?""",
            (now, now, completion_id, claim_id))
        deferred = cur.rowcount == 1
    if deferred:
        # Deferred, not delivered: the copy this process offered is gone, so a later sweep (or a
        # restarted watcher) must be allowed to offer the row again.
        _forget_offer(completion_id)
    return deferred


def drop_plugin_completion(completion_id: str, claim_id: str) -> bool:
    """Terminally drop a claimed completion whose target session is permanently gone."""
    if not completion_id or not claim_id:
        return False
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return False
        cur = conn.execute(
            """UPDATE plugin_completions
                  SET delivery_state='dropped', updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
                WHERE completion_id=? AND delivery_state='pending' AND delivery_claim=?""",
            (now, completion_id, claim_id))
        dropped = cur.rowcount == 1
    if dropped:
        _forget_offer(completion_id)
    return dropped


def return_plugin_completion_offer(evt: Mapping[str, Any]) -> None:
    """Hand a queued copy back after a consumer discarded it with the row still pending."""
    completion_id = str((evt or {}).get("completion_id") or "")
    if completion_id:
        _forget_offer(completion_id)


# ── replay / sweep ──────────────────────────────────────────────────────────

def restore_pending_plugin_completions(target_queue) -> int:
    """Re-offer this profile's pending completions after a restart; return how many were offered.

    Rows older than ``_MAX_REPLAY_AGE_S`` are terminally dropped: replaying a stale completion
    would start a turn nobody is waiting on. Restored copies carry ``restored=True`` (in memory
    only) so an unfiltered legacy drain leaves them for a consumer that can prove ownership.
    """
    from agent.runtime_session_store import is_worker_process
    if is_worker_process():
        return 0
    if not _db_path().exists():
        return 0  # a replay must never create (or migrate) the ledger
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return 0
        rows = conn.execute(
            """SELECT completion_id, event_json, created_at FROM plugin_completions
                WHERE delivery_state='pending' AND event_json IS NOT NULL
                ORDER BY created_at, completion_id""").fetchall()
        return _offer_rows(conn, rows, target_queue, now, startup=True)


def sweep_stale_plugin_completions(target_queue, *, now: Optional[float] = None) -> int:
    """Re-offer pending rows whose claim lease expired (their consumer died mid-injection)."""
    now = time.time() if now is None else now
    cutoff = now - _CLAIM_LEASE_S
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return 0
        # Two distinct kinds of row need a fresh offer:
        #   * an ABANDONED CLAIM -- the consumer died mid-injection; its queue copy is gone, so the
        #     process-local offer marker must be replaced or the row waits for a restart;
        #   * a DEFERRED row -- a consumer handed it back as undeliverable and there is no copy.
        # A row that was never claimed is already sitting on a queue: re-offering it would grow
        # duplicate copies of one event (three sweeps produced 2, 3, 4 copies).
        rows = conn.execute(
            """SELECT completion_id, event_json, created_at,
                      (delivery_claim IS NOT NULL) AS abandoned
                 FROM plugin_completions
                WHERE delivery_state='pending' AND event_json IS NOT NULL
                  AND ((delivery_claim IS NOT NULL AND delivery_claimed_at < ?)
                       OR (delivery_deferred_at IS NOT NULL AND delivery_deferred_at < ?))
                ORDER BY created_at, completion_id""",
            (cutoff, now - _SWEEP_STALE_S)).fetchall()
        abandoned = {row[0] for row in rows if row[3]}
        return _offer_rows(conn, rows, target_queue, now, force=abandoned)


def _offer_rows(conn, rows: Iterable[tuple], target_queue, now: float, *,
                startup: bool = False, force: Iterable[str] = ()) -> int:
    home, offered = hermes_home_key(get_hermes_home()), 0
    force = set(force)
    for row in rows:
        completion_id, payload, created_at = row[0], row[1], row[2]
        abandoned = bool(row[3]) if len(row) > 3 else False
        if created_at and (now - created_at) > _MAX_REPLAY_AGE_S:
            conn.execute(
                """UPDATE plugin_completions SET delivery_state='dropped', updated_at=?,
                      delivery_claim=NULL, delivery_claimed_at=NULL
                    WHERE completion_id=? AND delivery_state='pending'""",
                (now, completion_id))
            logger.warning("Plugin completion %s is %.1fh old (cap %.1fh); dropping the replay.",
                           completion_id, (now - created_at) / 3600.0, _MAX_REPLAY_AGE_S / 3600.0)
            continue
        evt = _decode_event(payload)
        if evt is None:
            conn.execute(
                """UPDATE plugin_completions SET delivery_state='dropped', updated_at=?
                    WHERE completion_id=? AND delivery_state='pending'""",
                (now, completion_id))
            logger.warning("Plugin completion %s has an unreadable payload; dropping it.",
                           completion_id)
            continue
        if not _mark_offered(home, completion_id, force=completion_id in force):
            continue
        if len(row) > 3 and not abandoned:
            # This deferred row is on a queue again; do not let the next sweep add another copy.
            conn.execute(
                "UPDATE plugin_completions SET delivery_deferred_at=NULL WHERE completion_id=?",
                (completion_id,))
        evt["restored"] = True
        if startup:
            # A first-drain replay has no in-process owner: a surface must prove ownership of this
            # copy before spending a turn on it (same rule the delegation replay applies).
            evt["startup_replay"] = True
        target_queue.put(evt)
        offered += 1
    return offered


# ── internals: routing capture ──────────────────────────────────────────────

def _require_async_delivery() -> None:
    """Refuse capture where no later turn can reach this session."""
    from gateway.session_context import async_delivery_supported
    if not async_delivery_supported():
        raise CompletionDeliveryUnsupported(
            "this session cannot receive a completion later (one-shot, cron, Kanban worker, "
            "managed worker or a stateless API session); poll the operation instead of publishing.")


def _bound_session_id() -> str:
    from gateway.session_context import get_session_env
    return str(get_session_env("HERMES_SESSION_ID", "") or "")


def _current_route(session_id: str) -> dict[str, str]:
    """The current invocation's route: session vars where bound, session id otherwise.

    The CLI binds only ``HERMES_SESSION_ID`` and compares the queue's ``session_key`` against its
    own session id, so an unbound key falls back to the id: that is the value its drain matches.
    """
    from gateway.session_context import get_session_env

    def read(name: str) -> str:
        return str(get_session_env(name, "") or "")

    route = {
        "session_key": read("HERMES_SESSION_KEY"),
        "session_id": session_id or read("HERMES_SESSION_ID"),
        "platform": read("HERMES_SESSION_PLATFORM"),
        "chat_id": read("HERMES_SESSION_CHAT_ID"),
        "chat_type": read("HERMES_SESSION_CHAT_TYPE"),
        "thread_id": read("HERMES_SESSION_THREAD_ID"),
        "scope_id": read("HERMES_SESSION_SCOPE_ID"),
        "user_id": read("HERMES_SESSION_USER_ID"),
        "user_name": read("HERMES_SESSION_USER_NAME"),
        "ui_session_id": read("HERMES_UI_SESSION_ID"),
    }
    if not route["session_key"]:
        route["session_key"] = route["session_id"]
    return route


def _route_kwargs(route: Mapping[str, str]) -> dict[str, Any]:
    return {name: str(route.get(name) or "") for name in (
        "session_key", "session_id", "platform", "chat_id", "chat_type", "thread_id",
        "scope_id", "user_id", "user_name", "ui_session_id")}


def _active_profile_name() -> str:
    from hermes_cli.profiles import get_active_profile_name
    return str(get_active_profile_name() or "default")


# ── internals: validation ───────────────────────────────────────────────────

def _clean_plugin_id(plugin_id: Any) -> str:
    text = str(plugin_id or "").strip()
    if not text or len(text) > _MAX_PLUGIN_ID_CHARS:
        raise CompletionError("plugin_id must be a non-empty string of at most 128 characters.")
    return text


def _clean_session_id(session_id: Any) -> str:
    if session_id is None:
        return ""
    if not isinstance(session_id, str):
        raise CompletionTargetError("session_id must be a string.")
    return session_id.strip()


def _clean_event_id(event_id: Any) -> str:
    if not isinstance(event_id, str) or not event_id.strip():
        raise CompletionRequestError("event_id must be a non-empty string.")
    text = event_id.strip()
    if len(text) > _MAX_EVENT_ID_CHARS:
        raise CompletionRequestError(f"event_id must be at most {_MAX_EVENT_ID_CHARS} characters.")
    return text


def _clean_content(content: Any) -> str:
    if not isinstance(content, str) or not content.strip():
        raise CompletionRequestError("content must be a non-empty string.")
    if len(content) > _MAX_CONTENT_CHARS:
        raise CompletionRequestError(f"content must be at most {_MAX_CONTENT_CHARS} characters.")
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(content)
    except Exception:  # pragma: no cover - redaction is a guard, not a hard dependency
        logger.debug("Content redaction unavailable; publishing as given", exc_info=True)
        return content


def _clean_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise CompletionRequestError("metadata must be a mapping.")
    value = dict(metadata)
    try:
        encoded = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError) as exc:
        raise CompletionRequestError("metadata must be JSON-serializable.") from exc
    if len(encoded.encode()) > _MAX_METADATA_BYTES:
        raise CompletionRequestError(f"metadata must be at most {_MAX_METADATA_BYTES} bytes.")
    return value


def _coerce_target(target: Any) -> CompletionTarget:
    if isinstance(target, CompletionTarget):
        target.validate()
        return target
    if isinstance(target, Mapping):
        return CompletionTarget.from_dict(target)
    raise CompletionTargetError("target must be a CompletionTarget or its to_dict() mapping.")


# ── internals: storage ──────────────────────────────────────────────────────

def _db_path():
    return get_hermes_home() / "state.db"


# One ordered tuple for the target row: the SELECT list and the dataclass field zip must not drift
# (a positional mismatch would silently hand a session_key the profile home).
_TARGET_COLUMNS = (
    "target_id", "plugin_id", "session_key", "session_id", "platform", "chat_id", "chat_type",
    "thread_id", "scope_id", "user_id", "user_name", "ui_session_id", "profile", "profile_home",
    "created_at", "expires_at",
)
_SELECT_TARGET = (
    "SELECT " + ", ".join(_TARGET_COLUMNS) + " FROM plugin_completion_targets WHERE target_id=?")


_dedup_ready: set[str] = set()


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db
    from hermes_constants import mkdir_under_hermes_home as _mkdir
    from hermes_state import _secure_state_db_files
    from hermes_state_schema import reconcile_state_schema

    path = _db_path()
    _mkdir(path.parent)
    _secure_state_db_files(path, create_main=True)
    conn = open_db(path, db_label="state.db (plugin completions)", busy_timeout_ms=10_000,
                   wal=False, row_factory=None, initialize=reconcile_state_schema)
    # The dedup index names a reconciled column, so it cannot live in SCHEMA_SQL. Reconciling
    # callers create it; an already-open database (or a path that skipped the initializer) may not
    # have it, and without the constraint a re-publish inserts a second row. One attempt per path
    # per process, because a database that predates the index may already contain duplicates.
    key = str(path)
    if key not in _dedup_ready:
        _dedup_ready.add(key)
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_plugin_completions_dedup "
                "ON plugin_completions(plugin_id, conversation_id, event_id)")
            conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("Plugin completion dedup index unavailable in %s: %s", path, exc)
    _secure_state_db_files(path)
    return conn


def _transaction():
    from hermes_cli.sqlite_util import transaction
    return transaction(_connect())


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='plugin_completions'").fetchone()
    return row is not None


def _schema_ready(conn: sqlite3.Connection) -> bool:
    return _table_exists(conn) and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='plugin_completion_targets'"
    ).fetchone() is not None


def _insert_target(target: CompletionTarget) -> None:
    with _DB_LOCK, _transaction() as conn:
        if not _schema_ready(conn):
            raise CompletionError("state.db cannot store plugin completion targets.")
        conn.execute(
            """INSERT INTO plugin_completion_targets
                   (target_id, plugin_id, profile_home, session_key, session_id, platform, chat_id,
                    chat_type, thread_id, scope_id, user_id, user_name, ui_session_id, profile,
                    created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (target.target_id, target.plugin_id, target.profile_home, target.session_key,
             target.session_id, target.platform, target.chat_id, target.chat_type, target.thread_id,
             target.scope_id, target.user_id, target.user_name, target.ui_session_id,
             target.profile, target.created_at, target.expires_at))


def _conversation_id(target: CompletionTarget) -> str:
    """Dedup identity of the CONVERSATION a target belongs to.

    A TUI tab carries ``ui_session_id``; a CLI or gateway conversation carries ``session_id``.
    Context compression keeps the id; ``/new`` starts a different one. Only a bare chat route with
    no conversation id falls back to ``session_key``, where dedup stays chat-wide as before.
    """
    return target.ui_session_id or target.session_id or target.session_key or ""


def _load_target(target_id: str) -> Optional[CompletionTarget]:
    with _DB_LOCK, _transaction() as conn:
        if not _schema_ready(conn):
            return None
        row = conn.execute(_SELECT_TARGET, (target_id,)).fetchone()
    if row is None:
        return None
    return CompletionTarget(contract_version=PUBLIC_CONTRACT_VERSION,
                            **dict(zip(_TARGET_COLUMNS, row)))


def _insert_completion(
    completion_id: str, plugin_id: str, target: CompletionTarget, event_id: str,
    event: Mapping[str, Any], now: float,
) -> bool:
    """Insert one pending row; False when ``(plugin, session, event_id)`` already exists."""
    payload = json.dumps(dict(event), default=str)
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO plugin_completions
                   (completion_id, target_id, plugin_id, session_key, conversation_id, event_id,
                    event_json, created_at, updated_at, delivery_state, delivery_attempts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0)""",
            (completion_id, target.target_id, plugin_id, target.session_key,
             _conversation_id(target), event_id, payload, now, now))
        return cur.rowcount == 1


def _existing_completion(plugin_id: str, conversation_id: str,
                         event_id: str) -> Optional[dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return None
        row = conn.execute(
            """SELECT completion_id, plugin_id, event_id, delivery_state, delivery_attempts,
                      created_at, delivered_at
                 FROM plugin_completions
                WHERE plugin_id=? AND conversation_id=? AND event_id=?""",
            (plugin_id, conversation_id, event_id)).fetchone()
    if row is None:
        return None
    keys = ("completion_id", "plugin_id", "event_id", "delivery_state", "delivery_attempts",
            "created_at", "delivered_at")
    return dict(zip(keys, row))


def _receipt(row: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "completion_id": row["completion_id"],
        "event_id": row["event_id"],
        "delivery_state": row["delivery_state"],
        "delivery_attempts": row["delivery_attempts"],
        "accepted_at": row["created_at"],
        "delivered_at": row["delivered_at"],
        "detail": (
            "This event id was already published for this session; nothing was queued again. "
            "A delivered row is never re-notified."),
    }


def _prune() -> None:
    """Bound the ledger: delivered rows age out, and pending rows are capped."""
    cutoff = time.time() - _RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        if not _table_exists(conn):
            return
        conn.execute("DELETE FROM plugin_completions WHERE delivery_state='delivered' AND updated_at<?",
                     (cutoff,))
        # Prune targets by their own expiry, never by capture age: a target captured for a long
        # background watch is still live at day 7. Legacy rows (expires_at IS NULL) have no
        # declared expiry, so they are only aged out once nothing pending references them --
        # deleting one would turn a legitimate publish into "unknown completion target".
        conn.execute(
            """DELETE FROM plugin_completion_targets
                WHERE (expires_at IS NOT NULL AND expires_at < ?)
                   OR (expires_at IS NULL AND created_at < ?
                       AND target_id NOT IN (SELECT target_id FROM plugin_completions
                                              WHERE delivery_state='pending'))""",
            (cutoff, cutoff))
        count = conn.execute(
            "SELECT COUNT(*) FROM plugin_completions WHERE delivery_state='pending'").fetchone()[0]
        if count > _MAX_PENDING_ROWS:
            conn.execute(
                """DELETE FROM plugin_completions WHERE completion_id IN (
                       SELECT completion_id FROM plugin_completions
                        WHERE delivery_state='pending'
                        ORDER BY created_at ASC LIMIT ?)""", (count - _MAX_PENDING_ROWS,))


def _decode_event(payload: str) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ── internals: queue + offer bookkeeping ────────────────────────────────────

def _offer(event: Mapping[str, Any], profile_home: str) -> None:
    with _offer_lock:
        _offered.add((hermes_home_key(profile_home), str(event.get("completion_id") or "")))
    from tools.process_registry import process_registry
    process_registry.completion_queue.put(dict(event))


def _mark_offered(home: str, completion_id: str, *, force: bool = False) -> bool:
    with _offer_lock:
        key = (home, completion_id)
        if key in _offered and not force:
            return False
        _offered.add(key)
        return True


def _forget_offer(completion_id: str) -> None:
    with _offer_lock:
        _offered.difference_update({key for key in _offered if key[1] == completion_id})


class _profile_scope:
    """Bind ``profile_home`` for the state.db write, restoring the caller's scope afterwards."""

    def __init__(self, profile_home: str) -> None:
        self._home = profile_home
        self._token = None

    def __enter__(self):
        if self._home and get_hermes_home_override() != self._home:
            self._token = set_hermes_home_override(self._home)
        return self

    def __exit__(self, *_exc) -> None:
        if self._token is not None:
            reset_hermes_home_override(self._token)
