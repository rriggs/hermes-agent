"""``PluginContext`` delivery seams: how a plugin puts content into the conversation that invoked it.

Two paths, deliberately different:

* **Live injection** (``inject_message``) queues a turn through the gateway / TUI injector the host
  registered for this plugin's profile. It only works while that surface is running the session.
* **Durable completion publishing** (``capture_completion_target`` + ``publish_completion``) binds a
  target during the tool call and delivers a result later through ``agent/plugin_completions.py``,
  surviving a restart. It is the supported path for background work.

They live in this sibling rather than in ``hermes_cli/plugins.py`` because the facade stays a
registry: delivery behaviour belongs at the edge (root AGENTS.md). ``PluginContext`` inherits this
mixin, so ``hasattr``/``dir`` and the plugin-host method announcement are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


class PluginSessionDeliveryMixin:
    """Mixin of :class:`hermes_cli.plugins.PluginContext`; expects ``plugin_id``/``_manager``."""

    def inject_message(
        self, content: str, role: str = "user", *, session_key: str | None = None,
        origin: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue *content* for a session (``session_key``) or a chat in this plugin's own profile
        (``origin``); ``True`` means a host accepted it. Contract: plugins_injection.py, plugin docs."""
        from hermes_cli.plugins_injection import inject_plugin_message
        return inject_plugin_message(self, content, role, session_key=session_key, origin=origin)

    def queue_context_note(self, text: str, *, key: str = "") -> bool:
        """Queue a model-visible note for the current session without starting a turn.

        Rides the session's next tool result or user turn (``agent/plugin_live_notes.py``), so the
        model sees it and the cached prefix is untouched. Returns False when no live agent is
        reachable from this process (the caller may surface the same text in a tool result).
        """
        if not isinstance(text, str) or not text.strip():
            return False
        agent = getattr(getattr(self._manager, "_cli_ref", None), "agent", None)
        if agent is None:
            return False
        from agent.plugin_live_notes import queue_plugin_notice
        queue_plugin_notice(agent, key or self.plugin_id, text.strip())
        return True

    def session_cwd(self) -> str:
        """The calling session's recorded cwd, never the gateway's own (``plugin_session``)."""
        from hermes_cli.plugin_session import context_cwd
        return context_cwd(self._manager)

    def session_surface(self) -> str:
        """The calling session's client (``cli``/``tui``/``desktop``/``acp``) or messaging platform."""
        from hermes_cli.plugin_session import context_surface
        return context_surface(self._manager)

    def _gateway_injection_allowed(self) -> bool:
        """Return whether this plugin may trigger gateway session turns (read from the plugin's
        own profile config, never the calling thread's ambient home)."""
        from hermes_cli.config import load_config_readonly
        from hermes_cli.plugins_loader import _plugin_home_scope
        from hermes_cli.plugins_state import _plugin_settings_entry
        try:
            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config_readonly() or {}
        except Exception:  # a config read failure must deny, never grant
            logger.debug("Plugin %s: gateway-injection consent unreadable", self.plugin_id,
                         exc_info=True)
            return False
        return (_plugin_settings_entry(cfg, self.plugin_id) or {}).get("allow_gateway_injection") is True

    # -- durable completions -------------------------------------------------
    def capture_completion_target(self, *, session_id: Optional[str] = None) -> dict:
        """Bind a durable delivery target to THIS tool invocation's session and route.

        Returns an opaque, JSON-safe mapping the plugin may persist (its target survives a process
        restart). Pass the dispatcher-provided ``session_id`` when the handler receives one: a value
        that disagrees with the bound session is refused, so a plugin can never capture a target for
        a session it is not running in. Raises ``CompletionDeliveryUnsupported`` where no later turn
        can reach the session (one-shot, cron, Kanban or managed worker, stateless API session).

        Pairs with :meth:`publish_completion`; the pair is the only supported way for a plugin to
        deliver an async result into the conversation that started the work.
        """
        from agent.plugin_completions import capture_completion_target
        return capture_completion_target(self.plugin_id, session_id=session_id).to_dict()

    def publish_completion(
        self, *, target: Any, event_id: str, content: str, metadata: Optional[dict] = None,
    ) -> dict:
        """Persist a completion for ``target`` and queue it for the target session's next idle turn.

        ``event_id`` is the caller's idempotency key: re-publishing it for the same session (a
        restarted watcher re-observing the same terminal state) returns ``status='duplicate'`` and
        queues nothing. The receipt's ``status='queued'`` means queue admission only -- not that the
        session ran the content or that a reply was sent. Raises for a malformed request, a content
        or metadata over the documented limits, or a target that is unknown, expired, or belongs to
        another plugin.
        """
        from agent.plugin_completions import publish_completion
        return publish_completion(
            self.plugin_id, target=target, event_id=event_id, content=content, metadata=metadata)
