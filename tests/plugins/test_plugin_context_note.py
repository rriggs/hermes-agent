"""``ctx.queue_context_note``: a plugin tells the model something without starting a turn.

The A2A plugin uses it on session resume to report reclaimed and lost task ownership. The note
must reach the model through the live-notes path (next tool result or user turn), and it must not
be silently dropped when no live agent is reachable -- it returns False so the caller can surface
the same text itself.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.plugin_live_notes import pop_plugin_notices
from hermes_cli.plugins_session_delivery import PluginSessionDeliveryMixin


class _Ctx(PluginSessionDeliveryMixin):
    plugin_id = "a2a-async"


def _ctx_with(agent):
    ctx = _Ctx()
    ctx._manager = SimpleNamespace(_cli_ref=SimpleNamespace(agent=agent))
    return ctx


def test_a_context_note_reaches_the_agents_next_turn():
    agent = SimpleNamespace()
    ctx = _ctx_with(agent)

    assert ctx.queue_context_note("A2A ownership reclaimed: t1") is True

    assert "A2A ownership reclaimed: t1" in pop_plugin_notices(agent)
    assert pop_plugin_notices(agent) == "", "a note is delivered once"


def test_a_note_needs_a_live_agent_and_real_text():
    assert _ctx_with(None).queue_context_note("anything") is False
    assert _ctx_with(None).queue_context_note("   ") is False
