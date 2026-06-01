"""Regression tests for the Codex-review issue fixes (#1-#5)."""

from __future__ import annotations

import types

import pytest

from research_agent.discord_bot.bot import PerKeyLocks, checkpoint_result_message
from research_agent.memory.episodic import EpisodicStore
from research_agent.memory.manager import MemoryManager


class _FakeEpisodic:
    """Records calls; can be told to fail on log_action."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple] = []

    async def touch_channel(self, channel_id, total_tokens):
        self.calls.append(("touch_channel", channel_id, total_tokens))

    async def set_summary(self, channel_id, summary):
        self.calls.append(("set_summary", channel_id, summary))

    async def log_action(self, kind, summary, channel_id=None, metadata=None):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(("log_action", kind, channel_id))


def _manager_with(episodic) -> MemoryManager:
    m = MemoryManager(None)
    m.episodic = episodic
    m.semantic = types.SimpleNamespace(enabled=False)
    return m


# --- #1: auto-summary is persisted to the episodic store -----------------------

async def test_remember_persists_summary_when_provided():
    fake = _FakeEpisodic()
    await _manager_with(fake).remember("c1", "u", "a", 100, summary="ROLLING SUMMARY")
    assert ("set_summary", "c1", "ROLLING SUMMARY") in fake.calls


async def test_remember_skips_summary_when_empty():
    fake = _FakeEpisodic()
    await _manager_with(fake).remember("c1", "u", "a", 100, summary="")
    assert not any(c[0] == "set_summary" for c in fake.calls)


# --- #2: experiment field allowlist -------------------------------------------

async def test_update_experiment_rejects_unknown_field():
    store = EpisodicStore(None)
    with pytest.raises(ValueError, match="Unknown experiment field"):
        await store.update_experiment(1, status="done", bogus="x")


async def test_update_experiment_accepts_known_field_noop_without_pool():
    store = EpisodicStore(None)  # disabled (no pool) -> returns after validation
    assert await store.update_experiment(1, status="done", metrics={"acc": 0.9}) is None


# --- #3: background persistence logs and swallows failures ---------------------

async def test_remember_swallows_and_logs_failure(caplog):
    fake = _FakeEpisodic(fail=True)
    # Must not raise even though log_action blows up.
    await _manager_with(fake).remember("c1", "u", "a")
    assert "Background memory persistence failed" in caplog.text


# --- #4: honest checkpoint messaging ------------------------------------------

def test_checkpoint_message_no_memory():
    msg = checkpoint_result_message(memory_configured=False, semantic_saved=False)
    assert "isn't configured" in msg


def test_checkpoint_message_semantic_disabled():
    msg = checkpoint_result_message(memory_configured=True, semantic_saved=False)
    assert "disabled" in msg and "conversation store" in msg


def test_checkpoint_message_semantic_saved():
    msg = checkpoint_result_message(memory_configured=True, semantic_saved=True)
    assert "long-term" in msg


# --- #5: per-channel locks -----------------------------------------------------

def test_perkeylocks_same_key_same_lock():
    locks = PerKeyLocks()
    assert locks.get("a") is locks.get("a")


def test_perkeylocks_distinct_keys_distinct_locks():
    locks = PerKeyLocks()
    assert locks.get("a") is not locks.get("b")
