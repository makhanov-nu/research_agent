"""Tests for the two-track scored consortium: pure helpers + session flow."""

from __future__ import annotations

from research_agent.config import settings
from research_agent.consortium import capture_council
from research_agent.consortium.consortium import (
    Consortium,
    build_document,
    idea_synopsis,
    normalize_and_rank,
    parse_ideas,
    parse_scores,
    render_transcript,
)


# --- pure helpers ------------------------------------------------------------

def test_parse_ideas_splits_on_marker_and_caps():
    text = (
        "=== IDEA ===\nFirst\n=== IDEA ===\nSecond\n"
        "=== IDEA ===\nThird\n=== IDEA ===\nFourth"
    )
    assert parse_ideas(text, max_n=3) == ["First", "Second", "Third"]


def test_parse_ideas_fallback_and_error_sentinel():
    assert parse_ideas("one idea, no marker") == ["one idea, no marker"]
    assert parse_ideas("[deepseek/x could not respond: boom]") == []
    assert parse_ideas("") == []


def test_parse_scores_prefers_json_filters_and_clamps():
    # ids outside the pool are dropped; scores clamp to [0, 10].
    assert parse_scores('{"1": 8, "2": 3, "9": 5}', {1, 2}) == {1: 8.0, 2: 3.0}
    assert parse_scores('{"1": 99, "2": -4}', {1, 2}) == {1: 10.0, 2: 0.0}


def test_parse_scores_falls_back_to_lines():
    assert parse_scores("#1: 7\n#2 - 4", {1, 2}) == {1: 7.0, 2: 4.0}


def test_normalize_and_rank_normalizes_across_raters():
    pool = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    scores = {
        "lenient": {1: 9, 2: 8, 3: 10},  # everything high
        "harsh": {1: 2, 2: 1, 3: 4},     # everything low
    }
    ranked = normalize_and_rank(pool, scores)
    # idea 3 is best for both raters -> rank 1; idea 2 worst -> last.
    assert [i["id"] for i in ranked] == [3, 1, 2]
    assert ranked[0]["score"] == 7.0  # raw mean of 10 and 4
    assert ranked[0]["n_scores"] == 2


def test_idea_synopsis_and_document():
    assert idea_synopsis("short") == "short"
    assert idea_synopsis("x " * 500, limit=20).endswith("…")
    doc = build_document("T", [("Top", "the idea"), ("Round 1 — idea debate", "the talk")])
    assert "# Research ideas — T" in doc
    assert "## Top" in doc and "the idea" in doc and "idea debate" in doc


def test_render_transcript_empty_and_labeled():
    assert render_transcript([]) == "(no discussion yet)"
    assert "[m/a]:" in render_transcript([("m/a", "hi")])


def test_panel_models_parsing(monkeypatch):
    monkeypatch.setattr(settings, "consortium_models", "a/x , b/y,, c/z ")
    assert settings.panel_models == ["a/x", "b/y", "c/z"]


# --- session flow (fakes: phase methods stubbed, no LLM) ---------------------

class _FakeConsortium(Consortium):
    """Stubs the phase methods so the state machine can be tested without models."""

    def __init__(self, tmp_path):
        super().__init__([], ["m/a", "m/b", "m/c", "m/d"], "chair/x", str(tmp_path))

    async def make_brief(self, topic, focus=""):
        return f"brief for {topic}", []

    async def propose_independent(self, brief):
        ideas = [
            {"text": f"{m} idea {k}", "source": "independent", "by": m}
            for m in self.panel for k in (1, 2, 3)
        ]
        return ideas, []

    async def debate_ideas(self, brief, prior=""):
        ideas = [{"text": f"{m} debated", "source": "debated", "by": m} for m in self.panel]
        return ideas, [("Brief", brief), ("m/a", "...")], []

    async def score_pool(self, pool, instruction=None):
        # Lower id -> higher score, so ranking is deterministic.
        scores = {m: {i["id"]: max(0, 10 - i["id"]) for i in pool} for m in self.panel}
        return scores, []

    async def vote_pool(self, pool):
        return await self.score_pool(pool)

    async def polish_idea(self, idea_text, instructions, prior=""):
        indep = [
            {"text": f"{m} polished {idea_text[:6]}", "source": "independent", "by": m}
            for m in self.panel
        ]
        debated = {"text": f"debated polish {idea_text[:6]}", "source": "debated", "by": "panel"}
        return indep, debated, [("x", "y")], []


async def test_round1_pools_16_ideas_and_returns_top5(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("topic")
    top = await session.run_round1()

    assert len(session.pool) == 16                      # 12 independent + 4 debated
    assert {i["id"] for i in session.pool} == set(range(1, 17))
    assert {i["source"] for i in session.pool} == {"independent", "debated"}
    assert [i["id"] for i in top] == [1, 2, 3, 4, 5]    # lower id scored higher
    assert session.phase == "scored"
    assert all("score" in i for i in top)


async def test_select_and_polish_votes_and_caps_to_5(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("topic")
    await session.run_round1()
    top = await session.select_and_polish(picks=[1, 2])

    # 2 ideas × (4 independent + 1 debated) = 10 proposals -> capped to top 5.
    assert len(session.pool) == 10
    assert len(top) == 5
    assert session.phase == "polished"
    assert session.round_no == 2
    assert [i["id"] for i in session.selected] == [1, 2]  # selection remembered


async def test_again_reuses_selection(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("topic")
    await session.run_round1()
    await session.select_and_polish(picks=[3])
    assert [i["id"] for i in session.selected] == [3]
    # 1 idea -> 5 proposals (<=5) -> all returned.
    assert len(session.top) == 5

    await session.select_and_polish(picks=None, comments="go deeper on proofs")
    assert [i["id"] for i in session.selected] == [3]     # unchanged
    assert session.round_no == 3


async def test_finalize_saves_document_with_debate(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("my topic")
    await session.run_round1()
    result = await session.finalize()

    assert session.finalized and result["rounds"] == 1
    assert result["top"] == session.top
    saved = next((tmp_path / "ideas").glob("*.md")).read_text()
    assert "# Research ideas — my topic" in saved
    assert "idea debate" in saved.lower()                 # debate transcript persisted


async def test_ideate_runs_round1_only_non_interactive(tmp_path):
    result = await _FakeConsortium(tmp_path).ideate("topic")
    assert result["rounds"] == 1
    assert len(result["top"]) == 5
    assert result["rel_path"].startswith("ideas/")


# --- memory capture ----------------------------------------------------------

async def test_capture_council_records_experience_and_lesson():
    class _Mem:
        def __init__(self):
            self.exp, self.lessons = [], []

        async def log_experience(self, kind, summary, channel_id=None, metadata=None):
            self.exp.append((kind, summary, channel_id, metadata))

        async def record_lesson(self, text, *, kind, channel_id=None, **kw):
            self.lessons.append((kind, text, channel_id))

    mem = _Mem()
    await capture_council(mem, "chan-1", "specdec", "the ideas", "ideas/x.md", rounds=2)
    assert mem.exp and mem.exp[0][0] == "council_session"
    assert "converged after 2 round(s)" in mem.exp[0][1]
    assert mem.exp[0][2] == "chan-1"
    assert mem.exp[0][3] == {"topic": "specdec", "rel_path": "ideas/x.md"}
    assert mem.lessons and mem.lessons[0][0] == "council" and mem.lessons[0][2] == "chan-1"
    await capture_council(None, "c", "t", "i")  # no-op without memory
