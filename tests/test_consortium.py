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


def test_parse_ideas_drops_preamble_before_first_marker():
    # A model's "Sure, here are my ideas:" preamble must NOT become idea #1.
    text = "Sure, here you go:\n=== IDEA ===\nA\n=== IDEA ===\nB\n=== IDEA ===\nC"
    assert parse_ideas(text, max_n=3) == ["A", "B", "C"]
    # The single-extract (debate) path returns the idea, not the preamble.
    assert parse_ideas("My best:\n=== IDEA ===\nThe Idea", max_n=1) == ["The Idea"]


def test_parse_ideas_rejects_output_without_marker():
    # Missing the contract marker is an output-contract violation now, not a
    # fallback to "treat the whole blob as one idea" — it's rejected outright.
    assert parse_ideas("one idea, no marker") == []
    assert parse_ideas("[deepseek/x could not respond: boom]") == []
    assert parse_ideas("") == []


def test_parse_scores_prefers_json_filters_and_clamps():
    # ids outside the pool are dropped; scores clamp to [0, 10]; no reason
    # lines present here, so reasons default to "".
    assert parse_scores('{"1": 8, "2": 3, "9": 5}', {1, 2}) == {1: (8.0, ""), 2: (3.0, "")}
    assert parse_scores('{"1": 99, "2": -4}', {1, 2}) == {1: (10.0, ""), 2: (0.0, "")}


def test_parse_scores_falls_back_to_lines_and_captures_reason():
    assert parse_scores("#1: 7\n#2 - 4", {1, 2}) == {1: (7.0, "7"), 2: (4.0, "")}
    text = '{"1": 8, "2": 3}\n#1: groundbreaking formalism\n#2: incremental'
    parsed = parse_scores(text, {1, 2})
    assert parsed == {1: (8.0, "groundbreaking formalism"), 2: (3.0, "incremental")}


def test_normalize_and_rank_normalizes_across_raters():
    pool = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    scores = {
        "lenient": {1: (9, "ok"), 2: (8, "ok"), 3: (10, "great")},  # everything high
        "harsh": {1: (2, "weak"), 2: (1, "weak"), 3: (4, "least bad")},  # everything low
    }
    ranked = normalize_and_rank(pool, scores)
    # idea 3 is best for both raters -> rank 1; idea 2 worst -> last.
    assert [i["id"] for i in ranked] == [3, 1, 2]
    assert ranked[0]["score"] == 7.0  # raw mean of 10 and 4
    assert ranked[0]["n_scores"] == 2
    assert {r["model"] for r in ranked[0]["raters"]} == {"lenient", "harsh"}
    assert all(r["reason"] for r in ranked[0]["raters"])


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
        threads = {m: [f"propose-thread-{m}"] for m in self.panel}
        return ideas, threads, []

    async def debate_ideas(self, brief, prior=""):
        # Chair-extracted: 2 ownerless ideas, not one per panelist.
        ideas = [{"text": "debated synthesis A", "source": "debated", "by": "panel"},
                 {"text": "debated synthesis B", "source": "debated", "by": "panel"}]
        return ideas, [("Brief", brief), ("m/a", "...")], []

    async def score_round1(self, pool, propose_threads):
        # Lower id -> higher score, so ranking is deterministic. No self-exclusion
        # in this stub — that bias-elimination behavior has its own real-code path.
        scores = {m: {i["id"]: (max(0, 10 - i["id"]), f"reason for {i['id']}") for i in pool}
                  for m in self.panel}
        return scores, [], []


async def test_round1_pools_14_ideas_no_cut(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("topic")
    top = await session.run_round1()

    assert len(session.pool) == 14                      # 12 independent + 2 debated
    assert {i["id"] for i in session.pool} == set(range(1, 15))
    assert {i["source"] for i in session.pool} == {"independent", "debated"}
    assert len(top) == 14                                # no top-N cut
    assert [i["id"] for i in top[:5]] == [1, 2, 3, 4, 5]  # lower id scored higher
    assert session.phase == "scored"
    assert all("score" in i for i in top)
    assert all(i["raters"] for i in top)                 # per-rater score+reason kept


async def test_finalize_saves_document_with_debate(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("my topic")
    await session.run_round1()
    result = await session.finalize()

    assert session.finalized and result["rounds"] == 1
    assert result["top"] == session.top
    main = next(p for p in (tmp_path / "ideas").glob("*.md") if not p.name.endswith("-references.md"))
    saved = main.read_text()
    assert "# Research ideas — my topic" in saved
    assert "idea debate" in saved.lower()                 # debate transcript persisted
    assert "Bibliography" in saved                        # per-idea bibliography section
    refs_path = tmp_path / "ideas" / result["rel_references_path"].split("/", 1)[1]
    assert refs_path.exists()


async def test_ideate_runs_round1_only_non_interactive(tmp_path):
    result = await _FakeConsortium(tmp_path).ideate("topic")
    assert result["rounds"] == 1
    assert len(result["top"]) == 14                       # no top-N cut
    assert result["rel_path"].startswith("ideas/")
    assert result["rel_references_path"].startswith("ideas/")
    assert result["rel_references_path"].endswith("-references.md")


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
