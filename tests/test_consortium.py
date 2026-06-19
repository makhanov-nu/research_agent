"""Tests for the scored consortium: pure helpers + the StateGraph round flow."""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from research_agent.config import settings
from research_agent.consortium import capture_council
from research_agent.consortium.consortium import (
    Consortium,
    _strip_reasoning,
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


# --- parser hardening: reasoning-block leak + scaffold filter (council_31) ----

def test_strip_reasoning_removes_think_block_keeps_following_content():
    text = "<think>\nscratchpad: planning my answer\n</think>\nThe real answer."
    assert _strip_reasoning(text).strip() == "The real answer."
    # No think tags -> strict no-op (this is what keeps every other test green).
    assert _strip_reasoning("plain text, no tags") == "plain text, no tags"
    # A stray lone closing tag (truncated block) is also removed.
    assert _strip_reasoning("leftover</think>\nanswer").strip() == "leftover\nanswer"


def test_parse_ideas_ignores_idea_marker_inside_think_block():
    # Reproduces council_31: the reasoning-model chair restates the output FORMAT
    # inside <think> -- both as a line-start "=== IDEA ===" header (with
    # [placeholder] template) AND as a quoted "=== IDEA ===" mid-sentence -- then
    # writes the two REAL ideas after </think>. Without the think-strip, the
    # in-think header is line-anchored too, so it would split out the scaffold as
    # ideas #1/#2 and the real ideas would fall outside the [:2] slice.
    text = (
        "<think>\n"
        "Let me recall the format: the spec says each idea begins with a quoted\n"
        "\"=== IDEA ===\" marker inline, then a Title and Method.\n"
        "=== IDEA ===\n"
        "a rough scratch note about federated aggregation, nothing real yet\n"
        "=== IDEA ===\n"
        "Title: [Concise name]\n"
        "Problem & motivation: [1-2 sentences]\n"
        "</think>\n"
        "=== IDEA ===\n"
        "Title: Federated Curvature-Penalized Aggregation\n"
        "Method: penalize the Hessian trace across client updates.\n"
        "=== IDEA ===\n"
        "Title: Dual-Encoder Cross-Modal Retrieval\n"
        "Method: contrastive alignment with a shared projection head.\n"
    )
    ideas = parse_ideas(text, max_n=2, model="chair")
    assert len(ideas) == 2
    assert ideas[0].startswith("Title: Federated Curvature-Penalized Aggregation")
    assert ideas[1].startswith("Title: Dual-Encoder Cross-Modal Retrieval")
    assert all("[Concise name]" not in i for i in ideas)
    assert all("scratch note" not in i for i in ideas)


def test_parse_ideas_ignores_inline_quoted_marker_mid_sentence():
    # An inline quoted "=== IDEA ===" mid-sentence (not at line start) and no
    # real marker -> no spurious idea (the split is line-anchored).
    text = 'Each idea starts with "=== IDEA ===" and includes a title.'
    assert parse_ideas(text, max_n=2) == []


def test_parse_ideas_drops_residual_template_scaffold_segment():
    # Belt-and-suspenders: a VISIBLE leak (no think tags to strip) full of
    # [placeholder] tokens is dropped by _looks_like_idea before the cap, so the
    # one real idea after it still survives.
    text = (
        "=== IDEA ===\n"
        "Title: [Concise name]\n"
        "Problem & motivation: [1-2 sentences]\n"
        "Method: [Key equations and derivation]\n"
        "=== IDEA ===\n"
        "Title: Sparse Mixture Routing\n"
        "Method: route tokens by a learned gating temperature.\n"
    )
    ideas = parse_ideas(text, max_n=2)
    assert ideas == [
        "Title: Sparse Mixture Routing\nMethod: route tokens by a learned gating temperature."
    ]


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


def test_parse_scores_ignores_json_inside_think_block():
    # Example JSON inside the reasoning scratchpad must not be read as a ballot;
    # the REAL scores here are #N: lines only, so without the strip-call the
    # in-think {"1": 1, "2": 1} would win via the JSON-first gate.
    text = (
        '<think>The format is a JSON object like {"1": 1, "2": 1}. Now let me\n'
        'actually judge these ideas.</think>\n'
        '#1: 8\n#2: 3'
    )
    assert parse_scores(text, {1, 2}) == {1: (8.0, "8"), 2: (3.0, "3")}


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


def test_normalize_and_rank_drops_zero_rater_idea():
    # An idea nobody scored must be ABSENT from the result, not shipped at a
    # fictitious 0.0 "consensus" score. Scored ideas remain.
    pool = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    scores = {
        "r1": {1: (7, "ok"), 3: (5, "meh")},  # idea 2 scored by nobody
        "r2": {1: (8, "good"), 3: (4, "weak")},
    }
    ranked = normalize_and_rank(pool, scores)
    assert {i["id"] for i in ranked} == {1, 3}  # idea 2 dropped
    assert all(i["n_scores"] >= 1 for i in ranked)


def test_normalize_and_rank_council_31_collapse_drops_unscored():
    # Reproduce the council_31 shape: only a subset of raters voted, with
    # disjoint coverage, and at least one idea got zero raters. The zero-rater
    # idea(s) must be dropped from the ranked output.
    pool = [{"id": i, "text": f"idea {i}"} for i in range(1, 6)]
    scores = {
        "qwen": {1: (6, "q-on-1"), 2: (7, "q-on-2")},      # only DeepSeek's ideas
        "deepseek": {3: (8, "d-on-3"), 4: (5, "d-on-4")},  # only Qwen's ideas
        # idea 5: zero raters (the unscored idea #8 of the real run).
    }
    ranked = normalize_and_rank(pool, scores)
    ids = {i["id"] for i in ranked}
    assert 5 not in ids                       # zero-rater idea dropped
    assert ids == {1, 2, 3, 4}
    assert all(i["n_scores"] >= 1 for i in ranked)


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


# --- StateGraph round flow (fakes: the `_chat` seam stubbed, no network) ------

@tool
def paperclip(command: str) -> str:
    """Fake literature search returning a fixed result (for the budget test)."""
    return (
        "Found 1 papers  [s_fake]\n"
        "  1. A Fake Paper\n     Doe, J.\n"
        "     arx_2401.00009 · arXiv · 2024-01-09\n     https://x\n"
    )


def _assert_tool_sequence(messages):
    """Enforce the OpenAI/DeepInfra contract the real provider enforces: an
    assistant message with tool_calls must be followed by a tool result. Catches
    a budget-cutoff finalize that leaves a dangling tool_call (the 400 the
    `_budget_tool_answers` fix prevents) — without this the fakes false-green it.
    """
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if not isinstance(nxt, ToolMessage):
                raise ValueError(
                    "dangling tool_call not followed by a tool result "
                    f"(at index {i})"
                )


class _FakeChat:
    """Decides its reply from the last message; tool-bound => always search."""

    def __init__(self, model: str, with_tools: bool, *, votes: bool = True):
        self.model = model
        self.with_tools = with_tools
        self.votes = votes

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        _assert_tool_sequence(messages)
        text = getattr(messages[-1], "content", "") or ""
        if self.with_tools:  # the GLM-5.1 failure mode: never stops searching
            return AIMessage(content="", tool_calls=[{
                "name": "paperclip", "args": {"command": "search -s arxiv x"},
                "id": f"c{uuid.uuid4().hex[:8]}"}])
        if "Score EACH idea" in text or "JSON object mapping" in text:
            if not self.votes:
                return AIMessage(content="(I abstain.)")  # no parseable scores
            ids = re.findall(r"#(\d+):", text)
            return AIMessage(content=json.dumps({i: max(0, 10 - int(i)) for i in ids}))
        if "State your strongest research direction" in text:
            return AIMessage(content=f"{self.model}: strongest direction; open gap.")
        n = 2 if "EXACTLY 2" in text else 3
        return AIMessage(content="\n".join(
            f"=== IDEA ===\n**Title:** {self.model} idea {k}\n"
            f"**Method:** approach {k} `arx_2401.0000{k}`"
            for k in range(1, n + 1)))


class _FakeConsortium(Consortium):
    """Drives the real graph with a stubbed `_chat` seam (no network).

    `panel_voters=None` => every panelist casts a ballot; a set restricts which
    panelists vote (the chair always rates, as in production).
    """

    panel_voters: set | None = None

    def __init__(self, tmp_path, tools=None):
        super().__init__(tools or [], ["m/a", "m/b", "m/c", "m/d"], "chair/x", str(tmp_path))

    def _chat(self, model, *, max_tokens, with_tools=False):
        votes = (
            model == self.chair_model
            or self.panel_voters is None
            or model in self.panel_voters
        )
        return _FakeChat(model, with_tools, votes=votes)

    async def make_brief(self, topic, focus=""):
        return f"brief for {topic}", [
            {"type": "ai", "content": "<brief>", "speaker": "chair:brief"}]


async def test_round1_pools_14_ideas_scored_and_ranked(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("topic")
    top = await session.run_round1()

    assert len(session.pool) == 14                       # 12 independent + 2 debated
    assert {i["id"] for i in session.pool} == set(range(1, 15))
    assert {i["source"] for i in session.pool} == {"independent", "debated"}
    assert len(top) == 14                                # chair rates all -> none dropped
    assert session.phase == "scored"
    assert all("score" in i and i["raters"] for i in top)
    # chair-as-neutral-rater gives every idea an independent cross-score.
    assert all(i["n_scores"] >= 1 for i in top)
    # ranked best-first by the normalized key.
    norms = [i["score_norm"] for i in top]
    assert norms == sorted(norms, reverse=True)


async def test_finalize_saves_document_with_debate(tmp_path):
    session = _FakeConsortium(tmp_path).new_session("my topic")
    await session.run_round1()
    result = await session.finalize()

    assert session.finalized and result["rounds"] == 1
    assert result["top"] == session.top
    main = next(p for p in (tmp_path / "ideas").glob("*.md")
                if not p.name.endswith("-references.md"))
    saved = main.read_text()
    assert "# Research ideas — my topic" in saved
    assert "idea debate" in saved.lower()                 # debate transcript persisted
    assert "Bibliography" in saved                        # per-idea bibliography section
    refs_path = tmp_path / "ideas" / result["rel_references_path"].split("/", 1)[1]
    assert refs_path.exists()


async def test_ideate_runs_round1_only_non_interactive(tmp_path):
    result = await _FakeConsortium(tmp_path).ideate("topic")
    assert result["rounds"] == 1
    assert len(result["top"]) == 14
    assert result["rel_path"].startswith("ideas/")
    assert result["rel_references_path"].startswith("ideas/")
    assert result["rel_references_path"].endswith("-references.md")


async def test_budget_guard_terminates_relentless_searcher(tmp_path):
    # The fake model ALWAYS requests a tool (GLM-5.1's failure). The run must
    # still terminate — the budget edge forces a finalize instead of looping
    # into the recursion wall — and produce the full scored pool.
    session = _FakeConsortium(tmp_path, tools=[paperclip]).new_session("t")
    top = await session.run_round1()
    assert len(session.pool) == 14 and len(top) == 14


class _RaisingChat:
    """A model client that always raises — simulates a provider 429/timeout."""

    def __init__(self, model):
        self.model = model

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        raise RuntimeError(f"{self.model} engine_overloaded (429)")


class _BrokenModelConsortium(_FakeConsortium):
    """One panelist raises on every call; the round must isolate it, not abort."""

    broken = "m/c"

    def _chat(self, model, *, max_tokens, with_tools=False):
        if model == self.broken:
            return _RaisingChat(model)
        return super()._chat(model, max_tokens=max_tokens, with_tools=with_tools)


async def test_round1_isolates_a_failing_model(tmp_path):
    # A provider error on one model (the council_31 Kimi 429) must NOT crash the
    # round — it's flagged and the surviving models still produce a ranking.
    session = _BrokenModelConsortium(tmp_path).new_session("topic")
    top = await session.run_round1()  # must not raise

    assert top, "round should still rank ideas despite one model failing"
    # m/c authored nothing; the other three panelists each contributed 3.
    indep = [i for i in session.pool if i["source"] == "independent"]
    assert len(indep) == 9 and all(i["by"] != "m/c" for i in indep)
    flags = " ".join(session.flags)
    assert "m/c failed" in flags  # surfaced, not swallowed


class _CollapsedConsortium(_FakeConsortium):
    """council_31 collapse: only one panelist casts a ballot (the rest abstain)."""

    panel_voters = {"m/a"}


async def test_round1_low_panel_coverage_flags_but_chair_prevents_drops(tmp_path):
    session = _CollapsedConsortium(tmp_path).new_session("topic")
    top = await session.run_round1()

    # The chair rates everything, so no idea collapses to zero raters ...
    assert len(top) == 14
    assert all(i["n_scores"] >= 1 for i in top)

    flags = " ".join(session.flags)
    # ... but only 1 of 4 panelists cast a ballot -> loud low-confidence flag,
    # and no idea is dropped (chair coverage prevents zero-rater drops).
    assert "low-confidence ranking" in flags and "1/4" in flags
    assert "dropped" not in flags
    assert flags.count("failed to vote") >= 3   # m/b, m/c, m/d abstained


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
