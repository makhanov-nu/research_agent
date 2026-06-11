"""Tests for the auto-label and eval tooling.

Covers:
- derive_auto_label() over representative traces (valid/invalid/error verdicts,
  multiple verifiers, no signals, round tie-breaking).
- effective_label() export precedence (user > auto > judge).
- Freeze selection logic (label priority order).
- Judge JSON parsing (_parse_judge_response with mocked LLM, parse failure skipping).
"""

from __future__ import annotations

import json

import pytest

from research_agent.agents.task_store import derive_auto_label
from research_agent.evals.judge import _parse_judge_response
from research_agent.evals.harness import _label_priority, _parse_compare_response
from research_agent.training.export import effective_label


# ---------------------------------------------------------------------------
# derive_auto_label
# ---------------------------------------------------------------------------

class TestDeriveAutoLabel:
    def test_no_trace_signals_returns_none(self):
        label, signals = derive_auto_label([])
        assert label is None
        assert signals == {}

    def test_non_dict_entries_ignored(self):
        label, signals = derive_auto_label(["string", 42, None])
        assert label is None
        assert signals == {}

    def test_single_valid_verdict_no_artifact_returns_good(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid", "feedback": "ok"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert signals["verdicts"] == {"citation_check": "valid"}
        assert "missing_citations" not in signals

    def test_single_invalid_verdict_returns_bad(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "methodology_validator",
             "verdict": "invalid", "feedback": "bad method"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "bad"
        assert signals["verdicts"] == {"methodology_validator": "invalid"}

    def test_error_verdict_ignored_returns_none_when_only_signal(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "paper_verifier",
             "verdict": "error", "feedback": "exception"},
        ]
        label, signals = derive_auto_label(trace)
        assert label is None
        assert signals == {}

    def test_multiple_verifiers_all_valid_no_artifact_returns_good(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid"},
            {"type": "critique", "round": 1, "verifier": "methodology_validator",
             "verdict": "valid"},
            {"type": "critique", "round": 1, "verifier": "paper_verifier",
             "verdict": "valid"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert set(signals["verdicts"].keys()) == {
            "citation_check", "methodology_validator", "paper_verifier"
        }

    def test_any_final_invalid_verdict_returns_bad_even_if_others_valid(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid"},
            {"type": "critique", "round": 1, "verifier": "methodology_validator",
             "verdict": "invalid"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "bad"

    def test_higher_round_overrides_earlier_invalid_verdict(self):
        """If round 1 is invalid but round 2 is valid, the final verdict is valid."""
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "invalid"},
            {"type": "critique", "round": 2, "verifier": "citation_check",
             "verdict": "valid"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert signals["verdicts"]["citation_check"] == "valid"

    def test_higher_round_overrides_earlier_valid_to_invalid(self):
        """Round 1 valid, round 2 invalid → final is invalid → bad."""
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid"},
            {"type": "critique", "round": 2, "verifier": "citation_check",
             "verdict": "invalid"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "bad"
        assert signals["verdicts"]["citation_check"] == "invalid"

    def test_error_in_later_round_does_not_override_valid(self):
        """Errors are always ignored, even at a higher round."""
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid"},
            {"type": "critique", "round": 2, "verifier": "citation_check",
             "verdict": "error"},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert signals["verdicts"]["citation_check"] == "valid"

    def test_artifact_with_zero_missing_citations_plus_valid_verdicts_is_good(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid"},
            {"type": "artifact", "tex": "out/paper.tex", "bib": "out/paper.bib",
             "missing_citations": []},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert signals["missing_citations"] == 0

    def test_artifact_with_missing_citations_returns_bad(self):
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "valid"},
            {"type": "artifact", "tex": "out/paper.tex", "bib": "out/paper.bib",
             "missing_citations": ["smith2020", "jones2021"]},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "bad"
        assert signals["missing_citations"] == 2

    def test_artifact_only_no_critiques_zero_missing_returns_good(self):
        """An artifact with zero missing citations alone is a positive signal."""
        trace = [
            {"type": "artifact", "tex": "x.tex", "bib": "x.bib",
             "missing_citations": []},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert signals["missing_citations"] == 0

    def test_artifact_only_with_missing_citations_returns_bad(self):
        trace = [
            {"type": "artifact", "tex": "x.tex", "bib": "x.bib",
             "missing_citations": ["a"]},
        ]
        label, signals = derive_auto_label(trace)
        assert label == "bad"

    def test_last_artifact_entry_is_used(self):
        """When multiple artifact entries exist, the last one wins."""
        trace = [
            {"type": "artifact", "missing_citations": ["old"]},
            {"type": "artifact", "missing_citations": []},  # final: 0 missing
        ]
        label, signals = derive_auto_label(trace)
        assert label == "good"
        assert signals["missing_citations"] == 0

    def test_missing_citations_integer_in_artifact_counts_as_zero(self):
        """Non-list missing_citations value (e.g. int) is treated as 0."""
        trace = [
            {"type": "artifact", "missing_citations": 0},
        ]
        label, signals = derive_auto_label(trace)
        # non-list → treated as 0 by the `len(mc) if isinstance(mc, list) else 0` branch
        assert signals["missing_citations"] == 0

    def test_only_error_verdicts_plus_artifact_no_missing_returns_good(self):
        """Errors removed; artifact with 0 missing is still a positive signal."""
        trace = [
            {"type": "critique", "round": 1, "verifier": "citation_check",
             "verdict": "error"},
            {"type": "artifact", "missing_citations": []},
        ]
        label, signals = derive_auto_label(trace)
        # No valid/invalid verdicts, but artifact says 0 missing → good
        assert label == "good"
        assert signals["missing_citations"] == 0

    def test_round_none_treated_as_zero(self):
        """Entries without a round key should default to round 0."""
        trace = [
            {"type": "critique", "verifier": "citation_check", "verdict": "invalid"},
            {"type": "critique", "round": None, "verifier": "citation_check",
             "verdict": "valid"},
        ]
        # Both are round 0; the second (round=None→0) should win (>= comparison).
        label, signals = derive_auto_label(trace)
        assert label == "good"


# ---------------------------------------------------------------------------
# effective_label (export precedence)
# ---------------------------------------------------------------------------

class TestEffectiveLabel:
    def _row(self, quality=None, auto_quality=None, judge_score=None):
        return {
            "quality": quality,
            "auto_quality": auto_quality,
            "judge_score": judge_score,
        }

    def test_user_quality_wins_over_auto_and_judge(self):
        row = self._row(quality="good", auto_quality="bad", judge_score=1)
        label, source = effective_label(row)
        assert label == "good"
        assert source == "user"

    def test_user_bad_wins_over_auto_good(self):
        row = self._row(quality="bad", auto_quality="good", judge_score=5)
        label, source = effective_label(row)
        assert label == "bad"
        assert source == "user"

    def test_auto_quality_used_when_no_user_label(self):
        row = self._row(quality=None, auto_quality="good")
        label, source = effective_label(row)
        assert label == "good"
        assert source == "auto"

    def test_auto_bad_used_when_no_user_label(self):
        row = self._row(quality=None, auto_quality="bad")
        label, source = effective_label(row)
        assert label == "bad"
        assert source == "auto"

    def test_judge_score_gte_4_returns_good(self):
        for score in (4, 5):
            row = self._row(judge_score=score)
            label, source = effective_label(row)
            assert label == "good", f"score={score}"
            assert source == "judge"

    def test_judge_score_lte_2_returns_bad(self):
        for score in (1, 2):
            row = self._row(judge_score=score)
            label, source = effective_label(row)
            assert label == "bad", f"score={score}"
            assert source == "judge"

    def test_judge_score_3_returns_none(self):
        row = self._row(judge_score=3)
        label, source = effective_label(row)
        assert label is None
        assert source is None

    def test_no_labels_returns_none(self):
        row = self._row()
        label, source = effective_label(row)
        assert label is None
        assert source is None

    def test_auto_wins_over_judge(self):
        row = self._row(auto_quality="bad", judge_score=5)
        label, source = effective_label(row)
        assert label == "bad"
        assert source == "auto"

    def test_whitespace_in_quality_stripped(self):
        row = self._row(quality="  good  ")
        label, source = effective_label(row)
        assert label == "good"
        assert source == "user"

    def test_judge_score_as_string_int(self):
        """judge_score may come back as a string from some DB drivers."""
        row = self._row(judge_score="5")
        label, source = effective_label(row)
        assert label == "good"
        assert source == "judge"

    def test_invalid_user_quality_value_falls_through(self):
        """Unexpected values in quality column should fall through to auto/judge."""
        row = self._row(quality="unknown", auto_quality="good")
        label, source = effective_label(row)
        assert label == "good"
        assert source == "auto"


# ---------------------------------------------------------------------------
# Export dataset: effective_label_filter
# ---------------------------------------------------------------------------

class _FakeStore:
    """Stands in for TaskStore.list_for_export with canned rows."""

    def __init__(self, rows):
        self._rows = rows

    async def list_for_export(self, *, agents=None, quality=None, since=None, limit=100_000):
        rows = self._rows
        if quality:
            rows = [r for r in rows if r.get("quality") in quality]
        if agents:
            rows = [r for r in rows if r.get("agent") in agents]
        return rows


async def test_effective_label_filter_good(tmp_path):
    from research_agent.training.export import export_dataset

    store = _FakeStore([
        # user-labeled good
        {"id": 1, "agent": "methodology", "input": "i1", "result": "o1",
         "quality": "good", "auto_quality": None, "judge_score": None},
        # auto-labeled good (no user label)
        {"id": 2, "agent": "methodology", "input": "i2", "result": "o2",
         "quality": None, "auto_quality": "good", "judge_score": None},
        # auto-labeled bad → excluded
        {"id": 3, "agent": "methodology", "input": "i3", "result": "o3",
         "quality": None, "auto_quality": "bad", "judge_score": None},
        # judge good (no user/auto label)
        {"id": 4, "agent": "methodology", "input": "i4", "result": "o4",
         "quality": None, "auto_quality": None, "judge_score": 4},
        # no labels → excluded
        {"id": 5, "agent": "methodology", "input": "i5", "result": "o5",
         "quality": None, "auto_quality": None, "judge_score": None},
    ])
    manifest = await export_dataset(store, tmp_path, effective_label_filter="good")
    assert manifest["methodology"]["count"] == 3  # ids 1, 2, 4


async def test_effective_label_filter_bad(tmp_path):
    from research_agent.training.export import export_dataset

    store = _FakeStore([
        {"id": 1, "agent": "lit", "input": "i1", "result": "o1",
         "quality": "bad", "auto_quality": None, "judge_score": None},
        {"id": 2, "agent": "lit", "input": "i2", "result": "o2",
         "quality": None, "auto_quality": "bad", "judge_score": None},
        {"id": 3, "agent": "lit", "input": "i3", "result": "o3",
         "quality": "good", "auto_quality": None, "judge_score": None},
        {"id": 4, "agent": "lit", "input": "i4", "result": "o4",
         "quality": None, "auto_quality": None, "judge_score": 2},
    ])
    manifest = await export_dataset(store, tmp_path, effective_label_filter="bad")
    assert manifest["lit"]["count"] == 3  # ids 1, 2, 4


async def test_label_source_in_metadata(tmp_path):
    """Each example's metadata must include label and label_source."""
    from research_agent.training.export import export_dataset

    store = _FakeStore([
        {"id": 1, "agent": "methodology", "input": "q", "result": "a",
         "quality": "good", "auto_quality": None, "judge_score": None},
    ])
    await export_dataset(store, tmp_path)
    lines = (tmp_path / "methodology.jsonl").read_text().splitlines()
    ex = json.loads(lines[0])
    assert ex["metadata"]["label"] == "good"
    assert ex["metadata"]["label_source"] == "user"


# ---------------------------------------------------------------------------
# Freeze: label priority ordering
# ---------------------------------------------------------------------------

class TestLabelPriority:
    def test_user_has_highest_priority(self):
        assert _label_priority("user") > _label_priority("auto")
        assert _label_priority("auto") > _label_priority("judge")
        assert _label_priority("judge") > _label_priority(None)

    def test_unknown_source_treated_as_lowest(self):
        assert _label_priority("unknown_source") == 0
        assert _label_priority(None) == 0


# ---------------------------------------------------------------------------
# Judge JSON parsing
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_clean_json(self):
        text = '{"score": 4, "rationale": "Solid work."}'
        result = _parse_judge_response(text)
        assert result == (4, "Solid work.")

    def test_score_5(self):
        result = _parse_judge_response('{"score": 5, "rationale": "Excellent."}')
        assert result is not None
        assert result[0] == 5

    def test_score_1(self):
        result = _parse_judge_response('{"score": 1, "rationale": "Failing."}')
        assert result == (1, "Failing.")

    def test_markdown_fence_stripped(self):
        text = '```json\n{"score": 3, "rationale": "Acceptable."}\n```'
        result = _parse_judge_response(text)
        assert result is not None
        assert result[0] == 3

    def test_score_out_of_range_returns_none(self):
        assert _parse_judge_response('{"score": 6, "rationale": "x"}') is None
        assert _parse_judge_response('{"score": 0, "rationale": "x"}') is None

    def test_missing_score_returns_none(self):
        assert _parse_judge_response('{"rationale": "ok"}') is None

    def test_unparseable_text_returns_none(self):
        assert _parse_judge_response("not json at all") is None

    def test_score_as_string_int_coerced(self):
        """Model might return score as a JSON string like "4"."""
        result = _parse_judge_response('{"score": "4", "rationale": "Good."}')
        assert result is not None
        assert result[0] == 4

    def test_embedded_json_in_prose(self):
        """Model wraps JSON in prose — we extract the first {...} block."""
        text = 'Here is my verdict: {"score": 4, "rationale": "Good work."} done.'
        result = _parse_judge_response(text)
        assert result is not None
        assert result[0] == 4

    def test_rationale_stripped(self):
        result = _parse_judge_response('{"score": 3, "rationale": "  ok  "}')
        assert result is not None
        assert result[1] == "ok"


# ---------------------------------------------------------------------------
# Harness: parse compare response
# ---------------------------------------------------------------------------

class TestParseCompareResponse:
    def test_clean_verdict_better(self):
        text = '{"verdict": "better", "score": 5, "rationale": "Much better."}'
        result = _parse_compare_response(text)
        assert result == {"verdict": "better", "score": 5, "rationale": "Much better."}

    def test_valid_verdicts(self):
        for v in ("better", "tie", "worse"):
            text = json.dumps({"verdict": v, "score": 3, "rationale": "ok"})
            result = _parse_compare_response(text)
            assert result is not None
            assert result["verdict"] == v

    def test_invalid_verdict_returns_none(self):
        text = '{"verdict": "unknown", "score": 3, "rationale": "x"}'
        assert _parse_compare_response(text) is None

    def test_score_out_of_range_returns_none(self):
        assert _parse_compare_response(
            '{"verdict": "tie", "score": 6, "rationale": "x"}'
        ) is None

    def test_markdown_fence_stripped(self):
        text = "```\n{\"verdict\": \"tie\", \"score\": 3, \"rationale\": \"ok\"}\n```"
        result = _parse_compare_response(text)
        assert result is not None
        assert result["verdict"] == "tie"


# ---------------------------------------------------------------------------
# Judge run_judge (LLM mocked)
# ---------------------------------------------------------------------------

class _MockLLM:
    """Minimal mock for an async LangChain chat model."""

    def __init__(self, responses):
        self._responses = iter(responses)

    async def ainvoke(self, messages):
        content = next(self._responses)

        class _R:
            def __init__(self, c):
                self.content = c

        return _R(content)


async def test_judge_one_valid_response():
    from research_agent.evals.judge import _judge_one

    llm = _MockLLM(['{"score": 4, "rationale": "Good output."}'])
    row = {"id": 1, "agent": "methodology", "input": "design X", "result": "Here is the design."}
    result = await _judge_one(llm, row)
    assert result == (4, "Good output.")


async def test_judge_one_parse_failure_returns_none():
    from research_agent.evals.judge import _judge_one

    llm = _MockLLM(["not valid json at all"])
    row = {"id": 2, "agent": "lit", "input": "search X", "result": "found papers"}
    result = await _judge_one(llm, row)
    assert result is None


async def test_judge_one_empty_input_skips():
    from research_agent.evals.judge import _judge_one

    llm = _MockLLM([])  # should never be called
    row = {"id": 3, "agent": "lit", "input": "", "result": "some result"}
    result = await _judge_one(llm, row)
    assert result is None


async def test_judge_one_dry_run_no_llm_call():
    from research_agent.evals.judge import _judge_one

    called = []

    class _TrackLLM:
        async def ainvoke(self, messages):
            called.append(True)
            raise AssertionError("should not be called in dry-run")

    row = {"id": 4, "agent": "x", "input": "q", "result": "a"}
    result = await _judge_one(_TrackLLM(), row, dry_run=True)
    assert result is None
    assert called == []
