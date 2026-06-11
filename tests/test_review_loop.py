"""Tests for the shared draft → critique → revise loop and its integration
with the dispatcher runners.

Covers:
  - review_loop: invalid→revise→valid path
  - review_loop: round cap respected (does not revise more than allowed)
  - review_loop: valid on first try → no revision
  - citation_critique: triggers exactly one revision when missing_citations non-empty
  - critique trace entries are persisted correctly in trace list
  - verifier error → draft accepted (no exception propagated)
  - methodology runner: invalid→revise→valid records two validator task rows and
    emits "✓ validated after 1 revision" in the summary
  - methodology runner: validator error leaves draft accepted, no crash
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from research_agent.agents.review_loop import citation_critique, run_review_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_draft(latex: str, missing: list[str] | None = None):
    """Return a minimal writer result dict."""
    return {
        "latex": latex,
        "tex_path": "/tmp/out.tex",
        "bib_path": "/tmp/out.bib",
        "n_refs": 1,
        "missing_citations": missing or [],
        "trace": [{"type": "ai", "content": latex}],
    }


# ---------------------------------------------------------------------------
# review_loop unit tests
# ---------------------------------------------------------------------------

async def test_valid_on_first_try_no_revision():
    """When critique says valid on round 1, run_review_loop returns after one draft."""
    draft_calls = []

    async def _draft(*, task: str) -> dict:
        draft_calls.append(task)
        return _make_draft("\\section{M}")

    async def _critique(orig_task, draft):
        return True, ""

    trace = []
    result = await run_review_loop(
        original_task="build a methodology",
        draft_fn=_draft,
        critique_fn=_critique,
        trace=trace,
        rounds=2,
    )
    assert len(draft_calls) == 1
    assert result["latex"] == "\\section{M}"
    assert len(trace) == 1
    assert trace[0]["verdict"] == "valid"
    assert trace[0]["round"] == 1
    assert trace[0]["superseded_draft"] is None


async def test_invalid_then_valid_revision():
    """invalid on round 1, valid on round 2 — one revision, correct trace."""
    drafts = [
        _make_draft("\\section{Bad}"),
        _make_draft("\\section{Fixed}"),
    ]
    draft_iter = iter(drafts)
    draft_tasks: list[str] = []

    async def _draft(*, task: str) -> dict:
        draft_tasks.append(task)
        return next(draft_iter)

    call_count = 0

    async def _critique(orig_task, draft):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False, "- Missing ablations"
        return True, ""

    trace = []
    result = await run_review_loop(
        original_task="idea",
        draft_fn=_draft,
        critique_fn=_critique,
        trace=trace,
        rounds=2,
    )
    assert result["latex"] == "\\section{Fixed}"
    assert len(draft_tasks) == 2
    # revision task must contain the REVISION REQUEST prefix and original task
    assert "REVISION REQUEST" in draft_tasks[1]
    assert "Missing ablations" in draft_tasks[1]
    assert "idea" in draft_tasks[1]
    # trace: one invalid entry (with superseded_draft) then one valid entry
    assert len(trace) == 2
    assert trace[0]["verdict"] == "invalid"
    assert trace[0]["superseded_draft"] == "\\section{Bad}"
    assert trace[1]["verdict"] == "valid"
    assert trace[1]["superseded_draft"] is None


async def test_round_cap_respected():
    """With rounds=1, no revision happens even when critique says invalid."""
    draft_calls = []

    async def _draft(*, task: str) -> dict:
        draft_calls.append(task)
        return _make_draft("\\section{Original}")

    async def _critique(orig_task, draft):
        return False, "- Issue"

    trace = []
    result = await run_review_loop(
        original_task="idea",
        draft_fn=_draft,
        critique_fn=_critique,
        trace=trace,
        rounds=1,
    )
    # Only the original draft call.
    assert len(draft_calls) == 1
    assert result["latex"] == "\\section{Original}"
    assert trace[0]["verdict"] == "invalid"


async def test_round_cap_with_two_rounds_gives_at_most_one_revision():
    """With rounds=2, at most one revision attempt is made."""
    draft_calls = []

    async def _draft(*, task: str) -> dict:
        draft_calls.append(task)
        return _make_draft(f"draft{len(draft_calls)}")

    async def _critique(orig_task, draft):
        return False, "- Still bad"  # always invalid

    trace = []
    await run_review_loop(
        original_task="x",
        draft_fn=_draft,
        critique_fn=_critique,
        trace=trace,
        rounds=2,
    )
    # round 1: original draft, round 2: one revision attempt
    assert len(draft_calls) == 2
    assert len(trace) == 2
    assert all(e["verdict"] == "invalid" for e in trace)


async def test_verifier_error_accepts_current_draft():
    """If critique raises, run_review_loop logs the error and returns the draft."""
    draft_calls = []

    async def _draft(*, task: str) -> dict:
        draft_calls.append(task)
        return _make_draft("\\section{Draft}")

    async def _critique(orig_task, draft):
        raise RuntimeError("network failure")

    trace = []
    result = await run_review_loop(
        original_task="idea",
        draft_fn=_draft,
        critique_fn=_critique,
        trace=trace,
        rounds=2,
    )
    assert result["latex"] == "\\section{Draft}"
    assert len(draft_calls) == 1  # no revision attempted
    assert trace[0]["verdict"] == "error"


async def test_revision_draft_error_keeps_previous_draft():
    """If the revision call raises, we keep the previously-accepted draft."""
    iteration = [0]

    async def _draft(*, task: str) -> dict:
        iteration[0] += 1
        if iteration[0] == 1:
            return _make_draft("\\section{Original}")
        raise RuntimeError("writer crashed")

    async def _critique(orig_task, draft):
        return False, "- Issues"

    trace = []
    result = await run_review_loop(
        original_task="idea",
        draft_fn=_draft,
        critique_fn=_critique,
        trace=trace,
        rounds=2,
    )
    assert result["latex"] == "\\section{Original}"


# ---------------------------------------------------------------------------
# citation_critique
# ---------------------------------------------------------------------------

async def test_citation_critique_valid_when_no_missing():
    valid, feedback = await citation_critique("task", _make_draft("x", missing=[]))
    assert valid is True
    assert feedback == ""


async def test_citation_critique_invalid_with_feedback():
    valid, feedback = await citation_critique("task", _make_draft("x", missing=["smith2020", "jones2021"]))
    assert valid is False
    assert "smith2020" in feedback
    assert "jones2021" in feedback
    assert "citation_check" == citation_critique.__verifier_name__


async def test_citation_critique_triggers_one_revision():
    """review_loop with citation_critique and one missing key triggers one revision."""
    # Round 1 draft has a missing citation; round 2 draft has none.
    drafts = [
        _make_draft("\\section{A}~\\cite{ghost}", missing=["ghost"]),
        _make_draft("\\section{A}", missing=[]),
    ]
    it = iter(drafts)
    calls: list[str] = []

    async def _draft(*, task: str) -> dict:
        calls.append(task)
        return next(it)

    trace = []
    result = await run_review_loop(
        original_task="topic",
        draft_fn=_draft,
        critique_fn=citation_critique,  # pass directly so __verifier_name__ is visible
        trace=trace,
        rounds=2,
    )
    assert len(calls) == 2
    assert result["missing_citations"] == []
    assert trace[0]["verifier"] == "citation_check"
    assert trace[0]["verdict"] == "invalid"
    assert trace[1]["verdict"] == "valid"


# ---------------------------------------------------------------------------
# Dispatcher integration: methodology runner
# ---------------------------------------------------------------------------

class _FakeTaskStore:
    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next = 1

    async def create(self, agent, input, channel_id=None, parent_id=None):
        tid = self._next
        self._next += 1
        self.rows[tid] = {
            "id": tid, "agent": agent, "input": input, "status": "pending",
            "result": None, "trace": [], "error": None,
        }
        return tid

    async def mark_running(self, tid):
        if tid is not None:
            self.rows[tid]["status"] = "running"

    async def finish(self, tid, result, trace):
        if tid is not None:
            self.rows[tid].update(status="done", result=result, trace=trace)

    async def fail(self, tid, error, trace):
        if tid is not None:
            self.rows[tid].update(status="failed", error=error, trace=trace)


class _FakeWriter:
    def __init__(self, tmp_path: Path, latex_seq=None):
        self._tmp = tmp_path
        self._latex_seq = list(latex_seq or ["\\section{M}"])
        self._call = 0
        self.tasks_received: list[str] = []

    async def draft(self, task, dirpath=None, lessons="", **_kwargs):
        self.tasks_received.append(task)
        latex = self._latex_seq[min(self._call, len(self._latex_seq) - 1)]
        self._call += 1
        directory = Path(dirpath) if dirpath else self._tmp
        directory.mkdir(parents=True, exist_ok=True)
        tex = directory / "out.tex"
        tex.write_text(latex)
        return {
            "tex_path": str(tex),
            "bib_path": "",
            "n_refs": 1,
            "latex": latex,
            "missing_citations": [],
            "trace": [],
        }


class _Writers:
    def __init__(self, tmp_path: Path, methodologist_latexes=None, paper_latexes=None):
        self.reviewer = _FakeWriter(tmp_path / "lit")
        self.methodologist = _FakeWriter(tmp_path / "meth", methodologist_latexes)
        self.paper_writer = _FakeWriter(tmp_path / "paper", paper_latexes)


async def test_methodology_runner_invalid_revise_valid(tmp_path, monkeypatch):
    """validate_methodology returns INVALID on round 1, VALID on round 2;
    summary should say '✓ validated after 1 revision' and two validator task rows
    should appear in the task store.
    """
    import research_agent.agents.methodology_validator as mv_mod
    import research_agent.config as config_mod

    from research_agent.agents.dispatcher import build_runners
    from research_agent.projects.store import ProjectStore

    val_calls = [0]

    async def _fake_validate(model, tools, orig_task, text, memory=None):
        val_calls[0] += 1
        if val_calls[0] == 1:
            return False, "- Missing ablations"
        return True, ""

    monkeypatch.setattr(config_mod.settings, "validation_rounds", 2)
    monkeypatch.setattr(mv_mod, "validate_methodology", _fake_validate)

    ts = _FakeTaskStore()
    writers = _Writers(tmp_path)
    projects = ProjectStore(pool=None, output_dir=str(tmp_path))
    runners = build_runners(
        model=None,
        mcp_tools=["stub"],  # truthy → enables validator
        writers=writers,
        consortium=None,
        projects=projects,
        task_store=ts,
    )
    summary, trace = await runners["methodology"]("design ablation study", "chan-1")

    # Summary should mention revision.
    assert "✓ validated after 1 revision" in summary

    # Two methodology_validator task rows were created.
    val_rows = [r for r in ts.rows.values() if r["agent"] == "methodology_validator"]
    assert len(val_rows) == 2, f"expected 2 validator rows, got {len(val_rows)}: {ts.rows}"

    # The first validator row should be done (not failed).
    assert val_rows[0]["status"] == "done"
    assert val_rows[1]["status"] == "done"
    assert val_rows[1]["result"].startswith("VALID")

    # Methodology writer was called twice (original + revision).
    assert writers.methodologist._call == 2

    # Critique trace entries should be in the full trace.
    critique_entries = [e for e in trace if e.get("type") == "critique"
                        and e.get("verifier") == "methodology_validator"]
    assert len(critique_entries) == 2
    assert critique_entries[0]["verdict"] == "invalid"
    assert critique_entries[1]["verdict"] == "valid"


async def test_methodology_runner_validator_error_accepts_draft(tmp_path, monkeypatch):
    """If validate_methodology raises, the job still completes (no crash)."""
    import research_agent.agents.methodology_validator as mv_mod
    import research_agent.config as config_mod

    from research_agent.agents.dispatcher import build_runners
    from research_agent.projects.store import ProjectStore

    async def _bad_validate(*args, **kwargs):
        raise RuntimeError("LLM unreachable")

    monkeypatch.setattr(config_mod.settings, "validation_rounds", 2)
    monkeypatch.setattr(mv_mod, "validate_methodology", _bad_validate)

    ts = _FakeTaskStore()
    writers = _Writers(tmp_path)
    projects = ProjectStore(pool=None, output_dir=str(tmp_path))
    runners = build_runners(
        model=None,
        mcp_tools=["stub"],
        writers=writers,
        consortium=None,
        projects=projects,
        task_store=ts,
    )
    # Should not raise.
    summary, trace = await runners["methodology"]("design study", "chan-2")
    assert "methodology" in summary.lower()

    # Validator row should be marked failed.
    val_rows = [r for r in ts.rows.values() if r["agent"] == "methodology_validator"]
    assert len(val_rows) == 1
    assert val_rows[0]["status"] == "failed"


async def test_methodology_runner_valid_no_revision(tmp_path, monkeypatch):
    """When validator returns VALID on first pass, summary says '✓ validated' and
    only one validator task row is created.
    """
    import research_agent.agents.methodology_validator as mv_mod
    import research_agent.config as config_mod

    from research_agent.agents.dispatcher import build_runners
    from research_agent.projects.store import ProjectStore

    async def _valid(*args, **kwargs):
        return True, ""

    monkeypatch.setattr(config_mod.settings, "validation_rounds", 2)
    monkeypatch.setattr(mv_mod, "validate_methodology", _valid)

    ts = _FakeTaskStore()
    writers = _Writers(tmp_path)
    projects = ProjectStore(pool=None, output_dir=str(tmp_path))
    runners = build_runners(
        model=None, mcp_tools=["stub"], writers=writers,
        consortium=None, projects=projects, task_store=ts,
    )
    summary, trace = await runners["methodology"]("my idea", "chan-3")
    assert "✓ validated" in summary
    assert "revision" not in summary

    val_rows = [r for r in ts.rows.values() if r["agent"] == "methodology_validator"]
    assert len(val_rows) == 1
    assert writers.methodologist._call == 1


async def test_citation_critique_in_artifact_runner(tmp_path, monkeypatch):
    """_artifact_runner applies citation critique; a draft with missing citations
    causes exactly one revision and the critique entry appears in the trace.
    """
    import research_agent.config as config_mod

    from research_agent.agents.dispatcher import build_runners
    from research_agent.projects.store import ProjectStore

    monkeypatch.setattr(config_mod.settings, "validation_rounds", 2)

    # First draft has a missing citation; second is clean.
    class _CitingWriter(_FakeWriter):
        def __init__(self, tmp_path):
            super().__init__(tmp_path)
            self._missing_seq = [["ghost"], []]

        async def draft(self, task, dirpath=None, lessons="", **_kwargs):
            self.tasks_received.append(task)
            idx = min(self._call, len(self._missing_seq) - 1)
            missing = self._missing_seq[idx]
            latex = "\\section{A}~\\cite{ghost}" if missing else "\\section{A}"
            self._call += 1
            directory = Path(dirpath) if dirpath else self._tmp
            directory.mkdir(parents=True, exist_ok=True)
            tex = directory / "out.tex"
            tex.write_text(latex)
            return {
                "tex_path": str(tex),
                "bib_path": "",
                "n_refs": 1,
                "latex": latex,
                "missing_citations": missing,
                "trace": [],
            }

    class _WritersWithCiting:
        def __init__(self):
            self.reviewer = _CitingWriter(tmp_path / "lit")
            self.methodologist = _FakeWriter(tmp_path / "meth")
            self.paper_writer = _FakeWriter(tmp_path / "paper")

    writers = _WritersWithCiting()
    projects = ProjectStore(pool=None, output_dir=str(tmp_path))
    runners = build_runners(
        model=None, mcp_tools=[], writers=writers,
        consortium=None, projects=projects,
    )

    summary, trace = await runners["literature_review"]("topic", "chan-4")
    # Two draft calls: original + revision.
    assert writers.reviewer._call == 2
    # Revision task contains the REVISION REQUEST prefix.
    assert "REVISION REQUEST" in writers.reviewer.tasks_received[1]

    # citation_check critique entry should be in trace.
    crit = [e for e in trace if e.get("type") == "critique" and e.get("verifier") == "citation_check"]
    assert len(crit) == 2  # round 1 invalid, round 2 valid
    assert crit[0]["verdict"] == "invalid"
    assert crit[1]["verdict"] == "valid"


async def test_critique_trace_entries_shape():
    """Every trace entry has the required keys for training data."""
    iteration = [0]

    async def _draft(*, task: str) -> dict:
        iteration[0] += 1
        latex = f"version{iteration[0]}"
        return _make_draft(latex, missing=["x"] if iteration[0] == 1 else [])

    trace = []
    await run_review_loop(
        original_task="topic",
        draft_fn=_draft,
        critique_fn=citation_critique,
        trace=trace,
        rounds=2,
    )
    for entry in trace:
        assert "type" in entry and entry["type"] == "critique"
        assert "round" in entry
        assert "verifier" in entry
        assert "verdict" in entry
        assert "feedback" in entry
        assert "superseded_draft" in entry
