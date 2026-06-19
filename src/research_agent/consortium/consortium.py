"""The two-track ideation consortium.

A panel of frontier reasoning models (via OpenRouter) generates ideas through two
*isolated* tracks in a SINGLE round, then everything is scored and ranked:

- **Independent track** — each panelist works ALONE from the chair's brief and
  proposes 3 ideas (diversity: their errors stay decorrelated).
- **Debated track** — the panelists hold a separate shared conversation, blind to
  the independents (capped at 1 turn); the chair extracts the 2 strongest,
  genuinely distinct ideas that emerged (ownerless: "debated (panel)").

The two pools merge (independent ideas anonymized by self-exclusion — a model
never sees its own idea on its own ballot), and every panelist scores them 0-10
under a hard-calibration, expert-reviewer rubric by CONTINUING its own
propose-phase thread (tool-free, never the debate thread, so a panelist that
argued for a debated idea isn't scoring from a context already invested in it).
Every idea that survives the output contract is returned, ranked — no top-N cut,
no second polish/vote round.

`Consortium.ideate(...)` is this single round, run non-interactively for the
orchestrator's `brainstorm_research_ideas` tool and the background dispatcher.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-request wall-clock timeout (seconds) for every consortium model call, so a
# single stalled DeepInfra/OpenRouter request can't freeze the whole round (seen
# in a dry-run: one call hung ~12 min). A reasoning model emitting 6k tokens can
# legitimately take ~70s, so this is generous.
_REQUEST_TIMEOUT = 300.0

IDEA_MARKER = "=== IDEA ==="
# Line-start anchored (multiline): only a marker that begins its own line splits
# an idea. A quoted/inline `"=== IDEA ==="` mid-sentence (which reasoning models
# love to do when restating the output format) must NOT create a spurious split.
_IDEA_SPLIT = re.compile(r"(?im)^[ \t]*" + re.escape(IDEA_MARKER))
# A well-formed <think>...</think> reasoning block (non-greedy, any case, spans
# newlines). Reasoning models (e.g. deepseek-r1) emit these inline in message
# content; the scaffolding inside must never reach the idea/score parser.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_SCORE_LINE = re.compile(r"#?\s*(\d+)\s*[:=\-]\s*(\d+(?:\.\d+)?)")
_REASON_LINE = re.compile(r"^\s*#\s*(\d+)\s*:\s*(.+)$", re.MULTILINE)
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


def _strip_reasoning(text: str) -> str:
    """Remove inline `<think>...</think>` reasoning blocks from model output.

    Reasoning models (e.g. deepseek-r1, the consortium chair) emit their chain of
    thought as a `<think>...</think>` block inline in the message *content*, with
    the real answer after `</think>`. That scratchpad routinely restates the
    required output format — including the literal `=== IDEA ===` marker and JSON
    score examples — which then masquerades as real ideas/scores at the parse
    boundary (see council_31). We strip well-formed blocks first, then any stray
    lone `<think>`/`</think>` tag left by truncation. Text with no think tags is
    returned unchanged.

    Strip ONLY at the parse boundary (parse_ideas/parse_scores), never in
    _flatten: the dashboard trace (serialize_messages) intentionally preserves
    reasoning.
    """
    text = _THINK_BLOCK.sub("", text or "")
    return _THINK_TAG.sub("", text)


# Literal template placeholders that only ever appear in the OUTPUT-FORMAT
# scaffold the panel/chair is asked to fill in — never in a real, written idea.
_SCAFFOLD_PLACEHOLDERS = (
    "[Concise name]",
    "[1-2 sentences]",
    "[Key equations",
    "[Specific improvements",
    "[Theoretical/empirical",
)
# A generic "[ ... ]" bracketed placeholder token (no nested brackets). Two or
# more of these in one segment is a strong template-leak signal; real ideas may
# carry the odd `[1]`-style reference, so a SINGLE one is tolerated.
_BRACKET_PLACEHOLDER = re.compile(r"\[[^\[\]]+\]")


def _looks_like_idea(seg: str) -> bool:
    """Reject-on-signal guard: does this segment look like a real idea, not a
    leaked output-format scaffold or reasoning fragment?

    DEFAULT ACCEPT — only reject on a clear scaffolding signal, so short toy
    bodies ("First", "The Idea") used elsewhere still pass. The think-strip is
    the primary defense; this is belt-and-suspenders for a VISIBLE leak that
    survived it (e.g. a malformed/unclosed think block, or a template echoed
    outside any think tags).
    """
    low = seg.lower()
    # A residual think tag means reasoning leaked through unstripped.
    if "<think>" in low or "</think>" in low:
        return False
    # Any literal template placeholder is a dead giveaway of the scaffold.
    if any(ph.lower() in low for ph in _SCAFFOLD_PLACEHOLDERS):
        return False
    # Several generic "[...]" placeholders together = an unfilled template.
    if len(_BRACKET_PLACEHOLDER.findall(seg)) >= 2:
        return False
    return True


# --- pure helpers (unit-tested) ----------------------------------------------

def parse_ideas(text: str, max_n: int = 3, *, model: str = "") -> list[str]:
    """Split a panelist's output into idea blocks on the IDEA_MARKER.

    Any preamble before the first marker is dropped (models routinely prefix a
    "Sure, here are my ideas:"), so it never masquerades as idea #1. Output
    lacking the marker breaks the contract and is rejected outright (logged),
    rather than silently entering the pool as a raw, unstructured blob.
    """
    text = (text or "").strip()
    # Drop reasoning-model scratchpad first: a <think> block routinely restates
    # the format template (literal "=== IDEA ===" + placeholders), which would
    # otherwise split out as bogus ideas ahead of the real ones (council_31).
    text = _strip_reasoning(text)
    if not text or (text.startswith("[") and "could not" in text[:80]):
        return []
    if IDEA_MARKER.lower() not in text.lower():
        logger.warning(
            "%s output missing %r marker; rejecting (output-contract violation)",
            model or "panelist", IDEA_MARKER,
        )
        return []
    # split()[1:] discards the segment before the first marker (the preamble).
    parts = [p.strip() for p in _IDEA_SPLIT.split(text)[1:] if p.strip()]
    # Belt-and-suspenders: drop any segment that is clearly a leaked template /
    # reasoning fragment BEFORE the cap, so it can't consume a real idea's slot.
    parts = [p for p in parts if _looks_like_idea(p)]
    return parts[:max_n]


def parse_scores(text: str, valid_ids: set[int]) -> dict[int, tuple[float, str]]:
    """Parse a scorer's output into {idea_id: (score in 0..10, reason)}.

    Scores prefer a JSON object ``{"1": 8, ...}``, falling back to ``#N: score``
    lines when no JSON parses. Reasons come from ``#N: <text>`` lines (the
    one-/two-sentence justification the prompt asks for) and default to "" when
    absent. Only ids in `valid_ids` are kept; scores are clamped to [0, 10]. An
    empty return means a failed ballot — the caller should flag it, not treat it
    as "the model scored nothing on purpose."
    """
    text = text or ""
    # Strip reasoning-model scratchpad: a <think> block can contain example JSON
    # ({"1": 8, ...}) or score lines that would corrupt the real ballot.
    text = _strip_reasoning(text)
    scores: dict[int, float] = {}

    def _store(key, val) -> None:
        try:
            i, s = int(key), float(val)
        except (TypeError, ValueError):
            return
        if i in valid_ids:
            scores[i] = max(0.0, min(10.0, s))

    for block in _JSON_OBJ.findall(text):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                _store(k, v)
    if not scores:
        for m in _SCORE_LINE.finditer(text):
            _store(m.group(1), m.group(2))

    reasons: dict[int, str] = {}
    for m in _REASON_LINE.finditer(text):
        try:
            i = int(m.group(1))
        except (TypeError, ValueError):
            continue
        if i in valid_ids:
            reasons[i] = m.group(2).strip()

    return {i: (s, reasons.get(i, "")) for i, s in scores.items()}


def normalize_and_rank(
    pool: list[dict],
    scores_by_model: dict[str, dict[int, tuple[float, str]]],
    min_raters: int = 1,
) -> list[dict]:
    """Attach aggregate scores to each idea and return them ranked best-first.

    Each rater's scores are z-normalized (so a lenient or harsh model can't
    dominate), then averaged; ties broken by the raw mean. Ideas keep `score`
    (raw mean, 0-10, for display), `score_norm` (the ranking key), and `raters`
    (per-model {model, score, reason} — who scored what and why, for the final
    presentation). Under self-exclusion `n_scores` varies per idea (an
    independent idea is rated by panel-minus-its-author; a debated idea by
    everyone) — that's expected, z-scores are mean-centered per rater regardless
    of how many ideas that rater saw.

    Ideas scored by fewer than `min_raters` raters are EXCLUDED from the ranked
    output (default 1 drops zero-rater ideas). Shipping a "consensus" rank for an
    idea nobody scored — its `score` would silently default to 0.0 — is a
    scoring-integrity bug (see council_31), so such ideas are dropped entirely
    rather than ranked at a fictitious 0.
    """
    # Per-model mean/std for z-normalization.
    stats: dict[str, tuple[float, float]] = {}
    for model, scored in scores_by_model.items():
        vals = [s for s, _ in scored.values()]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[model] = (mean, var ** 0.5)

    ranked: list[dict] = []
    for idea in pool:
        raws, norms, raters = [], [], []
        for model, scored in scores_by_model.items():
            if idea["id"] not in scored or model not in stats:
                continue
            raw, reason = scored[idea["id"]]
            mean, std = stats[model]
            raws.append(raw)
            norms.append((raw - mean) / std if std > 0 else 0.0)
            raters.append({"model": model, "score": raw, "reason": reason})
        enriched = {
            **idea,
            "score": round(sum(raws) / len(raws), 2) if raws else 0.0,
            "score_norm": sum(norms) / len(norms) if norms else 0.0,
            "n_scores": len(raws),
            "raters": raters,
        }
        # Drop ideas nobody (or too few) scored: an unscored idea would ship at a
        # fictitious 0.0 "consensus" score (council_31). Default min_raters=1.
        if enriched["n_scores"] < min_raters:
            continue
        ranked.append(enriched)
    ranked.sort(key=lambda i: (i["score_norm"], i["score"]), reverse=True)
    return ranked


def idea_synopsis(text: str, limit: int = 360) -> str:
    """A short, single-spaced preview of an idea for the chooser list."""
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def render_transcript(transcript: list[tuple[str, str]]) -> str:
    if not transcript:
        return "(no discussion yet)"
    return "\n\n".join(f"[{speaker}]:\n{text}" for speaker, text in transcript)


def build_document(topic: str, sections: list[tuple[str, str]]) -> str:
    """Assemble the saved markdown from (heading, body) sections."""
    parts = [f"# Research ideas — {topic}", ""]
    for heading, body in sections:
        parts += [f"## {heading}", (body or "").strip(), ""]
    return "\n".join(parts)


# A paperclip result entry looks like:
#   1. Social Perception of Faces in a Vision-Language Model
#      Carina I. Hausladen, Manuel Knott, Colin F. Camerer, Pietro Perona
#      arx_2408.14435 · arXiv · 2024-08-26
#      https://doi.org/10.1145/3715275.3732041
#      "<abstract>"
_PAPER_ENTRY = re.compile(
    r"^\s*\d+\.\s+(?P<title>.+)\n"
    r"\s+(?P<authors>.+)\n"
    r"\s+(?P<id>\S+)\s*·\s*(?P<source>[^·\n]+?)\s*·\s*(?P<date>\S+)"
    r"(?:\s*\n\s*(?P<url>https?://\S+))?",
    re.MULTILINE,
)


def extract_references(trace: list[dict]) -> dict[str, dict]:
    """Pull a deduped paper corpus (id -> citation fields) out of every
    paperclip result seen anywhere in the session trace (propose + debate)."""
    refs: dict[str, dict] = {}
    for step in trace:
        if step.get("type") != "tool":
            continue
        for m in _PAPER_ENTRY.finditer(step.get("content") or ""):
            pid = m.group("id")
            if pid in refs:
                continue
            refs[pid] = {
                "title": m.group("title").strip(),
                "authors": m.group("authors").strip(),
                "source": m.group("source").strip(),
                "date": m.group("date").strip(),
                "url": (m.group("url") or "").strip(),
            }
    return refs


def format_citation(pid: str, ref: dict) -> str:
    bits = [ref["authors"], f"*{ref['title']}*", f"{ref['source']}, {ref['date']}", f"`{pid}`"]
    if ref.get("url"):
        bits.append(ref["url"])
    return ". ".join(b for b in bits if b)


def idea_bibliography(idea_text: str, refs: dict[str, dict]) -> str:
    """The formatted citations for ids actually mentioned in this idea's text."""
    cited = [pid for pid in refs if pid in idea_text]
    if not cited:
        return "(no resolvable citations)"
    return "\n".join(f"{i}. {format_citation(pid, refs[pid])}" for i, pid in enumerate(cited, 1))


def references_document(refs: dict[str, dict]) -> str:
    """The full deduped session corpus, for the standalone references file."""
    if not refs:
        return "(no papers retrieved this session)"
    return "\n".join(f"- {format_citation(pid, ref)}" for pid, ref in refs.items())


# --- prompts -----------------------------------------------------------------

# Per-phase tool-call budgets, stated explicitly so panelists spend the
# recursion limit (30, ≈15 tool-call rounds) deliberately rather than randomly
# looping until they hit it — same pattern as the literature subagent's
# 8-search cap (see agents/literature.py).
_PHASE_BUDGETS = {
    "propose": "You have a budget of roughly 8 tool calls (searches + reads) for "
                "this phase. Plan your queries up front; don't re-run a search "
                "you've already done. Stop searching once you have enough to "
                "write strong, well-grounded ideas.",
    "debate": "You have a budget of roughly 5 tool calls for this turn — search "
              "only to settle a specific contested claim, not to re-survey the "
              "field from scratch.",
}


def _panel_system(model: str, shared: bool, phase: str = "propose") -> str:
    mode = (
        "This is a SHARED debate: you see the other panelists and must engage them "
        "by name — merge, refine, or push back."
        if shared else
        "You are working ALONE; you cannot see the other panelists."
    )
    budget = _PHASE_BUDGETS.get(phase, "")
    return (
        f"You are {model}, a frontier reasoning model on a research ideation panel. "
        f"{mode}\n\n"
        "Tools available:\n"
        "• paperclip (literature search) — pass a CLI string; the -s source flag is REQUIRED.\n"
        "  Usage: search -s <source> \"query\"   where source ∈ {arxiv, pmc, biorxiv, medrxiv}\n"
        "  Examples: search -s arxiv \"few-shot image classification\"\n"
        "            search -s pmc \"medical image segmentation transformer\"\n"
        "  For biomedical/clinical work search both arxiv AND pmc.\n"
        "• Tavily (web search) — for conference CFPs, blog posts, and non-paper sources.\n"
        "SEARCH before claiming novelty or SOTA — verify citations; use real arXiv ids / DOIs.\n"
        + (f"{budget}\n" if budget else "") +
        "\nBe rigorous and concrete: give the mathematical formulation (LaTeX) and a "
        "theoretical justification or proof sketch where it matters, and for every "
        "proposed improvement state WHAT to change, WHERE, and WHY."
    )


_IDEA_FORMAT = (
    f"Start EACH idea with a line `{IDEA_MARKER}` followed by: **Title**; **Problem "
    "& motivation**; **Novelty** vs cited prior work; **Method** with key "
    "formulae/derivation; **What/Where/Why** it improves; **Expected contribution**. "
    "Cite prior work inline using the paperclip id exactly as returned by search "
    "(e.g. `arx_2408.14435`, `PMC1234567`), not just a paper title or journal name — "
    "the bibliography is built by matching these ids, so an uncited or prose-only "
    "reference (\"Nature 2024\") won't resolve into it."
)

_PROPOSE_INDEP = (
    "Working alone, first SEARCH the literature and web to map the state of the art "
    "and the concrete open gaps. Then propose EXACTLY 3 ambitious-but-feasible "
    f"research ideas, each targeting a top (Q1) venue. {_IDEA_FORMAT} Output only the "
    "3 ideas."
)

_DEBATE_OPEN = (
    "Open the debate: SEARCH, then state your strongest research direction for this "
    "brief and the key open gaps. Be specific and ground claims in citations."
)

_DEBATE_REACT = (
    "Continue the debate. Engage the others BY NAME: merge complementary directions, "
    "attack weak ones with arguments or counterexamples, and push toward a genuinely "
    "novel synthesis no single opening reached. SEARCH to settle any contested claim."
)

# Chair-side extraction: ONE call over the full transcript, not one per panelist.
# The resulting ideas are panel syntheses with no single owner (author: "panel"),
# so there's no anonymity to preserve and no self-exclusion to bookkeep for them.
_DEBATE_EXTRACT_CHAIR = (
    "The panel debate below is over. From the FULL transcript, extract EXACTLY 2 "
    "distinct ideas that emerged — the two strongest, genuinely different "
    "directions (not two phrasings of the same idea; if the debate converged on "
    "one direction, extract that one plus the strongest dissenting alternative "
    f"that was raised). {_IDEA_FORMAT} Output exactly 2 ideas, nothing else."
)

# Appended as a HUMAN turn onto each panelist's OWN independent-track thread (never
# the debate thread — a model that argued for a debated idea is invested in it, and
# scoring it from that thread would reward participation, not merit). Continuing a
# thread means the system prompt can't be swapped, so the expert-evaluator framing
# has to be fully established here rather than in a fresh system message. No tools
# are bound for this call: it's a single, additive, tool-free turn.
_SCORE = (
    "Forget the tools — for this turn you are acting purely as an independent "
    "expert reviewer in this research field, scoring from domain knowledge you "
    "already have (including whatever you found earlier in this conversation). "
    "Do NOT search or call any tool; if you're unsure, judge on the idea's stated "
    "claims and citations as written.\n\n"
    "Score EACH idea below for a top (Q1) venue, 0-10, the way a rigorous Q1 "
    "reviewer would. CALIBRATE HARD — do NOT cluster around 7. Anchor: 0-2 "
    "unacceptable/flawed; 3-4 weak; 5-6 incremental; 7 solid but NOT top-venue; "
    "8-9 genuinely Q1; 10 landmark. Use the FULL range across the pool below, and "
    "let no more than two ideas share a score.\n\n"
    "Output a JSON object mapping idea number to score, e.g. {\"1\": 8, \"2\": 3}, "
    "then one line per idea as `#N: <1-2 sentence reason>`. If you cannot score an "
    "idea at all, omit its number rather than guessing."
)

def _brief_prompt(topic: str, focus: str) -> str:
    extra = f"\nThe researcher emphasizes: {focus}" if focus else ""
    return (
        "You are the chair of a research ideation panel. Expand the researcher's "
        f"topic into ONE comprehensive brief given to every panelist: the precise "
        "problem area, what counts as a Q1 contribution here, the key sub-questions, "
        "any constraints, and what to search for. Two short paragraphs, no preamble."
        f"\n\nTopic: {topic}{extra}"
    )


class Consortium:
    def __init__(self, lit_tools, panel_models, chair_model, output_dir,
                 temperature: float = 0.6, recall=None, debate_turns: int = 2,
                 rounds: int = 1):
        # `lit_tools` is the shared MCP tool pool (paperclip + Tavily web search).
        self.tools = lit_tools or []
        self.panel = list(panel_models)
        self.chair_model = chair_model
        self.output_dir = Path(output_dir) / "ideas"
        self.temperature = temperature
        self.debate_turns = max(1, debate_turns)
        self.rounds = rounds  # kept for back-compat; unused by the new flow
        # Optional async callable (query -> str) of prior insights to seed the
        # DEBATE track (the independent track stays fresh each session).
        self.recall = recall

    def new_session(self, topic: str, focus: str = "") -> "ConsortiumSession":
        return ConsortiumSession(self, topic, focus)

    # -- low-level primitive (the single seam tests/fakes can stub) --

    async def _agent_say(self, model: str, system: str, instruction: str,
                         transcript: list[tuple[str, str]] | None = None) -> tuple[str, list]:
        """Run one tool-using agent turn; return (reply_text, message_history)."""
        from langgraph.prebuilt import ToolNode, create_react_agent

        from ..llm import build_openrouter_chat

        # handle_tool_errors moved from create_react_agent to ToolNode in LangGraph 1.x
        agent = create_react_agent(
            build_openrouter_chat(model, self.temperature, max_tokens=6000,
                                  timeout=_REQUEST_TIMEOUT),
            ToolNode(self.tools, handle_tool_errors=True), prompt=system,
        )
        if transcript:
            content = f"Debate so far:\n\n{render_transcript(transcript)}\n\n---\n{instruction}"
        else:
            content = instruction
        try:
            res = await agent.ainvoke(
                {"messages": [("user", content)]}, config={"recursion_limit": 30}
            )
            messages = res["messages"]
            return _flatten(messages[-1].content), messages
        except Exception as exc:  # noqa: BLE001 — one model failing must not abort
            logger.exception("Consortium agent %s failed", model)
            return f"[{model} could not respond: {exc}]", []

    async def _say_plain(self, model: str, messages: list) -> tuple[str, list]:
        """One tool-free turn appended to an existing message history.

        Unlike `_agent_say`, this never builds a ReAct agent or binds tools — it
        calls the chat model directly. Used for scoring (continuing a panelist's
        own propose-phase thread) and chair debate-extraction: both are single,
        additive, non-agentic turns with no search loop and so no recursion-limit
        exposure. Because the thread already carries its system message (from
        whichever `_agent_say` call started it), the framing for this turn must
        live in the new instruction itself — there's no system prompt to swap.
        """
        from ..llm import build_openrouter_chat

        llm = build_openrouter_chat(model, self.temperature, max_tokens=2000)
        try:
            reply = await llm.ainvoke(messages)
            new_messages = messages + [reply]
            return _flatten(reply.content), new_messages
        except Exception as exc:  # noqa: BLE001 — one model failing must not abort
            logger.exception("Consortium agent %s failed (plain turn)", model)
            return f"[{model} could not respond: {exc}]", messages

    # -- graph seams (stubbable; the graph nodes build their LLMs through here) --

    def _chat(self, model: str, *, max_tokens: int, with_tools: bool = False):
        """Build a chat model for a graph node; tool-bind only when asked.

        The single seam the panelist/scorer nodes construct LLMs through, so an
        offline smoke-run (or a test) can stub model behaviour by overriding this.
        """
        from ..llm import build_openrouter_chat

        llm = build_openrouter_chat(
            model, self.temperature, max_tokens=max_tokens, timeout=_REQUEST_TIMEOUT
        )
        if with_tools and self.tools:
            llm = llm.bind_tools(self.tools)
        return llm

    async def _prior(self, topic: str) -> str:
        """Recalled prior-debate insights (seeds the debate track); "" if none."""
        if self.recall is None:
            return ""
        try:
            return await self.recall(topic) or ""
        except Exception:  # noqa: BLE001
            return ""

    # -- chair --

    async def make_brief(self, topic: str, focus: str = "") -> tuple[str, list]:
        text, msgs = await self._agent_say(
            self.chair_model, "You are a precise research chair.", _brief_prompt(topic, focus)
        )
        # Strip the chair's <think> block from the brief: it's fed to every
        # panelist as their instruction and saved into the debate transcript, so
        # a reasoning-model scratchpad there wastes tokens and clutters the doc.
        # The trace (_collect) keeps the full reasoning for the dashboard.
        return _strip_reasoning(text).strip(), _collect("chair:brief", msgs)

    # -- round 1: the two idea tracks --

    async def propose_independent(
        self, brief: str
    ) -> tuple[list[dict], dict[str, list], list]:
        """Each panelist proposes 3 ideas ALONE (in parallel).

        Returns the ideas, each model's raw message history (system + human +
        tool turns), and the trace. The message histories are kept so scoring
        can later CONTINUE each model's own thread instead of starting fresh —
        a model judges with the literature context it already built, and never
        the (potentially groupthink-biased) debate transcript.
        """
        instruction = f"Research brief:\n{brief}\n\n{_PROPOSE_INDEP}"
        results = await asyncio.gather(*(
            self._agent_say(m, _panel_system(m, shared=False, phase="propose"), instruction)
            for m in self.panel
        ))
        ideas, threads, trace = [], {}, []
        for model, (text, msgs) in zip(self.panel, results, strict=True):
            trace += _collect(f"{model}:propose", msgs)
            threads[model] = msgs
            for body in parse_ideas(text, max_n=3, model=model):
                ideas.append({"text": body, "source": "independent", "by": model})
        return ideas, threads, trace

    async def debate_ideas(self, brief: str, prior: str = "") -> tuple[list[dict], list[tuple[str, str]], list]:
        """A shared debate (blind to the independents), then chair extraction.

        With `debate_turns=1` this is a single opening pass per panelist — there's
        no dedicated "engage by name" round, so what gets extracted is closer to
        "best of the opening positions" than an emergent synthesis; the chair
        prompt still asks for the two strongest *distinct* directions raised.

        Extraction is ONE chair call over the whole transcript (not one call per
        panelist): the resulting ideas are panel syntheses with no single author,
        so there's nothing to anonymize and no self-exclusion to bookkeep for
        them downstream.
        """
        transcript: list[tuple[str, str]] = [("Brief", brief)]
        if prior:
            transcript.append(("Prior debate insights (memory)", prior))
        trace: list = []
        for turn in range(self.debate_turns):
            instruction = _DEBATE_OPEN if turn == 0 else _DEBATE_REACT
            for model in self.panel:
                text, msgs = await self._agent_say(
                    model, _panel_system(model, shared=True, phase="debate"), instruction, transcript
                )
                transcript.append((model, text))
                trace += _collect(f"{model}:debate{turn + 1}", msgs)
        text, msgs = await self._agent_say(
            self.chair_model, "You are the research chair.", _DEBATE_EXTRACT_CHAIR, transcript
        )
        trace += _collect("chair:extract", msgs)
        ideas = [
            {"text": body, "source": "debated", "by": "panel"}
            for body in parse_ideas(text, max_n=2, model="chair")
        ]
        return ideas, transcript, trace

    async def score_round1(
        self, pool: list[dict], propose_threads: dict[str, list]
    ) -> tuple[dict[str, dict[int, tuple[float, str]]], list[str], list]:
        """Score the round-1 pool: threaded, self-excluding, tool-free.

        Each model scores by continuing its OWN independent/propose-phase
        thread — never the debate thread, so a panelist who argued for a
        debated idea isn't scoring from a context where it's already invested
        in that idea. The call is a single additive, tool-free turn (no
        ToolNode bound), so it carries none of the runaway-loop risk a fresh
        tool-using agent call would.

        Self-exclusion: a model's own independent idea(s) are omitted from its
        own ballot (the pool's `by` field is the only author record — there is
        no separate anonymity-bookkeeping structure to maintain). Debated ideas
        have no single author (`by == "panel"`) so every model scores them.

        A model that returns no parseable score at all is recorded as a failed
        ballot rather than silently dropped.
        """
        from langchain_core.messages import HumanMessage

        scores: dict[str, dict[int, tuple[float, str]]] = {}
        flags: list[str] = []
        trace: list = []

        async def _score_one(model: str) -> None:
            ballot = [
                idea for idea in pool
                if not (idea.get("source") == "independent" and idea.get("by") == model)
            ]
            valid = {i["id"] for i in ballot}
            listing = "\n\n".join(f"#{i['id']}:\n{i['text']}" for i in ballot)
            prompt = f"{_SCORE}\n\nIdeas:\n\n{listing}"
            thread = propose_threads.get(model) or []
            if not thread:
                flags.append(f"{model} had no propose-phase thread to score from")
                return
            text, new_thread = await self._say_plain(model, thread + [HumanMessage(content=prompt)])
            trace.extend(_collect(f"{model}:score", new_thread[len(thread):]))
            parsed = parse_scores(text, valid)
            if not parsed:
                flags.append(f"{model} failed to vote")
                return
            scores[model] = parsed

        await asyncio.gather(*(_score_one(m) for m in self.panel))
        return scores, flags, trace

    def _save(
        self, topic: str, document: str, references: str | None = None
    ) -> tuple[str, str, str | None, str | None]:
        """Write the proposal doc and, if given, a sibling references file
        sharing the same slug+timestamp (so the pairing is obvious on disk)."""
        from ..writing.latex import slugify

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = slugify(topic)
        name = f"{slug}-{stamp}.md"
        (self.output_dir / name).write_text(document)
        path, rel_path = str(self.output_dir / name), f"ideas/{name}"
        if references is None:
            return path, rel_path, None, None
        ref_name = f"{slug}-{stamp}-references.md"
        (self.output_dir / ref_name).write_text(references)
        return path, rel_path, str(self.output_dir / ref_name), f"ideas/{ref_name}"

    async def ideate(self, topic: str, focus: str = "") -> dict:
        """Non-interactive, single round: propose, debate, score. Returns
        every idea that passed the output contract, ranked — no top-N cut."""
        session = self.new_session(topic, focus)
        await session.run_round1()
        return await session.finalize()


def _collect(speaker: str, messages: list) -> list[dict]:
    """Serialize a turn's reasoning + tool calls for the dashboard trace."""
    if not messages:
        return []
    from ..agents.middleware import serialize_messages

    steps = []
    for step in serialize_messages(messages):
        if step.get("type") == "human":
            continue
        step["speaker"] = speaker
        steps.append(step)
    return steps


class ConsortiumSession:
    """A live, per-channel ideation session: a single propose+debate+score round."""

    def __init__(self, consortium: Consortium, topic: str, focus: str = ""):
        self.c = consortium
        self.topic = topic
        self.focus = focus
        self.phase = "new"          # new -> scored -> done
        self.brief = ""
        self.pool: list[dict] = []          # all ideas this round, with ids + scores
        self.ranked: list[dict] = []        # pool sorted best-first
        self.top: list[dict] = []           # every surviving idea, ranked (no cut)
        self.debate_transcripts: list[tuple[str, str]] = []  # (label, rendered)
        self.trace: list[dict] = []
        self.flags: list[str] = []          # e.g. "qwen/qwen3.7-plus failed to vote"
        self.round_no = 0
        self.finalized = False

    @property
    def panel(self) -> list[str]:
        return self.c.panel

    async def _prior(self) -> str:
        if self.c.recall is None:
            return ""
        try:
            return await self.c.recall(self.topic) or ""
        except Exception:  # noqa: BLE001
            return ""

    async def run_round1(self) -> list[dict]:
        """Run one consortium round by compiling and invoking the StateGraph.

        brief → propose (fan-out) → debate → extract → assemble → score
        (fan-out, +chair) → aggregate. The graph's final state is mapped back
        onto this session's attributes so `finalize()` (and the bot's
        interactive flow) keep their existing contract. Single round, no top-N
        cut: every idea that passed the output contract is scored and ranked.
        """
        from .graph import build_consortium_graph

        self.round_no = 1
        app = build_consortium_graph(self.c)
        final = await app.ainvoke(
            {"topic": self.topic, "focus": self.focus},
            config={"recursion_limit": 60},
        )
        self.brief = final.get("brief", "")
        self.pool = final.get("pool", [])
        self.ranked = final.get("ranked", [])
        self.top = self.ranked
        self.flags = final.get("flags", [])
        self.trace = final.get("trace", [])
        self.debate_transcripts = [
            ("Round 1 — idea debate", render_transcript(final.get("transcript", [])))
        ]
        self.phase = "scored"
        return self.top

    def render_top(self) -> str:
        """A numbered, scored synopsis of every ranked idea, with author and
        the reasons raters gave — for Discord and for the methodology handoff."""
        if not self.top:
            return "(no ideas yet)"
        lines = []
        if self.flags:
            lines.append("⚠️ " + "; ".join(self.flags))
        for idea in self.top:
            author = "debated (panel)" if idea.get("source") == "debated" else idea.get("by", "?")
            reasons = "; ".join(
                f"{r['model']}: {r['reason']}" for r in idea.get("raters", []) if r.get("reason")
            )
            line = f"**#{idea['id']} · {idea['score']:.1f}/10 · author: {author}**\n{idea_synopsis(idea['text'])}"
            if reasons:
                line += f"\n_Reasons: {reasons}_"
            lines.append(line)
        return "\n\n".join(lines)

    async def finalize(self) -> dict:
        refs = extract_references(self.trace)
        sections = []
        for i in self.top:
            author = "debated (panel)" if i.get("source") == "debated" else i.get("by", "?")
            heading = f"#{i['id']} — {i['score']:.1f}/10 — author: {author}"
            reasons = "\n".join(
                f"- {r['model']}: {r['reason']}" for r in i.get("raters", []) if r.get("reason")
            )
            body = (
                f"{i['text']}\n\n"
                f"**Reviewer notes**\n{reasons or '(no reasons recorded)'}\n\n"
                f"**Bibliography**\n{idea_bibliography(i['text'], refs)}"
            )
            sections.append((heading, body))
        if self.flags:
            sections.append(("Flags", "\n".join(f"- {f}" for f in self.flags)))
        document = build_document(self.topic, sections + self.debate_transcripts)
        path, rel_path, ref_path, rel_ref_path = self.c._save(
            self.topic, document, references_document(refs)
        )
        self.finalized = True
        self.phase = "done"
        return {
            "ideas": self.render_top(),
            "top": self.top,
            "debate_transcripts": self.debate_transcripts,
            "path": path,
            "rel_path": rel_path,
            "references_path": ref_path,
            "rel_references_path": rel_ref_path,
            "flags": self.flags,
            "n_models": len(self.panel),
            "rounds": self.round_no,
            "trace": self.trace,
        }
