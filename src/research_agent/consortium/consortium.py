"""The two-track ideation consortium.

A panel of frontier reasoning models (via OpenRouter) generates ideas through two
*isolated* tracks, then everything is scored and the strongest survive:

- **Independent track** — each panelist works ALONE from the chair's brief and
  proposes 3 ideas (diversity: their errors stay decorrelated).
- **Debated track** — the panelists hold a separate shared conversation, blind to
  the independents, and each submits its single strongest debate-born idea
  (emergent synthesis). The debate transcript is saved for future sessions.

The two pools merge (anonymized), every panelist scores all of them 0-10 under an
anti-neutrality rubric, and the chair ranks them (normalizing across raters) into
a top-5 for the researcher. The researcher picks; the panel then *polishes* the
chosen idea(s) — again on both tracks — and votes; the best are returned.

`Consortium.ideate(...)` runs round 1 only (propose + score) non-interactively for
the orchestrator's `brainstorm_research_ideas` tool and the background dispatcher.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

IDEA_MARKER = "=== IDEA ==="
_IDEA_SPLIT = re.compile(re.escape(IDEA_MARKER), re.IGNORECASE)
_SCORE_LINE = re.compile(r"#?\s*(\d+)\s*[:=\-]\s*(\d+(?:\.\d+)?)")
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


# --- pure helpers (unit-tested) ----------------------------------------------

def parse_ideas(text: str, max_n: int = 3) -> list[str]:
    """Split a panelist's output into idea blocks on the IDEA_MARKER.

    Falls back to the whole text as a single idea when the marker is absent.
    An error sentinel (``[... could not respond ...]``) yields no ideas.
    """
    text = (text or "").strip()
    if not text or (text.startswith("[") and "could not" in text[:80]):
        return []
    parts = [p.strip() for p in _IDEA_SPLIT.split(text) if p.strip()]
    return parts[:max_n]


def parse_scores(text: str, valid_ids: set[int]) -> dict[int, float]:
    """Parse a scorer's output into {idea_id: score in 0..10}.

    Prefers a JSON object ``{"1": 8, ...}``; falls back to ``#N: score`` lines.
    Only ids in `valid_ids` are kept; scores are clamped to [0, 10].
    """
    out: dict[int, float] = {}

    def _store(key, val) -> None:
        try:
            i, s = int(key), float(val)
        except (TypeError, ValueError):
            return
        if i in valid_ids:
            out[i] = max(0.0, min(10.0, s))

    for block in _JSON_OBJ.findall(text or ""):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                _store(k, v)
    if not out:
        for m in _SCORE_LINE.finditer(text or ""):
            _store(m.group(1), m.group(2))
    return out


def normalize_and_rank(pool: list[dict], scores_by_model: dict[str, dict[int, float]]) -> list[dict]:
    """Attach aggregate scores to each idea and return them ranked best-first.

    Each rater's scores are z-normalized (so a lenient or harsh model can't
    dominate), then averaged; ties broken by the raw mean. Ideas keep `score`
    (raw mean, 0-10, for display) and `score_norm` (the ranking key).
    """
    # Per-model mean/std for z-normalization.
    stats: dict[str, tuple[float, float]] = {}
    for model, scores in scores_by_model.items():
        vals = list(scores.values())
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[model] = (mean, var ** 0.5)

    ranked: list[dict] = []
    for idea in pool:
        raws, norms = [], []
        for model, scores in scores_by_model.items():
            if idea["id"] not in scores or model not in stats:
                continue
            raw = scores[idea["id"]]
            mean, std = stats[model]
            raws.append(raw)
            norms.append((raw - mean) / std if std > 0 else 0.0)
        enriched = {
            **idea,
            "score": round(sum(raws) / len(raws), 2) if raws else 0.0,
            "score_norm": sum(norms) / len(norms) if norms else 0.0,
            "n_scores": len(raws),
        }
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


# --- prompts -----------------------------------------------------------------

def _panel_system(model: str, shared: bool) -> str:
    mode = (
        "This is a SHARED debate: you see the other panelists and must engage them "
        "by name — merge, refine, or push back."
        if shared else
        "You are working ALONE; you cannot see the other panelists."
    )
    return (
        f"You are {model}, a frontier reasoning model on a research ideation panel. "
        f"{mode}\n\n"
        "Tools: scientific literature search (paperclip) and web search (Tavily). "
        "SEARCH before claiming novelty or SOTA — verify, never invent citations "
        "(use real arXiv id / DOI / URL).\n\n"
        "Be rigorous and concrete: give the mathematical formulation (LaTeX) and a "
        "theoretical justification or proof sketch where it matters, and for every "
        "proposed improvement state WHAT to change, WHERE, and WHY."
    )


_IDEA_FORMAT = (
    f"Start EACH idea with a line `{IDEA_MARKER}` followed by: **Title**; **Problem "
    "& motivation**; **Novelty** vs cited prior work; **Method** with key "
    "formulae/derivation; **What/Where/Why** it improves; **Expected contribution**."
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

_DEBATE_EXTRACT = (
    "The debate is over. Submit your SINGLE strongest idea that emerged from it "
    f"(it may be a synthesis). {_IDEA_FORMAT} Output only that one idea."
)

_SCORE = (
    "Score EACH idea below for a top (Q1) venue, 0-10. CALIBRATE HARD — do NOT "
    "cluster around 7. Anchor: 0-2 unacceptable/flawed; 3-4 weak; 5-6 incremental; "
    "7 solid but NOT top-venue; 8-9 genuinely Q1; 10 landmark. Use the FULL range — "
    "at least one idea ≤4 and at least one ≥8 — and let no more than two "
    "ideas share a score. SEARCH if you need to verify novelty. Output a JSON object "
    "mapping idea number to score, e.g. {\"1\": 8, \"2\": 3}, then one line per idea "
    "as `#N: <one-sentence justification>`."
)

_VOTE = (
    "These are polished proposals. Vote by scoring EACH 0-10 on Q1-readiness using "
    "the same hard calibration (no clustering at 7; use the full range). Output the "
    "JSON object of number->score, then `#N: <justification>` per proposal."
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


def _polish_indep_prompt(instructions: str) -> str:
    return (
        "Polish the idea below into a rigorous, submission-ready proposal: full "
        "methodology, concrete methods, and proofs/derivations for the core claims. "
        "SEARCH deeper to ground every choice and citation. Keep the original "
        f"contribution but make it defensible at a top venue.\n\n{instructions}"
    )

_POLISH_DEBATE_OPEN = (
    "Debate how to harden the idea below for a top venue: the methodology, the "
    "methods, and the proofs/derivations. Raise the hardest objections and how to "
    "answer them. SEARCH for the standard baselines, datasets, and protocols."
)

_POLISH_DEBATE_FINAL = (
    "The debate is over. Write the consolidated polished proposal the panel "
    "converged on: rigorous methodology, methods, and proofs/derivations, with "
    "cited baselines and an experiment plan. Clean Markdown; keep LaTeX for math."
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
        from langgraph.prebuilt import create_react_agent

        from ..llm import build_openrouter_chat

        agent = create_react_agent(
            build_openrouter_chat(model, self.temperature, max_tokens=6000),
            self.tools, prompt=system,
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

    # -- chair --

    async def make_brief(self, topic: str, focus: str = "") -> tuple[str, list]:
        text, msgs = await self._agent_say(
            self.chair_model, "You are a precise research chair.", _brief_prompt(topic, focus)
        )
        return text, _collect("chair:brief", msgs)

    # -- round 1: the two idea tracks --

    async def propose_independent(self, brief: str) -> tuple[list[dict], list]:
        """Each panelist proposes 3 ideas ALONE (in parallel)."""
        instruction = f"Research brief:\n{brief}\n\n{_PROPOSE_INDEP}"
        results = await asyncio.gather(*(
            self._agent_say(m, _panel_system(m, shared=False), instruction)
            for m in self.panel
        ))
        ideas, trace = [], []
        for model, (text, msgs) in zip(self.panel, results, strict=True):
            trace += _collect(f"{model}:propose", msgs)
            for body in parse_ideas(text, max_n=3):
                ideas.append({"text": body, "source": "independent", "by": model})
        return ideas, trace

    async def debate_ideas(self, brief: str, prior: str = "") -> tuple[list[dict], list[tuple[str, str]], list]:
        """A shared debate (blind to the independents); each submits its best idea."""
        transcript: list[tuple[str, str]] = [("Brief", brief)]
        if prior:
            transcript.append(("Prior debate insights (memory)", prior))
        trace: list = []
        for turn in range(self.debate_turns):
            instruction = _DEBATE_OPEN if turn == 0 else _DEBATE_REACT
            for model in self.panel:
                text, msgs = await self._agent_say(
                    model, _panel_system(model, shared=True), instruction, transcript
                )
                transcript.append((model, text))
                trace += _collect(f"{model}:debate{turn + 1}", msgs)
        # Extraction: each panelist's single strongest debate-born idea.
        ideas: list[dict] = []
        extracts = await asyncio.gather(*(
            self._agent_say(m, _panel_system(m, shared=True), _DEBATE_EXTRACT, transcript)
            for m in self.panel
        ))
        for model, (text, msgs) in zip(self.panel, extracts, strict=True):
            trace += _collect(f"{model}:extract", msgs)
            picked = parse_ideas(text, max_n=1)
            if picked:
                ideas.append({"text": picked[0], "source": "debated", "by": model})
        return ideas, transcript, trace

    async def score_pool(self, pool: list[dict], instruction: str = _SCORE) -> tuple[dict[str, dict[int, float]], list]:
        """Every panelist scores every idea (in parallel, anonymized)."""
        valid = {i["id"] for i in pool}
        listing = "\n\n".join(f"#{i['id']}:\n{i['text']}" for i in pool)
        prompt = f"{instruction}\n\nIdeas:\n\n{listing}"
        results = await asyncio.gather(*(
            self._agent_say(m, _panel_system(m, shared=False), prompt) for m in self.panel
        ))
        scores: dict[str, dict[int, float]] = {}
        trace: list = []
        for model, (text, msgs) in zip(self.panel, results, strict=True):
            trace += _collect(f"{model}:score", msgs)
            parsed = parse_scores(text, valid)
            if parsed:
                scores[model] = parsed
        return scores, trace

    async def vote_pool(self, pool: list[dict]) -> tuple[dict[str, dict[int, float]], list]:
        """Panelists vote on polished proposals (re-score with the vote rubric)."""
        return await self.score_pool(pool, instruction=_VOTE)

    # -- round 2: polish (both tracks) --

    async def polish_idea(self, idea_text: str, instructions: str,
                          prior: str = "") -> tuple[list[dict], dict | None, list[tuple[str, str]], list]:
        instr = instructions or "Develop this idea into its strongest form."
        block = f"{instr}\n\nIdea:\n{idea_text}"
        # Independent polish (each panelist alone, in parallel).
        indep = await asyncio.gather(*(
            self._agent_say(m, _panel_system(m, shared=False), _polish_indep_prompt(block))
            for m in self.panel
        ))
        proposals, trace = [], []
        for model, (text, msgs) in zip(self.panel, indep, strict=True):
            trace += _collect(f"{model}:polish", msgs)
            if text and not text.startswith("["):
                proposals.append({"text": text, "source": "independent", "by": model})
        # Debated polish (shared conversation, blind to the independent polishes).
        transcript: list[tuple[str, str]] = [("Idea to harden", idea_text), ("Chair", instr)]
        if prior:
            transcript.append(("Prior debate insights (memory)", prior))
        for model in self.panel:
            text, msgs = await self._agent_say(
                model, _panel_system(model, shared=True), _POLISH_DEBATE_OPEN, transcript
            )
            transcript.append((model, text))
            trace += _collect(f"{model}:polish-debate", msgs)
        final, msgs = await self._agent_say(
            self.chair_model, "You are the research chair.", _POLISH_DEBATE_FINAL, transcript
        )
        trace += _collect("chair:polish-final", msgs)
        debated = {"text": final, "source": "debated", "by": "panel"} if final and not final.startswith("[") else None
        return proposals, debated, transcript, trace

    def _save(self, topic: str, document: str) -> tuple[str, str]:
        from ..writing.latex import slugify

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{slugify(topic)}-{stamp}.md"
        (self.output_dir / name).write_text(document)
        return str(self.output_dir / name), f"ideas/{name}"

    async def ideate(self, topic: str, focus: str = "") -> dict:
        """Non-interactive: round 1 only (propose + score), return the top 5."""
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
    """A live, per-channel ideation session: propose+score -> select -> polish+vote."""

    def __init__(self, consortium: Consortium, topic: str, focus: str = ""):
        self.c = consortium
        self.topic = topic
        self.focus = focus
        self.phase = "new"          # new -> scored -> polished -> done
        self.brief = ""
        self.pool: list[dict] = []          # all ideas this round, with ids + scores
        self.ranked: list[dict] = []        # pool sorted best-first
        self.top: list[dict] = []           # what was shown to the researcher
        self.selected: list[dict] = []      # round-1 ideas the researcher chose
        self.debate_transcripts: list[tuple[str, str]] = []  # (label, rendered)
        self.trace: list[dict] = []
        self.round_no = 0
        self.busy = False
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
        """Propose (independent + debated), score the merged pool, return top 5."""
        self.round_no = 1
        self.brief, brief_trace = await self.c.make_brief(self.topic, self.focus)
        self.trace += brief_trace
        prior = await self._prior()

        independent, t1 = await self.c.propose_independent(self.brief)
        debated, debate_tx, t2 = await self.c.debate_ideas(self.brief, prior)
        self.debate_transcripts.append(("Round 1 — idea debate", render_transcript(debate_tx)))
        self.trace += t1 + t2

        self.pool = [{**idea, "id": n} for n, idea in enumerate(independent + debated, 1)]
        scores, t3 = await self.c.score_pool(self.pool)
        self.trace += t3
        self.ranked = normalize_and_rank(self.pool, scores)
        self.top = self.ranked[:5]
        self.phase = "scored"
        return self.top

    async def select_and_polish(self, picks: list[int] | None = None, comments: str = "") -> list[dict]:
        """Polish the chosen ideas (both tracks) and vote; return top 5 (or all).

        `picks` selects from the round-1 top (and is remembered); pass None on a
        follow-up round to re-polish the same selection with new `comments`.
        """
        if picks is not None:
            matched = [i for i in self.top if i["id"] in set(picks)]
            if not matched:
                logger.warning("No ideas matched picks %s; using the top idea.", picks)
            self.selected = matched or self.top[:1]
        chosen = self.selected or self.top[:1]
        self.round_no += 1
        prior = await self._prior()
        instructions = (
            f"Researcher's notes for this round: {comments}" if comments else
            "Develop this into its strongest, most defensible form."
        )

        polished: list[dict] = []
        for idea in chosen:
            indep, debated, tx, tr = await self.c.polish_idea(idea["text"], instructions, prior)
            self.debate_transcripts.append(
                (f"Round {self.round_no} — polish debate (idea #{idea['id']})", render_transcript(tx))
            )
            self.trace += tr
            polished += indep
            if debated is not None:
                polished.append(debated)

        self.pool = [{**p, "id": n} for n, p in enumerate(polished, 1)]
        votes, tv = await self.c.vote_pool(self.pool)
        self.trace += tv
        self.ranked = normalize_and_rank(self.pool, votes)
        self.top = self.ranked if len(self.ranked) <= 5 else self.ranked[:5]
        self.phase = "polished"
        return self.top

    def render_top(self) -> str:
        """A numbered, scored synopsis of the current top ideas for Discord."""
        if not self.top:
            return "(no ideas yet)"
        lines = []
        for idea in self.top:
            tag = "🤝 debated" if idea.get("source") == "debated" else "🧠 independent"
            lines.append(
                f"**#{idea['id']} · {idea['score']:.1f}/10 · {tag}**\n{idea_synopsis(idea['text'])}"
            )
        return "\n\n".join(lines)

    async def finalize(self) -> dict:
        document = build_document(
            self.topic,
            [(f"#{i['id']} ({i['score']:.1f}/10, {i['source']})", i["text"]) for i in self.top]
            + self.debate_transcripts,
        )
        path, rel_path = self.c._save(self.topic, document)
        self.finalized = True
        self.phase = "done"
        return {
            "ideas": self.render_top(),
            "top": self.top,
            "debate_transcripts": self.debate_transcripts,
            "path": path,
            "rel_path": rel_path,
            "n_models": len(self.panel),
            "rounds": self.round_no,
            "trace": self.trace,
        }
