"""The shared-session ideation consortium.

A panel of frontier models (via OpenRouter) debates in ONE shared session. Each
panelist is a tool-using agent — it searches the scientific literature (paperclip)
and the web (Tavily) before speaking, reads the running transcript, and engages
the others by name. The session is interactive: the researcher weighs in between
rounds (`ConsortiumSession.run_round(feedback=...)`), and a chair synthesis turns
the debate into a rigorous, validated research proposal (problem, novelty,
theoretical justification + formulae, what/where/why to improve, risks, venue).

`Consortium.ideate(...)` runs the same machinery non-interactively (round 1 +
N debate rounds + synthesis) for the orchestrator's `brainstorm_research_ideas`
tool.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


def render_transcript(transcript: list[tuple[str, str]]) -> str:
    """Render the shared discussion as a labeled dialogue."""
    if not transcript:
        return "(no discussion yet)"
    return "\n\n".join(f"[{speaker}]:\n{text}" for speaker, text in transcript)


def build_document(topic: str, final: str, transcript: list[tuple[str, str]]) -> str:
    """Assemble the saved markdown: final proposal + full discussion appendix."""
    parts = [
        f"# Research ideas — {topic}",
        "",
        "## Validated proposal (chair synthesis)",
        final.strip(),
        "",
        "---",
        "## Full panel discussion (shared session)",
        render_transcript(transcript),
    ]
    return "\n".join(parts)


def _panel_system(model: str) -> str:
    return (
        f"You are {model}, one of several frontier AI models on a research ideation "
        "panel. This is ONE shared session: you see everything other panelists and "
        "the researcher have said, and must engage directly — build on, refine, or "
        "push back against their points by name.\n\n"
        "You have tools: a scientific literature search (paperclip) and web search "
        "(Tavily). SEARCH before claiming novelty or SOTA — verify, don't guess; "
        "cite real sources (arXiv id / DOI / URL) and never invent them.\n\n"
        "Be rigorous and concrete. Where it matters, give the mathematical "
        "formulation (LaTeX) and a theoretical justification or sketch of proof. For "
        "every proposed improvement, state WHAT to change, WHERE (which component or "
        "step), and WHY (the mechanism and the expected effect)."
    )


_PROPOSE = (
    "Round 1 — opening proposals. First SEARCH the literature and the web to map the "
    "state of the art and the concrete open gaps. Then propose 2-3 ambitious-but-"
    "feasible research ideas, each targeting a top (Q1) venue. For each idea give: "
    "**Title**; **Problem & motivation**; **Novelty** vs prior work (cite what "
    "exists); **Method** with the key formulae/derivation; **What/Where/Why** it "
    "improves; **Expected contribution**. If earlier panelists already proposed "
    "ideas, react to theirs and differentiate yours."
)

_DEBATE = (
    "Debate round — critique and converge. SEARCH to verify any contested claim. "
    "Engage the other panelists' ideas by name: challenge novelty, feasibility, and "
    "impact with theoretical arguments or counterexamples (include formulae where "
    "useful), merge or improve where it helps, and argue toward the single strongest "
    "candidate. Explicitly flag (with a citation) anything that already exists."
)

_FEEDBACK_NOTE = (
    "\n\nThe researcher just added feedback above. Address it DIRECTLY: incorporate "
    "it, or push back with reasons, and refine the leading idea accordingly."
)


def _chair_prompt(topic: str) -> str:
    return (
        "You are the chair. Using the full panel discussion and the researcher's "
        f"feedback above, write the VALIDATED research proposal on '{topic}' the "
        "panel converged on — the single strongest idea, plus 1-2 brief alternatives. "
        "For the main idea provide: **Title**, **Problem & motivation**, **Novelty** "
        "(explicitly contrasted with cited prior work), **Theoretical justification** "
        "(formal statement + formulae/proof sketch), **Method**, **What/Where/Why** "
        "each improvement helps, **Experiments to validate it**, **Key risks**, and "
        "**Suggested venue(s)**. Be concrete and honest about novelty risk. Clean "
        "Markdown; keep LaTeX for math."
    )


class Consortium:
    def __init__(self, lit_tools, panel_models, chair_model, output_dir,
                 temperature: float = 0.6, rounds: int = 1, recall=None):
        # `lit_tools` is the shared MCP tool pool (paperclip + Tavily web search).
        self.tools = lit_tools or []
        self.panel = list(panel_models)
        self.chair_model = chair_model
        self.output_dir = Path(output_dir) / "ideas"
        self.temperature = temperature
        self.rounds = max(0, rounds)
        # Optional async callable (query -> str) returning prior insights/lessons
        # to seed the debate, so the panel builds on past sessions.
        self.recall = recall

    def new_session(self, topic: str, focus: str = "") -> "ConsortiumSession":
        return ConsortiumSession(self, topic, focus)

    async def _agent_say(self, model: str, instruction: str,
                        transcript: list[tuple[str, str]]) -> tuple[str, list]:
        """Run a tool-using agent (panelist or chair) over the shared transcript.

        Returns (reply_text, full_message_history). The history carries each
        panelist's reasoning and literature/web tool calls, which the session
        accumulates into the dashboard trace.
        """
        from langgraph.prebuilt import create_react_agent

        from ..llm import build_openrouter_chat

        agent = create_react_agent(
            build_openrouter_chat(model, self.temperature, max_tokens=6000),
            self.tools, prompt=_panel_system(model),
        )
        content = (
            f"Panel discussion so far:\n\n{render_transcript(transcript)}\n\n"
            f"---\n{instruction}"
        )
        try:
            res = await agent.ainvoke(
                {"messages": [("user", content)]}, config={"recursion_limit": 30}
            )
            messages = res["messages"]
            return _flatten(messages[-1].content), messages
        except Exception as exc:  # noqa: BLE001 — one model failing must not abort
            logger.exception("Consortium agent %s failed", model)
            return f"[{model} could not respond: {exc}]", []

    async def _synthesize(self, topic: str,
                         transcript: list[tuple[str, str]]) -> tuple[str, list]:
        return await self._agent_say(self.chair_model, _chair_prompt(topic), transcript)

    def _save(self, topic: str, document: str) -> tuple[str, str]:
        from ..writing.latex import slugify

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{slugify(topic)}-{stamp}.md"
        path = self.output_dir / name
        path.write_text(document)
        return str(path), f"ideas/{name}"

    async def ideate(self, topic: str, focus: str = "") -> dict:
        """Non-interactive run (round 1 + N debate rounds + synthesis) for the tool."""
        session = self.new_session(topic, focus)
        await session.run_round()  # round 1: propose
        for _ in range(self.rounds):
            await session.run_round()  # debate
        return await session.finalize()


class ConsortiumSession:
    """A live, per-channel ideation session the researcher steers between rounds."""

    def __init__(self, consortium: Consortium, topic: str, focus: str = ""):
        self.c = consortium
        self.topic = topic
        self.focus = focus
        self.transcript: list[tuple[str, str]] = []
        # Full reasoning + tool-call trace across all turns (for the dashboard).
        self.trace: list[dict] = []
        self.round_no = 0
        self.busy = False
        self.finalized = False

    @staticmethod
    def _collect(speaker: str, round_no: int, messages: list) -> list[dict]:
        """Serialize a turn's reasoning + tool calls, tagged with who/which round.

        Drops the leading prompt (the transcript-so-far echo) — that text is
        already preserved verbatim in `self.transcript`, so keeping it here would
        just balloon the trace with duplicated context on every turn.
        """
        if not messages:
            return []
        from ..agents.middleware import serialize_messages

        steps = []
        for step in serialize_messages(messages):
            if step.get("type") == "human":
                continue
            step["speaker"] = speaker
            step["round"] = round_no
            steps.append(step)
        return steps

    @property
    def panel(self) -> list[str]:
        return self.c.panel

    async def run_round(self, feedback: str = "") -> str:
        """Run one debate round; return a digest of this round's responses."""
        self.round_no += 1
        if self.round_no == 1:
            if self.focus:
                self.transcript.append(("Focus", self.focus))
            if self.c.recall is not None:
                try:
                    prior = await self.c.recall(self.topic)
                except Exception:  # noqa: BLE001
                    prior = ""
                if prior:
                    self.transcript.append(("Prior insights (memory)", prior))
        if feedback:
            self.transcript.append(("Researcher (you)", feedback))

        instruction = _PROPOSE if self.round_no == 1 else _DEBATE
        if feedback:
            instruction += _FEEDBACK_NOTE

        for model in self.panel:
            text, msgs = await self.c._agent_say(model, instruction, self.transcript)
            self.transcript.append((model, text))
            self.trace.extend(self._collect(model, self.round_no, msgs))
        return self._round_digest()

    def _round_digest(self) -> str:
        last = self.transcript[-len(self.panel):] if self.panel else []
        return "\n\n".join(f"**{model}**\n{text}" for model, text in last)

    async def finalize(self) -> dict:
        final, chair_msgs = await self.c._synthesize(self.topic, self.transcript)
        self.trace.extend(self._collect("chair", self.round_no, chair_msgs))
        document = build_document(self.topic, final, self.transcript)
        path, rel_path = self.c._save(self.topic, document)
        self.finalized = True
        return {
            "ideas": final, "path": path, "rel_path": rel_path,
            "n_models": len(self.panel), "rounds": self.round_no,
            "trace": self.trace,
        }
