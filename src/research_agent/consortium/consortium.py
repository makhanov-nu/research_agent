"""The shared-session ideation consortium."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import build_openrouter_chat, get_llm

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
    """Assemble the saved markdown: final ideas + full discussion appendix."""
    parts = [
        f"# Research ideas — {topic}",
        "",
        "## Selected ideas (chair synthesis)",
        final.strip(),
        "",
        "---",
        "## Full panel discussion (shared session)",
        render_transcript(transcript),
    ]
    return "\n".join(parts)


def _panel_system(model: str) -> str:
    return (
        f"You are {model}, one of several frontier AI models sitting on a research "
        "ideation panel. This is ONE shared session: you can see everything other "
        "panelists have said and must engage with it directly — build on, refine, "
        "or push back against their points by name. Be rigorous, specific, and "
        "honest about novelty; ground claims in the background literature provided."
    )


_PROPOSE = (
    "Round 1 — opening proposals. Propose 2-3 concrete, ambitious-but-feasible "
    "research ideas on the topic, each targeting a top (Q1) conference or journal. "
    "If earlier panelists already proposed ideas, react to theirs and differentiate "
    "yours. For each idea give: title, core idea, novelty vs the background "
    "literature, a rough method, and the expected contribution."
)

_DEBATE = (
    "Debate round — critique and converge. Engage directly with the other "
    "panelists' ideas: challenge novelty, feasibility, and impact; merge or improve "
    "where useful; and argue toward the strongest 3 candidates. Explicitly flag any "
    "idea that likely already exists in the literature."
)

_GROUND_SYSTEM = (
    "You are preparing a tight background brief for a research ideation panel. Use "
    "the literature tools to find what already exists on the topic, then write a "
    "concise summary of the state of the art and — most importantly — the concrete "
    "open gaps and opportunities. Cite real papers you retrieved; do not invent any."
)


def _chair_synthesis_prompt(topic: str) -> str:
    return (
        "You are the chair. Using the background brief and the full panel discussion "
        f"above, select and refine EXACTLY 3 research ideas on '{topic}' most likely "
        "to yield Q1 conference/journal papers. For each idea provide: **Title**, "
        "**Problem & motivation**, **Novelty** (explicitly contrasted with the "
        "background literature), **Method sketch**, **Why Q1 / expected impact**, "
        "**Key risks**, and **Suggested venue(s)**. Be concrete and honest about "
        "novelty risk. Format in clean Markdown."
    )


class Consortium:
    def __init__(self, lit_tools, panel_models, chair_model, output_dir,
                 temperature: float = 0.6, rounds: int = 1):
        self.lit_tools = lit_tools or []
        self.panel = list(panel_models)
        self.chair_model = chair_model
        self.output_dir = Path(output_dir) / "ideas"
        self.temperature = temperature
        self.rounds = max(0, rounds)

    async def _ground(self, topic: str, focus: str) -> str:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(get_llm(), self.lit_tools, prompt=_GROUND_SYSTEM)
        task = f"Topic: {topic}" + (f"\nFocus: {focus}" if focus else "")
        try:
            res = await agent.ainvoke(
                {"messages": [("user", task)]}, config={"recursion_limit": 40}
            )
            return _flatten(res["messages"][-1].content)
        except Exception:  # noqa: BLE001
            logger.exception("Consortium grounding failed")
            return "(background brief unavailable)"

    async def _speak(self, model: str, instruction: str,
                     transcript: list[tuple[str, str]]) -> str:
        messages = [
            SystemMessage(content=_panel_system(model)),
            HumanMessage(
                content=(
                    f"Panel discussion so far:\n\n{render_transcript(transcript)}\n\n"
                    f"---\n{instruction}"
                )
            ),
        ]
        try:
            resp = await build_openrouter_chat(model, self.temperature).ainvoke(messages)
            return _flatten(resp.content)
        except Exception as exc:  # noqa: BLE001 — one model failing must not abort the panel
            logger.exception("Panelist %s failed", model)
            return f"[{model} could not respond: {exc}]"

    async def ideate(self, topic: str, focus: str = "") -> dict:
        transcript: list[tuple[str, str]] = []

        grounding = await self._ground(topic, focus)
        transcript.append(("Background (literature)", grounding))

        # Round 1: opening proposals (sequential so each hears the prior speakers).
        for model in self.panel:
            transcript.append((model, await self._speak(model, _PROPOSE, transcript)))

        # Debate rounds.
        for _ in range(self.rounds):
            for model in self.panel:
                transcript.append((model, await self._speak(model, _DEBATE, transcript)))

        # Chair synthesis over the full shared transcript.
        final = await self._speak(
            self.chair_model, _chair_synthesis_prompt(topic), transcript
        )

        document = build_document(topic, final, transcript)
        path, rel_path = self._save(topic, document)
        return {"ideas": final, "path": path, "rel_path": rel_path, "n_models": len(self.panel)}

    def _save(self, topic: str, document: str) -> tuple[str, str]:
        from ..writing.lit_review import slugify

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{slugify(topic)}-{stamp}.md"
        path = self.output_dir / name
        path.write_text(document)
        return str(path), f"ideas/{name}"
