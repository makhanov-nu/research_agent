"""LangGraph topology for the ideation consortium (Phase 1: core).

    make_brief
        │  (Send fan-out ×N)
        ▼
    run_panelist ─┐                      each run_panelist drives the panelist
        │ (fan-in)│                      subgraph:  agent ⇄ tools  → finalize
        ▼         │                                       └ budget guard ┘
    debate ───────┘
        ▼
    extract_debated ─► assemble_pool ─► run_scorer (Send fan-out, +chair) ─► aggregate

The panelist subgraph is the control point for the failures seen in council_31:

- **Budget guard** — `route_after_agent` forces the loop into `finalize` once the
  tool-call rounds are spent, instead of letting the model search into the
  `recursion_limit` wall and emit LangGraph's canned "need more steps" string
  (the GLM-5.1 failure). The recursion_limit stays only as a backstop.
- **Forced finalize + retry** — a tool-free emit turn; if the output doesn't
  parse into an idea, one retry, then drop (the Kimi empty-output failure is
  surfaced as a flag instead of a silent empty pool entry).

Phase 1 wires the topology with the EXISTING parsers/aggregator; the hardened
parser, quorum gate, and tests land in Phase 2.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from .consortium import (
    _DEBATE_EXTRACT_CHAIR,
    _DEBATE_OPEN,
    _PROPOSE_INDEP,
    _SCORE,
    _collect,
    _flatten,
    _panel_system,
    normalize_and_rank,
    parse_ideas,
    parse_scores,
    render_transcript,
)
from .state import ConsortiumState, PanelistState

logger = logging.getLogger(__name__)

# Per-phase tool-call-round budgets (rounds, not individual calls). Spent budget
# forces a finalize; PANELIST_RECURSION is a generous backstop above 2×budget.
PROPOSE_BUDGET = 8
DEBATE_BUDGET = 5
EXTRACT_BUDGET = 3
PANELIST_RECURSION = 40
MAX_FINALIZE_RETRIES = 1

_FINALIZE_PROPOSE = (
    "Stop searching now. Using everything above, write your ideas in the required "
    f"format. {_PROPOSE_INDEP}"
)
_FINALIZE_DEBATE = (
    "Stop searching now. State your strongest research direction and the key open "
    "gaps concisely, grounded in what you found above. (No special format.)"
)
_FINALIZE_EXTRACT = "Stop searching now. " + _DEBATE_EXTRACT_CHAIR
_FINALIZE_RETRY = (
    "Your previous reply did not contain a parseable idea. Output ONLY the "
    "idea(s), each beginning with a line `=== IDEA ===` followed by the required "
    "fields. No preamble, no commentary."
)
_SCORE_SYSTEM = (
    "You are a rigorous, independent expert reviewer for a top (Q1) venue in this "
    "research field."
)

_FINALIZE_ASK = {
    "propose": _FINALIZE_PROPOSE,
    "debate": _FINALIZE_DEBATE,
    "extract": _FINALIZE_EXTRACT,
}


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return _flatten(m.content)
    return ""


def _budget_tool_answers(messages: list) -> list:
    """Synthetic tool results for a trailing AIMessage whose tool_calls were cut
    off by the budget guard.

    When `route_after_agent` forces a finalize mid-search, the last message is an
    AIMessage carrying unanswered `tool_calls`. OpenAI-compatible providers
    (DeepInfra) reject the next request unless every `tool_call_id` is answered by
    a `tool` message, so we answer them synthetically before appending the
    finalize turn. Persisting these keeps the thread valid for the scoring replay
    too (the thread is reused by `run_scorer`). Empty when nothing dangles.
    """
    if not messages:
        return []
    last = messages[-1]
    calls = getattr(last, "tool_calls", None) if isinstance(last, AIMessage) else None
    if not calls:
        return []
    return [
        ToolMessage(
            content="Search budget reached — no further results. Answer now with what you have.",
            tool_call_id=tc.get("id", ""),
            name=tc.get("name", "tool"),
        )
        for tc in calls
    ]


# --- panelist subgraph: bounded agent loop + forced finalize ------------------

def build_panelist_subgraph(c):
    tool_node = ToolNode(c.tools, handle_tool_errors=True) if c.tools else None

    async def agent(state: PanelistState) -> dict:
        model = state["model"]
        system = _panel_system(
            model, shared=state.get("shared", False),
            phase="debate" if state["phase"] in ("debate", "extract") else "propose",
        )
        llm = c._chat(model, max_tokens=6000, with_tools=bool(c.tools))
        resp = await llm.ainvoke([SystemMessage(content=system), *state["messages"]])
        return {"messages": [resp]}

    def route_after_agent(state: PanelistState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
        rounds = sum(
            1 for m in state["messages"]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        if wants_tools and tool_node is not None and rounds <= state["budget"]:
            return "tools"
        return "finalize"  # no tools wanted, or budget spent → forced finalize

    async def finalize(state: PanelistState) -> dict:
        model, phase = state["model"], state["phase"]
        attempts = state.get("attempts", 0)
        ask = _FINALIZE_RETRY if attempts else _FINALIZE_ASK[phase]
        # A budget cutoff can leave the last AIMessage holding unanswered
        # tool_calls; answer them synthetically so the provider doesn't 400 and
        # so the persisted thread stays valid for the scoring replay.
        answers = _budget_tool_answers(state["messages"])
        llm = c._chat(model, max_tokens=6000, with_tools=False)
        resp = await llm.ainvoke([*state["messages"], *answers, HumanMessage(content=ask)])
        text = _flatten(resp.content)
        if phase == "debate":
            ideas = []  # debate emits a free-text position; the chair extracts later
        else:
            by = "panel" if phase == "extract" else model
            source = "debated" if phase == "extract" else "independent"
            ideas = [
                {"text": b, "source": source, "by": by}
                for b in parse_ideas(text, max_n=state["max_ideas"], model=model)
            ]
        return {
            "messages": [*answers, HumanMessage(content=ask), resp],
            "ideas": ideas,
            "attempts": attempts + 1,
        }

    def route_after_finalize(state: PanelistState) -> str:
        if state["phase"] == "debate" or state.get("ideas"):
            return END
        if state.get("attempts", 0) <= MAX_FINALIZE_RETRIES:
            return "finalize"
        return END  # give up; run_panelist flags the empty result

    b = StateGraph(PanelistState)
    b.add_node("agent", agent)
    b.add_node("finalize", finalize)
    b.add_edge(START, "agent")
    if tool_node is not None:
        b.add_node("tools", tool_node)
        b.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
        b.add_edge("tools", "agent")
    else:
        b.add_edge("agent", "finalize")
    b.add_conditional_edges("finalize", route_after_finalize, {"finalize": "finalize", END: END})
    return b.compile()


# --- parent graph -------------------------------------------------------------

def build_consortium_graph(c):
    panelist = build_panelist_subgraph(c)

    async def _run_loop(payload: dict) -> dict:
        return await panelist.ainvoke(payload, config={"recursion_limit": PANELIST_RECURSION})

    async def make_brief(state: ConsortiumState) -> dict:
        topic = state["topic"]
        try:
            text, trace = await c.make_brief(topic, state.get("focus", ""))
        except Exception as exc:  # noqa: BLE001 — degrade to the bare topic, don't abort
            logger.exception("chair brief failed")
            return {"brief": topic, "prior": "",
                    "flags": [f"chair brief failed ({type(exc).__name__}); using the bare topic"]}
        prior = await c._prior(topic)
        return {"brief": text, "prior": prior, "trace": trace}

    def route_propose(state: ConsortiumState) -> list:
        instr = f"Research brief:\n{state['brief']}\n\n{_PROPOSE_INDEP}"
        return [
            Send("run_panelist", {
                "model": m, "phase": "propose", "shared": False,
                "budget": PROPOSE_BUDGET, "max_ideas": 3, "attempts": 0,
                "messages": [HumanMessage(content=instr)],
            })
            for m in c.panel
        ]

    async def run_panelist(payload: dict) -> dict:
        model = payload["model"]
        try:
            out = await _run_loop(payload)
        except Exception as exc:  # noqa: BLE001 — one model failing must not abort the round
            logger.exception("panelist %s failed to propose", model)
            return {"ideas": [], "threads": {model: []}, "trace": [],
                    "flags": [f"{model} failed to propose ({type(exc).__name__})"]}
        ideas = out.get("ideas", [])
        return {
            "ideas": ideas,
            "threads": {model: out["messages"]},
            "trace": _collect(f"{model}:propose", out["messages"]),
            "flags": [] if ideas else [f"{model} produced no parseable idea (propose)"],
        }

    async def debate(state: ConsortiumState) -> dict:
        transcript = [("Brief", state["brief"])]
        if state.get("prior"):
            transcript.append(("Prior debate insights (memory)", state["prior"]))
        trace: list = []
        flags: list = []
        for m in c.panel:  # sequential: a shared, growing transcript
            seed = f"Debate so far:\n\n{render_transcript(transcript)}\n\n---\n{_DEBATE_OPEN}"
            try:
                out = await _run_loop({
                    "model": m, "phase": "debate", "shared": True,
                    "budget": DEBATE_BUDGET, "max_ideas": 0, "attempts": 0,
                    "messages": [HumanMessage(content=seed)],
                })
            except Exception as exc:  # noqa: BLE001 — skip this voice, keep the debate going
                logger.exception("debate panelist %s failed", m)
                transcript.append((m, f"[{m} could not contribute this round]"))
                flags.append(f"{m} failed in debate ({type(exc).__name__})")
                continue
            transcript.append((m, _last_ai_text(out["messages"])))
            trace += _collect(f"{m}:debate1", out["messages"])
        return {"transcript": transcript, "trace": trace, "flags": flags}

    async def extract_debated(state: ConsortiumState) -> dict:
        instr = f"{render_transcript(state['transcript'])}\n\n---\n{_DEBATE_EXTRACT_CHAIR}"
        try:
            out = await _run_loop({
                "model": c.chair_model, "phase": "extract", "shared": True,
                "budget": EXTRACT_BUDGET, "max_ideas": 2, "attempts": 0,
                "messages": [HumanMessage(content=instr)],
            })
        except Exception as exc:  # noqa: BLE001 — no debated ideas, but the independents still rank
            logger.exception("chair debate-extraction failed")
            return {"ideas": [], "flags": [f"chair extraction failed ({type(exc).__name__})"]}
        return {"ideas": out.get("ideas", []), "trace": _collect("chair:extract", out["messages"])}

    async def assemble_pool(state: ConsortiumState) -> dict:
        pool = [{**idea, "id": n} for n, idea in enumerate(state.get("ideas", []), 1)]
        return {"pool": pool}

    def route_score(state: ConsortiumState) -> list:
        pool, threads = state["pool"], state.get("threads", {})
        sends = [
            Send("run_scorer", {"model": m, "is_chair": False,
                                "pool": pool, "thread": threads.get(m, [])})
            for m in c.panel
        ]
        # Chair as a neutral rater (authors nothing → scores all): guarantees every
        # idea gets an independent cross-score and real rater overlap (problem E).
        sends.append(Send("run_scorer", {"model": c.chair_model, "is_chair": True,
                                         "pool": pool, "thread": []}))
        return sends

    async def run_scorer(payload: dict) -> dict:
        model, pool, is_chair = payload["model"], payload["pool"], payload["is_chair"]
        ballot = [
            i for i in pool
            if is_chair or not (i.get("source") == "independent" and i.get("by") == model)
        ]
        valid = {i["id"] for i in ballot}
        listing = "\n\n".join(f"#{i['id']}:\n{i['text']}" for i in ballot)
        prompt = f"{_SCORE}\n\nIdeas:\n\n{listing}"
        thread = payload.get("thread") or []
        messages = (
            thread + [HumanMessage(content=prompt)] if thread
            else [SystemMessage(content=_SCORE_SYSTEM), HumanMessage(content=prompt)]
        )
        # 6000 tokens: reasoning models (R1/GLM/Kimi) burn budget on a <think>
        # pass before emitting the ballot; too low truncates into a spurious
        # "failed to vote" (per the structured-output probe).
        llm = c._chat(model, max_tokens=6000, with_tools=False)
        try:
            resp = await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 — a rater failing must not abort scoring
            logger.exception("scorer %s failed", model)
            return {"ballots": [], "trace": [],
                    "flags": [f"{model} failed to vote ({type(exc).__name__})"]}
        parsed = parse_scores(_flatten(resp.content), valid)
        return {
            "ballots": [{"model": model, "scores": parsed}] if parsed else [],
            "trace": _collect(f"{model}:score", [resp]),
            "flags": [] if parsed else [f"{model} failed to vote"],
        }

    async def aggregate(state: ConsortiumState) -> dict:
        scores_by_model = {
            b["model"]: b["scores"] for b in state.get("ballots", []) if b["scores"]
        }
        pool = state["pool"]
        ranked = normalize_and_rank(pool, scores_by_model, min_raters=1)

        # Coverage guard. The chair always rates, so a panel collapse can't drop
        # an idea to zero raters here — but if too few PANELISTS cast a valid
        # ballot the peer signal is thin, so surface it loudly (council_31).
        flags: list[str] = []
        n_panel_ballots = sum(1 for m in c.panel if m in scores_by_model)
        quorum = (len(c.panel) + 1) // 2
        if n_panel_ballots < quorum:
            flags.append(
                f"⚠️ low-confidence ranking: only {n_panel_ballots}/{len(c.panel)} "
                "panelists cast a valid ballot"
            )
        n_dropped = len(pool) - len(ranked)
        if n_dropped:
            flags.append(
                f"⚠️ {n_dropped} idea(s) dropped from the ranking: scored by no rater"
            )
        return {"ranked": ranked, "flags": flags}

    g = StateGraph(ConsortiumState)
    g.add_node("make_brief", make_brief)
    g.add_node("run_panelist", run_panelist)
    g.add_node("debate", debate)
    g.add_node("extract_debated", extract_debated)
    g.add_node("assemble_pool", assemble_pool)
    g.add_node("run_scorer", run_scorer)
    g.add_node("aggregate", aggregate)

    g.add_edge(START, "make_brief")
    g.add_conditional_edges("make_brief", route_propose, ["run_panelist"])
    g.add_edge("run_panelist", "debate")          # fan-in: debate after all panelists
    g.add_edge("debate", "extract_debated")
    g.add_edge("extract_debated", "assemble_pool")
    g.add_conditional_edges("assemble_pool", route_score, ["run_scorer"])
    g.add_edge("run_scorer", "aggregate")         # fan-in: aggregate after all scorers
    g.add_edge("aggregate", END)
    return g.compile()
