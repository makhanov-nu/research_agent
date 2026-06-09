"""Discord client that bridges messages to the research agent graph.

The bot replies in DMs and when @-mentioned in a server channel. Each Discord
channel maps to its own conversation thread (LangGraph checkpoint). On startup
it opens the Postgres pool (if configured), builds a durable checkpointer + the
memory manager, and launches the background maintenance loop.

Commands:
  !checkpoint / !summarize  - summarize this thread to long-term memory + reset
  !remember <text>          - store a durable preference/instruction
  !help                     - show commands
"""

from __future__ import annotations

import asyncio
import logging
import re

import discord

from ..agent import build_graph
from ..config import settings
from ..db import build_checkpointer, open_pool
from ..llm import get_llm
from ..memory.maintenance import run_loop
from ..memory.manager import MemoryManager
from ..memory.summarize import summarize_messages

logger = logging.getLogger(__name__)

DISCORD_MAX_CHARS = 2000

HELP_TEXT = (
    "**Commands**\n"
    "`!checkpoint` (or `!summarize`) — summarize this thread to long-term "
    "memory and reset the live context\n"
    "`!remember <text>` — store a durable preference/instruction\n"
    "`!gpu <user@ip>` — attach a fresh GPU box (bare Ubuntu) and provision it\n"
    "`!runs` — list experiments and their status\n"
    "`!approve <id>` — approve and launch a pending experiment\n"
    "`!cancel <id>` — cancel a running experiment\n"
    "`!getfile <path>` — fetch a written output (e.g. a drafted LaTeX review)\n"
    "`!ideate <topic>` — convene the consortium: independent + debated proposals, "
    "scored 0–10, top 5 returned. Reply with numbers (e.g. `2,4`) to develop them, "
    "`!ideate again <notes>` for another polish+vote round, `!ideate done` to "
    "finalize, `!ideate cancel` to drop\n"
    "`!project` — show this chat's project + artifacts (`!project name <x>` to rename)\n"
    "`!tasks` — recent subagent tasks (the dashboard)\n"
    "`!task <id>` — a task's status + result\n"
    "`!trace <id>` — export a task's full trace (reasoning + tool calls) as a file\n"
    "`!feedback <id> <good|bad> [note]` — rate a task; I bank it as a lesson and "
    "a training label\n"
    "`!help` — this message\n\n"
    "Otherwise, just talk to me — DM or @-mention."
)


def _chunk(text: str, size: int = DISCORD_MAX_CHARS) -> list[str]:
    """Split text into Discord-sized chunks, preferring line boundaries."""
    if not text:
        return ["(empty response)"]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > size:  # a single very long line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(current) + len(line) > size:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


def _parse_picks(text: str) -> list[int]:
    """Extract idea numbers from a researcher's reply (e.g. '2,4' or 'do 1 and 3')."""
    return [int(n) for n in re.findall(r"\d+", text or "")]


class PerKeyLocks:
    """Lazily-created asyncio locks keyed by a string (e.g. a thread id).

    Used to serialize graph invocations within a channel while letting
    different channels run concurrently. Locks are kept for the process
    lifetime; the key space (Discord channels) is small and bounded in practice.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock


def checkpoint_result_message(memory_configured: bool, semantic_saved: bool) -> str:
    """User-facing result for !checkpoint, honest about where the summary went."""
    if not memory_configured:
        return (
            "Checkpointed and reset the live context. Memory isn't configured, "
            "so the summary wasn't stored anywhere durable."
        )
    if semantic_saved:
        return (
            "Checkpointed. Summary saved to long-term (semantic) memory and the "
            "conversation store; live context reset."
        )
    return (
        "Checkpointed and reset the live context. Semantic memory is disabled, "
        "so I saved the summary only in the conversation store."
    )


class ResearchBot(discord.Client):
    def __init__(self, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        self.graph = None
        self.memory: MemoryManager | None = None
        self.experiments = None
        self.consortium = None
        self.consortium_sessions: dict = {}  # channel_id -> ConsortiumSession
        self.projects = None
        self.tasks = None
        self.dispatcher = None
        self.llm = None
        self._pool = None
        self._maintenance_task: asyncio.Task | None = None
        self._poller_task: asyncio.Task | None = None
        # Serialize graph/state operations per channel (see issue: ordering).
        self._channel_locks = PerKeyLocks()

    async def setup_hook(self) -> None:
        self.llm = get_llm()
        self._pool = await open_pool()
        checkpointer = await build_checkpointer(self._pool)

        if settings.memory_enabled and self._pool is not None:
            self.memory = MemoryManager(self._pool)
            await self.memory.setup()
            self._maintenance_task = self.loop.create_task(
                run_loop(self.memory, self.llm)
            )

        # The experiment runner needs the registry (episodic store). The GPU box
        # is attached at runtime (`!gpu`, ephemeral per experiment), so the runner
        # is built whenever memory is configured — not gated on a preset host.
        if self.memory is not None:
            from ..experiments.runner import ExperimentRunner
            from ..experiments.ssh_docker import SSHDockerBackend
            from ..experiments.workspace import Workspace

            self.experiments = ExperimentRunner(
                self.memory.episodic,
                SSHDockerBackend(),
                Workspace(settings.experiment_workspace_dir),
                settings.experiment_artifacts_dir,
                memory=self.memory,
            )
            self._poller_task = self.loop.create_task(self._run_job_poller())

        # Project store (each chat is a project): folders + DB registry.
        from ..projects import ProjectStore

        self.projects = ProjectStore(self._pool, settings.output_dir)
        await self.projects.setup()
        if self.experiments is not None:
            self.experiments.projects = self.projects

        # Task dashboard (cross-subagent registry); needs the DB pool.
        if self._pool is not None:
            from ..agents.task_store import TaskStore

            self.tasks = TaskStore(self._pool)
            await self.tasks.setup()

        # Load MCP tools once and share them with the graph and the consortium.
        from ..mcp_client import load_mcp_tools

        mcp_tools = await load_mcp_tools()

        if settings.openrouter_api_key:
            from ..consortium import Consortium

            self.consortium = Consortium(
                mcp_tools, settings.panel_models, settings.consortium_chair_model,
                settings.output_dir, settings.consortium_temperature,
                recall=self.memory.recall_lessons if self.memory else None,
                debate_turns=settings.consortium_debate_turns,
            )

        # Background dispatcher for async/parallel subagent runs (needs the
        # task store to track them).
        if self.tasks is not None:
            from ..agents.dispatcher import TaskDispatcher, build_runners
            from ..writing import build_writers

            writers = build_writers(get_llm(), mcp_tools, settings.output_dir)
            runners = build_runners(
                model=get_llm(), mcp_tools=mcp_tools, writers=writers,
                consortium=self.consortium, projects=self.projects, memory=self.memory,
            )
            self.dispatcher = TaskDispatcher(
                runners, self.tasks, self._on_task_complete,
                settings.max_parallel_tasks,
            )

        self.graph = await build_graph(
            checkpointer, self.memory, self.experiments,
            mcp_tools=mcp_tools, consortium=self.consortium, task_store=self.tasks,
            dispatcher=self.dispatcher, projects=self.projects,
        )
        logger.info(
            "Research agent ready (memory=%s, experiments=%s, consortium=%s).",
            bool(self.memory), bool(self.experiments), bool(self.consortium),
        )

    async def close(self) -> None:
        for task in (self._maintenance_task, self._poller_task):
            if task:
                task.cancel()
        if self.dispatcher is not None:
            await self.dispatcher.shutdown()
        if self._pool is not None:
            await self._pool.close()
        await super().close()

    async def _notify_channel(self, channel_id: str | None, message: str) -> None:
        """Post a message to a channel by id (used for background notifications)."""
        if not channel_id:
            return
        channel = self.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self.fetch_channel(int(channel_id))
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                logger.warning("Cannot reach channel %s to notify", channel_id)
                return
        for chunk in _chunk(message):
            await channel.send(chunk)

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)

    def _spawn_background(self, coro, description: str) -> asyncio.Task:
        """Run a fire-and-forget coroutine, logging any exception it raises."""
        task = self.loop.create_task(coro)

        def _log_exception(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error("Background task failed: %s", description, exc_info=exc)

        task.add_done_callback(_log_exception)
        return task

    async def _run_job_poller(self) -> None:
        """Periodically check active experiments and report completions."""
        while True:
            try:
                for change in await self.experiments.poll_active():
                    await self._report_state_change(change)
            except Exception:  # noqa: BLE001
                logger.exception("Job poller iteration failed")
            await asyncio.sleep(settings.job_poll_interval_seconds)

    async def _report_state_change(self, change) -> None:
        await self._notify_channel(change.channel_id, change.message)

    async def _on_task_complete(self, task_id, agent, status, channel_id) -> None:
        """Trigger: a finished background task wakes the orchestrator.

        The event itself carries no result — the dashboard is the single source
        of truth, so we READ the result (or error) back from the task store,
        inject it as an automated event on the task's thread, run a graph turn,
        and post the orchestrator's response to the channel.
        """
        if self.graph is None or not channel_id:
            return
        row = await self.tasks.get(task_id) if self.tasks is not None else None
        if row is None:
            payload = "(result unavailable — not found in the task dashboard)"
        elif status == "failed":
            payload = f"The task failed: {row.get('error') or 'unknown error'}"
        else:
            payload = row.get("result") or "(the task recorded no result)"
        # Tag the project so concurrent projects' completions are unambiguous.
        proj_tag = ""
        if self.projects is not None:
            proj = await self.projects.get_by_channel(str(channel_id))
            if proj:
                proj_tag = f" [project: {proj['name']} #{proj['id']}]"
        event = (
            f"[BACKGROUND TASK COMPLETE] #{task_id} ({agent}){proj_tag} — {status}.\n\n"
            f"{payload}\n\n"
            "This is an automated event (not from the researcher). Incorporate "
            "this result with the ongoing work, reply with what matters, and "
            "dispatch any useful follow-ups. If nothing needs saying, reply 'OK'."
        )
        thread_id = str(channel_id)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            async with self._channel_locks.get(thread_id):
                result = await self.graph.ainvoke(
                    {"messages": [("user", event)]}, config=config
                )
            reply = _flatten(result["messages"][-1].content).strip()
        except Exception:  # noqa: BLE001
            logger.exception("Orchestrator failed handling completion of #%s", task_id)
            await self._notify_channel(
                channel_id, f"Task #{task_id} ({agent}) {status}, but I hit an error processing it."
            )
            return
        if reply and reply.upper() != "OK":
            await self._notify_channel(channel_id, reply)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user or message.author.bot:
            return

        is_dm = message.guild is None
        mentioned = self.user in message.mentions
        if not (is_dm or mentioned):
            return

        content = message.content
        if self.user:
            for token in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
                content = content.replace(token, "")
        content = content.strip()

        thread_id = str(message.channel.id)
        config = {"configurable": {"thread_id": thread_id}}

        if content.startswith("!"):
            await self._handle_command(message, content, config)
            return
        if not content:
            await message.channel.send("Hi — what are we researching?")
            return

        # When a consortium session is awaiting the researcher's pick (right after
        # the round-1 top-5), a plain reply with idea numbers selects them. In other
        # phases the researcher steers with `!ideate again/done`, so plain replies
        # fall through to normal chat.
        session = self.consortium_sessions.get(thread_id)
        if session is not None and not session.finalized and session.phase == "scored":
            picks = _parse_picks(content)
            if picks:
                await self._consortium_polish(message, session, picks=picks)
            else:
                await message.channel.send(
                    "Reply with the idea numbers to develop (e.g. `2,4`), or `!ideate done`."
                )
            return

        await self._handle_chat(message, content, config, thread_id)

    async def _handle_chat(self, message, content, config, thread_id) -> None:
        try:
            # Serialize per channel so concurrent messages in the same thread
            # don't race on shared checkpoint/memory state.
            async with self._channel_locks.get(thread_id):
                async with message.channel.typing():
                    result = await self.graph.ainvoke(
                        {"messages": [("user", content)]}, config=config
                    )
            reply = _flatten(result["messages"][-1].content)
            cumulative = result.get("cumulative_tokens", 0)
            summary = result.get("summary") or ""
        except Exception:  # noqa: BLE001
            logger.exception("Error handling message")
            await message.channel.send(
                "Something went wrong while I was thinking. Check the logs."
            )
            return

        for chunk in _chunk(reply):
            await message.channel.send(chunk)

        # Persist the exchange to memory without blocking the reply. Passing the
        # current summary keeps the durable episodic summary in sync with any
        # auto-summarization that happened this turn (so idle archival has it).
        if self.memory is not None:
            self._spawn_background(
                self.memory.remember(
                    thread_id, content, reply, cumulative, summary=summary
                ),
                f"memory.remember(channel={thread_id})",
            )

    async def _handle_command(self, message, content, config) -> None:
        cmd, _, arg = content[1:].partition(" ")
        cmd = cmd.lower()

        if cmd == "help":
            await message.channel.send(HELP_TEXT)
        elif cmd in {"checkpoint", "summarize"}:
            await self._checkpoint(message, config)
        elif cmd == "remember":
            arg = arg.strip()
            if not arg:
                await message.channel.send("Usage: `!remember <text>`")
            elif self.memory is not None:
                await self.memory.procedural.add(arg, kind="preference")
                await message.channel.send("Noted — I'll remember that.")
            else:
                await message.channel.send("Memory isn't configured, so I can't store that.")
        elif cmd in {"runs", "approve", "cancel"}:
            await self._handle_experiment_command(message, cmd, arg.strip())
        elif cmd == "gpu":
            await self._attach_gpu(message, arg.strip())
        elif cmd == "getfile":
            await self._send_file(message, arg.strip())
        elif cmd == "ideate":
            await self._ideate(message, arg.strip())
        elif cmd in {"tasks", "task", "trace"}:
            await self._handle_task_command(message, cmd, arg.strip())
        elif cmd == "feedback":
            await self._handle_feedback(message, arg.strip())
        elif cmd == "project":
            await self._handle_project_command(message, arg.strip())
        else:
            await message.channel.send(f"Unknown command `!{cmd}`.\n\n{HELP_TEXT}")

    async def _handle_project_command(self, message, arg) -> None:
        """`!project` shows this chat's project; `!project name <x>` renames it."""
        if self.projects is None:
            await message.channel.send("Projects aren't configured (needs the DB).")
            return
        channel_id = str(message.channel.id)
        if arg.lower().startswith("name "):
            new_name = arg[5:].strip()
            proj = await self.projects.rename(channel_id, new_name)
            await message.channel.send(f"Project renamed to **{proj['name']}** (`{proj['slug']}`).")
            return
        proj = await self.projects.ensure(channel_id)
        arts = await self.projects.list_artifacts(proj["id"]) if proj.get("id") else []
        by_kind: dict[str, int] = {}
        for a in arts:
            by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in by_kind.items()) or "no artifacts yet"
        await message.channel.send(
            f"**Project:** {proj['name']} (`{proj['slug']}`)\n"
            f"Artifacts — {summary}\n"
            f"Folder: `outputs/projects/{proj['slug']}/`  ·  rename with `!project name <x>`"
        )

    async def _handle_task_command(self, message, cmd, arg) -> None:
        if self.tasks is None:
            await message.channel.send("The task dashboard isn't configured (needs the DB).")
            return

        if cmd == "tasks":
            rows = await self.tasks.list_recent(limit=15)
            if not rows:
                await message.channel.send("No tasks yet.")
                return
            lines = [
                f"#{r['id']} [{r['status']}] {r['agent']} — {r['input'][:60]}"
                for r in rows
            ]
            await message.channel.send("**Tasks (recent)**\n" + "\n".join(lines))
            return

        if not arg.isdigit():
            await message.channel.send(f"Usage: `!{cmd} <task_id>`")
            return
        task = await self.tasks.get(int(arg))
        if task is None:
            await message.channel.send(f"No task #{arg}.")
            return

        if cmd == "task":
            body = (
                f"**Task #{task['id']}** — {task['agent']} [{task['status']}]\n"
                f"Input: {task['input'][:300]}\n\n"
                f"{(task.get('result') or task.get('error') or '(no result yet)')}"
            )
            for chunk in _chunk(body):
                await message.channel.send(chunk)
            return

        # cmd == "trace": export the full trace as a JSON file via outputs/
        import json
        from pathlib import Path

        traces_dir = Path(settings.output_dir) / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        out = traces_dir / f"task_{task['id']}.json"
        out.write_text(json.dumps(task.get("trace") or [], indent=2, default=str))
        await message.channel.send(
            f"Exported trace for task #{task['id']} "
            f"({len(task.get('trace') or [])} steps): `!getfile traces/{out.name}`"
        )

    async def _handle_feedback(self, message, arg) -> None:
        """`!feedback <task_id> <good|bad> [note]` — label a task + bank a lesson.

        The verdict becomes a training-quality label on the task row, and (when
        memory is on) a high-signal lesson tagged by the task's agent kind, so
        future jobs of that kind are primed with your correction.
        """
        if self.tasks is None:
            await message.channel.send("The task dashboard isn't configured (needs the DB).")
            return
        parts = arg.split(maxsplit=2)
        if len(parts) < 2 or not parts[0].isdigit() or parts[1].lower() not in {"good", "bad"}:
            await message.channel.send("Usage: `!feedback <task_id> <good|bad> [note]`")
            return
        task_id, quality = int(parts[0]), parts[1].lower()
        note = parts[2].strip() if len(parts) > 2 else ""

        task = await self.tasks.get(task_id)
        if task is None:
            await message.channel.send(f"No task #{task_id}.")
            return

        ok = await self.tasks.set_feedback(task_id, quality, note or None)
        agent = task.get("agent") or "subagent"
        banked = False
        if ok and self.memory is not None:
            try:
                project = (
                    await self.projects.get_by_channel(str(message.channel.id))
                    if self.projects is not None else None
                )
                verdict = "was good — reuse this approach" if quality == "good" else "needs correction"
                lesson = (
                    f"[user feedback] A '{agent}' result for "
                    f"\"{(task.get('input') or '')[:160]}\" {verdict}."
                    + (f" Specifically: {note}" if note else "")
                )
                await self.memory.record_lesson(
                    lesson, kind=agent, channel_id=str(message.channel.id),
                    status=quality, project=(project["slug"] if project else None),
                )
                banked = True
            except Exception:  # noqa: BLE001 — label is saved; banking a lesson is best-effort
                logger.exception("Failed to bank feedback lesson for task #%s", task_id)

        if ok:
            tail = " and banked a lesson for next time." if banked else "."
            await message.channel.send(f"Logged **{quality}** feedback on task #{task_id}{tail}")
        else:
            await message.channel.send(f"Couldn't update task #{task_id}.")

    async def _ideate(self, message, arg) -> None:
        """Drive the consortium.

        `!ideate <topic>` runs round 1 (independent + debated proposals, then
        scoring) and posts the top 5; reply with numbers (or `!ideate pick 2,4`) to
        develop them; `!ideate again <notes>` runs another polish+vote round;
        `!ideate done` finalizes; `!ideate cancel` drops the session.
        """
        if self.consortium is None:
            await message.channel.send(
                "The consortium isn't configured (needs OPENROUTER_API_KEY)."
            )
            return

        thread_id = str(message.channel.id)
        session = self.consortium_sessions.get(thread_id)
        sub, _, rest = arg.strip().partition(" ")
        sub, rest = sub.lower(), rest.strip()

        if sub in {"done", "finish", "finalize"}:
            if session is not None:
                await self._finalize_consortium(message, session)
            else:
                await message.channel.send("No active session. Start one with `!ideate <topic>`.")
            return
        if sub in {"cancel", "stop", "abort"}:
            if session is not None:
                self.consortium_sessions.pop(thread_id, None)
                await message.channel.send("Consortium session cancelled.")
            else:
                await message.channel.send("No active consortium session to cancel.")
            return
        if sub in {"again", "refine", "another"}:
            if session is not None and session.phase == "polished":
                await self._consortium_polish(message, session, picks=None, comments=rest)
            else:
                await message.channel.send("Nothing to refine yet — pick ideas from a round-1 top-5 first.")
            return
        if sub == "pick":
            if session is not None and session.phase == "scored" and _parse_picks(rest):
                await self._consortium_polish(message, session, picks=_parse_picks(rest))
            else:
                await message.channel.send("`!ideate pick <numbers>` works on a round-1 top-5.")
            return

        # An active session: route by phase.
        if session is not None and not session.finalized:
            if session.phase == "scored" and _parse_picks(arg):
                await self._consortium_polish(message, session, picks=_parse_picks(arg))
            else:
                await message.channel.send(self._ideate_hint(session))
            return

        # No active session: start a fresh one on the given topic.
        if not arg.strip():
            await message.channel.send("Usage: `!ideate <topic>`")
            return
        session = self.consortium.new_session(arg.strip())
        self.consortium_sessions[thread_id] = session
        await message.channel.send(
            f"Convening the consortium on **{arg.strip()}**.\n"
            f"Panel: {', '.join(self.consortium.panel)} · chair: {self.consortium.chair_model}.\n"
            "Round 1: each model proposes 3 ideas **independently**, a separate "
            "**debate** track adds more, then everyone scores all of them. I'll post "
            "the top 5. (A few minutes…)"
        )
        await self._consortium_round1(message, session)

    @staticmethod
    def _ideate_hint(session) -> str:
        if session.phase == "scored":
            return "Reply with idea numbers to develop (e.g. `2,4`), or `!ideate done`."
        if session.phase == "polished":
            return "Reply `!ideate again <notes>` for another round, or `!ideate done` to finalize."
        return "The panel is working — one moment…"

    async def _consortium_round1(self, message, session) -> None:
        if session.busy:
            await message.channel.send("The panel is still deliberating — one moment…")
            return
        session.busy = True
        try:
            async with message.channel.typing():
                await session.run_round1()
        except Exception:  # noqa: BLE001
            logger.exception("Consortium round 1 failed")
            await message.channel.send("The consortium hit an error this round. Check the logs.")
            return
        finally:
            session.busy = False

        for chunk in _chunk("**Round 1 — top 5 (scored 0–10)**\n\n" + session.render_top()):
            await message.channel.send(chunk)
        await message.channel.send(
            "Reply with the numbers to develop (e.g. `2,4`), or `!ideate done` to finalize."
        )

    async def _consortium_polish(self, message, session, picks, comments: str = "") -> None:
        if session.busy:
            await message.channel.send("The panel is still deliberating — one moment…")
            return
        session.busy = True
        try:
            async with message.channel.typing():
                await session.select_and_polish(picks=picks, comments=comments)
        except Exception:  # noqa: BLE001
            logger.exception("Consortium polish round failed")
            await message.channel.send("The panel hit an error polishing. Check the logs.")
            return
        finally:
            session.busy = False

        for chunk in _chunk(f"**Round {session.round_no} — polished & voted**\n\n" + session.render_top()):
            await message.channel.send(chunk)
        await message.channel.send(
            "Reply `!ideate again <notes>` for another round, or `!ideate done` to finalize."
        )

    async def _finalize_consortium(self, message, session) -> None:
        if session.busy:
            await message.channel.send("The panel is still deliberating — one moment…")
            return
        session.busy = True
        try:
            async with message.channel.typing():
                result = await session.finalize()
        except Exception:  # noqa: BLE001
            logger.exception("Consortium finalize failed")
            await message.channel.send("The chair hit an error finalizing. Check the logs.")
            session.busy = False
            return
        session.busy = False
        self.consortium_sessions.pop(str(message.channel.id), None)

        # Save the chosen proposal(s) into the project's council folder.
        council_rel = ""
        if self.projects is not None:
            from ..projects import save_council_proposal

            project = await self.projects.ensure(str(message.channel.id))
            council_rel = await save_council_proposal(
                self.projects, project, session.topic, result["ideas"]
            )

        # Capture the session (incl. the debate) to memory so future ideation's
        # debate track recalls it.
        from ..consortium import capture_council

        await capture_council(
            self.memory, str(message.channel.id), session.topic,
            result["ideas"], result["rel_path"], rounds=result["rounds"],
        )

        for chunk in _chunk(result["ideas"]):
            await message.channel.send(chunk)
        tail = (
            f"Done after {result['rounds']} round(s). "
            f"Full transcript (ideas + debates): `!getfile {result['rel_path']}`\n"
        )
        if council_rel:
            tail += f"Saved for methodology: `!getfile {council_rel}`\n"
        tail += "Want me to hand a chosen idea to the methodology writer? Just say the word."
        await message.channel.send(tail)

    async def _send_file(self, message, relpath: str) -> None:
        if not relpath:
            await message.channel.send("Usage: `!getfile <path>` (relative to outputs/)")
            return
        from pathlib import Path

        base = Path(settings.output_dir).resolve()
        target = (base / relpath).resolve()
        # Confine to the outputs directory; reject traversal / absolute escapes.
        if target != base and base not in target.parents:
            await message.channel.send("Path is outside the outputs directory.")
            return
        if not target.is_file():
            await message.channel.send(f"No such file: `{relpath}`")
            return
        # Discord's default non-boosted upload limit is 25 MB; stay well under.
        if target.stat().st_size > 8 * 1024 * 1024:
            await message.channel.send("File is too large to upload (>8 MB).")
            return
        await message.channel.send(file=discord.File(str(target)))

    async def _attach_gpu(self, message, arg) -> None:
        """Attach a fresh GPU box (`!gpu user@ip`) and provision it in the background."""
        if self.experiments is None:
            await message.channel.send(
                "Experiment runner isn't configured (needs a database)."
            )
            return
        if not arg:
            await message.channel.send("Usage: `!gpu <user@ip>` (bare Ubuntu, your SSH key).")
            return

        msg = await self.experiments.set_compute(arg)
        await message.channel.send(
            f"{msg}\nProvisioning (Docker + NVIDIA toolkit + MLflow) — I'll report back."
        )

        async def _provision_and_report() -> None:
            try:
                report = await self.experiments.provision()
                survey = await self.experiments.survey()
            except Exception as exc:  # noqa: BLE001
                await self._notify_channel(
                    str(message.channel.id),
                    f"GPU provisioning failed: {exc}",
                )
                return
            tail = "\n".join(report.splitlines()[-15:])
            await self._notify_channel(
                str(message.channel.id),
                f"✅ GPU box ready.\n```\n{survey}\n```\n_provision log (tail):_\n```\n{tail}\n```",
            )

        self._spawn_background(_provision_and_report(), "gpu.provision")

    async def _handle_experiment_command(self, message, cmd, arg) -> None:
        if self.experiments is None:
            await message.channel.send("Experiment runner isn't configured.")
            return

        if cmd == "runs":
            rows = await self.memory.episodic.list_experiments(limit=15)
            if not rows:
                await message.channel.send("No experiments yet.")
                return
            lines = [f"#{r['id']} [{r['status']}] {r['title']}" for r in rows]
            await message.channel.send("**Experiments**\n" + "\n".join(lines))
            return

        if not arg.isdigit():
            await message.channel.send(f"Usage: `!{cmd} <experiment_id>`")
            return
        exp_id = int(arg)
        async with message.channel.typing():
            if cmd == "approve":
                result = await self.experiments.approve_and_launch(exp_id)
            else:  # cancel
                result = await self.experiments.cancel(exp_id)
        await message.channel.send(result)

    async def _checkpoint(self, message, config) -> None:
        from langchain_core.messages import RemoveMessage

        thread_id = config["configurable"]["thread_id"]

        # Hold the channel lock across read-summarize-reset so it can't
        # interleave with a concurrent chat turn on the same thread.
        async with self._channel_locks.get(thread_id):
            snapshot = await self.graph.aget_state(config)
            messages = snapshot.values.get("messages", []) if snapshot else []
            if not messages:
                await message.channel.send("Nothing to checkpoint yet.")
                return

            semantic_saved = False
            async with message.channel.typing():
                summary = await summarize_messages(
                    self.llm, messages, snapshot.values.get("summary", "")
                )
                removes = [
                    RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None)
                ]
                await self.graph.aupdate_state(
                    config, {"summary": summary, "messages": removes}
                )

                if self.memory is not None:
                    await self.memory.episodic.set_summary(thread_id, summary)
                    if self.memory.semantic.enabled:
                        await asyncio.to_thread(
                            self.memory.semantic.remember,
                            "Conversation checkpoint:",
                            summary,
                            f"channel:{thread_id}",
                        )
                        semantic_saved = True

        await message.channel.send(
            checkpoint_result_message(self.memory is not None, semantic_saved)
        )
