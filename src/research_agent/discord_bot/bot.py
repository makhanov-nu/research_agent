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
    "`!runs` — list experiments and their status\n"
    "`!approve <id>` — approve and launch a pending experiment\n"
    "`!cancel <id>` — cancel a running experiment\n"
    "`!getfile <path>` — fetch a written output (e.g. a drafted LaTeX review)\n"
    "`!ideate <topic>` — convene the multi-model consortium to propose 3 Q1 ideas\n"
    "`!tasks` — recent subagent tasks (the dashboard)\n"
    "`!task <id>` — a task's status + result\n"
    "`!trace <id>` — export a task's full trace (reasoning + tool calls) as a file\n"
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

        # The experiment runner needs the registry (episodic store), so it's
        # only available when memory is configured.
        if settings.compute_enabled and self.memory is not None:
            from ..experiments.runner import ExperimentRunner
            from ..experiments.ssh_docker import SSHDockerBackend
            from ..experiments.workspace import Workspace

            self.experiments = ExperimentRunner(
                self.memory.episodic,
                SSHDockerBackend(),
                Workspace(settings.experiment_workspace_dir),
                settings.experiment_artifacts_dir,
            )
            self._poller_task = self.loop.create_task(self._run_job_poller())

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
                settings.consortium_rounds,
            )

        # Background dispatcher for async/parallel subagent runs (needs the
        # task store to track them).
        if self.tasks is not None:
            from ..agents.dispatcher import TaskDispatcher, build_runners
            from ..writing.lit_review import LiteratureReviewer

            reviewer = LiteratureReviewer(get_llm(), mcp_tools, settings.output_dir)
            runners = build_runners(
                model=get_llm(), mcp_tools=mcp_tools, reviewer=reviewer,
                consortium=self.consortium,
            )
            self.dispatcher = TaskDispatcher(
                runners, self.tasks, self._notify_channel,
                settings.max_parallel_tasks,
            )

        self.graph = await build_graph(
            checkpointer, self.memory, self.experiments,
            mcp_tools=mcp_tools, consortium=self.consortium, task_store=self.tasks,
            dispatcher=self.dispatcher,
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
        elif cmd == "getfile":
            await self._send_file(message, arg.strip())
        elif cmd == "ideate":
            await self._ideate(message, arg.strip())
        elif cmd in {"tasks", "task", "trace"}:
            await self._handle_task_command(message, cmd, arg.strip())
        else:
            await message.channel.send(f"Unknown command `!{cmd}`.\n\n{HELP_TEXT}")

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

    async def _ideate(self, message, topic) -> None:
        if self.consortium is None:
            await message.channel.send(
                "The consortium isn't configured (needs OPENROUTER_API_KEY)."
            )
            return
        if not topic:
            await message.channel.send("Usage: `!ideate <topic>`")
            return

        models = ", ".join(self.consortium.panel)
        await message.channel.send(
            f"Convening the consortium on **{topic}**.\n"
            f"Panel: {models}. They'll ground in the literature, then "
            "propose → debate → synthesize. This takes a few minutes…"
        )
        try:
            async with message.channel.typing():
                result = await self.consortium.ideate(topic)
        except Exception:  # noqa: BLE001
            logger.exception("Consortium failed")
            await message.channel.send("The consortium hit an error. Check the logs.")
            return

        for chunk in _chunk(result["ideas"]):
            await message.channel.send(chunk)
        await message.channel.send(
            f"Full shared-session transcript: `!getfile {result['rel_path']}`"
        )

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
