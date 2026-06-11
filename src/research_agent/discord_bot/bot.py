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

# Attachment extensions we read as UTF-8 text (PDFs go through pypdf separately).
TEXT_EXTS = {
    "txt", "text", "md", "markdown", "tex", "latex", "bib", "rst",
    "csv", "tsv", "json", "yaml", "yml", "py", "log",
}

# Image extensions saved as binary to the project uploads folder.
IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "tiff", "tif"}

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
    "`!mute` — stop following this channel until you @-mention (or reply to) me again\n"
    "`!help` — this message\n\n"
    "Otherwise, just talk to me — DM, @-mention, or reply. Once you tag me in a "
    "channel or thread I'll keep following your messages there (and read any PDF / "
    "LaTeX / text files you attach) until you `!mute`."
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
        # Sticky engagement: channel_id -> set of author ids the bot is following
        # there (so it answers without a fresh @-mention each turn). In-process,
        # so a restart resets it; restored from checkpoints in setup_hook.
        self._engaged: dict[str, set[int]] = {}
        # Threads the bot has ever participated in (restored from DB on startup).
        # Used so the bot responds to messages in existing threads without
        # needing a fresh @-mention after a restart.
        self._known_threads: set[str] = set()
        # Channels we've already created a project for this process (cache so we
        # don't re-hit the DB on every message).
        self._ensured_projects: set[str] = set()
        # Per-message reply destinations: message_id -> thread (or channel).
        # discord.Message uses __slots__ so we can't set attributes on it directly.
        self._msg_dest: dict[int, discord.abc.Messageable] = {}

    async def _restore_known_threads(self) -> None:
        """Populate _known_threads from the checkpoints table on startup.

        After a restart _engaged is empty, so the bot would ignore messages in
        threads it previously participated in. Reading the checkpoint thread IDs
        once at startup lets it resume without requiring a fresh @-mention.
        """
        if self._pool is None:
            return
        try:
            async with self._pool.connection() as conn:
                rows = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
                self._known_threads.update(r["thread_id"] for r in await rows.fetchall())
            logger.info("Restored %d known thread(s) from checkpoints.", len(self._known_threads))
        except Exception:  # noqa: BLE001
            logger.exception("Could not restore known threads from checkpoints.")

    async def setup_hook(self) -> None:
        self.llm = get_llm()
        self._pool = await open_pool()
        checkpointer = await build_checkpointer(self._pool)
        await self._restore_known_threads()

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
            from ..llm import get_llm_for_role

            writers = build_writers(get_llm(), mcp_tools, settings.output_dir,
                                   model_for_role=get_llm_for_role)
            runners = build_runners(
                model=get_llm(), mcp_tools=mcp_tools, writers=writers,
                consortium=self.consortium, projects=self.projects, memory=self.memory,
                task_store=self.tasks, model_for_role=get_llm_for_role,
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
        # channel_id may be a thread; resolve to the parent channel for project lookup.
        proj_tag = ""
        if self.projects is not None:
            chan = self.get_channel(int(channel_id)) if channel_id else None
            project_key = self._project_channel_id(chan) if chan else str(channel_id) if channel_id else None
            proj = await self.projects.get_by_channel(project_key)
            if proj:
                proj_tag = f" [project: {proj['name']} #{proj['id']}]"
        event = (
            f"[BACKGROUND TASK COMPLETE] #{task_id} ({agent}){proj_tag} — {status}.\n\n"
            f"{payload}\n\n"
            "This is an automated event (not from the researcher). The researcher "
            "has already been notified of the result. Add context, propose next "
            "steps, or dispatch follow-up work if relevant. If nothing needs to "
            "be added, reply 'OK'."
        )
        thread_id = str(channel_id)
        config = {"configurable": {"thread_id": thread_id}}

        # Always send a direct notification (success or failure) so the user is
        # never left wondering. Don't rely solely on the orchestrator for this —
        # it might reply "OK" and suppress the message, leaving the thread silent.
        if status == "failed":
            error_summary = (row.get("error") or "unknown error") if row else "task not found"
            await self._notify_channel(
                channel_id, f"❌ Task #{task_id} ({agent}){proj_tag} failed: {error_summary}"
            )
            return

        # Send a compact success notice immediately so the user knows the task
        # finished, then let the orchestrator add context / next steps.
        task_result = row.get("result") or "(no result recorded)" if row else "(result unavailable)"
        await self._notify_channel(
            channel_id,
            f"✅ Task #{task_id} ({agent}){proj_tag} done.\n{task_result}",
        )

        try:
            async with self._channel_locks.get(thread_id):
                result = await self.graph.ainvoke(
                    {"messages": [("user", event)]}, config=config
                )
            reply = _flatten(result["messages"][-1].content).strip()
        except Exception:  # noqa: BLE001
            logger.exception("Orchestrator failed handling completion of #%s", task_id)
            return
        # Only relay the orchestrator's follow-up if it adds something beyond
        # a bare acknowledgement (the direct notice above already confirmed success).
        if reply and reply.upper() not in ("OK", "OK.", "DONE", "DONE."):
            await self._notify_channel(channel_id, reply)

    async def on_message(self, message: discord.Message) -> None:
        try:
            await self._handle_message(message)
        finally:
            self._msg_dest.pop(message.id, None)

    async def _handle_message(self, message: discord.Message) -> None:
        if message.author == self.user or message.author.bot:
            return

        is_dm = message.guild is None
        mentioned = bool(self.user) and self.user in message.mentions
        replied_to_me = await self._is_reply_to_me(message)
        addressed = is_dm or mentioned or replied_to_me

        author_id = message.author.id
        check_id = str(message.channel.id)
        engaged = (author_id in self._engaged.get(check_id, set())
                   or check_id in self._known_threads)

        # Answer when addressed, or when already engaged in this channel (sticky,
        # scoped to this author, or restored from checkpoints after a restart).
        if not (addressed or engaged):
            return

        # First @-mention in a plain text channel: open a thread so the bot's
        # replies don't flood the channel wall. Subsequent messages arrive inside
        # the thread (message.channel IS the thread), so this branch won't fire again.
        is_in_thread = isinstance(message.channel, discord.Thread)
        if addressed and not is_dm and not is_in_thread:
            dest = await self._to_thread(message)
            self._msg_dest[message.id] = dest
            thread_id = str(dest.id)
        else:
            thread_id = check_id

        # Being addressed (re)engages this author in this thread until !mute.
        if addressed and not is_dm:
            self._engaged.setdefault(thread_id, set()).add(author_id)
            self._known_threads.add(thread_id)

        # Ensure the channel's project exists (threads share the parent channel's project).
        await self._ensure_project(message)

        content = message.content
        if self.user:
            for token in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
                content = content.replace(token, "")
        content = content.strip()

        config = {"configurable": {"thread_id": thread_id}}

        # Commands are explicit and always work (even with no other text).
        if content.startswith("!"):
            await self._handle_command(message, content, config)
            return

        # Read any uploaded files (PDF via pypdf; .tex/.md/.txt as text). The full
        # text is saved as a project artifact; a capped copy is threaded inline.
        attach_text, attach_notes = await self._ingest_attachments(message)

        if not content and not attach_text:
            await self._ch(message).send(attach_notes or "Hi — what are we researching?")
            return

        # A plain reply of idea numbers picks consortium ideas while a session is
        # awaiting the choice (now works mention-free because we're engaged).
        session = self.consortium_sessions.get(thread_id)
        if session is not None and not session.finalized and session.phase == "scored":
            picks = _parse_picks(content)
            if picks:
                await self._consortium_polish(message, session, picks=picks)
            else:
                await self._ch(message).send(
                    "Reply with the idea numbers to develop (e.g. `2,4`), or `!ideate done`."
                )
            return

        if attach_notes:  # surface unreadable-file warnings alongside the answer
            await self._ch(message).send(attach_notes)

        effective = (content + ("\n\n" + attach_text if attach_text else "")).strip()
        await self._handle_chat(message, effective, config, thread_id)

    async def _is_reply_to_me(self, message: discord.Message) -> bool:
        """True if the message is a Discord 'reply' to one of the bot's messages."""
        ref = getattr(message, "reference", None)
        if ref is None:
            return False
        resolved = getattr(ref, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved.author == self.user
        ref_id = getattr(ref, "message_id", None)
        if ref_id:
            try:
                ref_msg = await message.channel.fetch_message(ref_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                return False
            return ref_msg.author == self.user
        return False

    @staticmethod
    def _channel_label(message: discord.Message) -> str | None:
        """A friendly project name from the Discord surface (thread/channel/DM)."""
        ch = message.channel
        name = getattr(ch, "name", None)  # TextChannel/Thread have .name; DMs don't
        if name:
            parent = getattr(ch, "parent", None)
            pname = getattr(parent, "name", None)
            return f"{pname}/{name}" if pname else name
        author = getattr(message.author, "display_name", None) or "researcher"
        return f"DM · {author}"

    @staticmethod
    def _project_channel_id(channel) -> str:
        """Project key: parent channel id for threads, own id otherwise.

        Threads inside a server channel share that channel's project so all
        conversations in e.g. #sgprotonet appear under one research project.
        """
        parent_id = getattr(channel, "parent_id", None)
        return str(parent_id) if parent_id else str(channel.id)

    @staticmethod
    def _project_name(channel, author=None) -> str:
        """Human-readable project name: parent channel for threads, channel name, or DM."""
        parent = getattr(channel, "parent", None)
        if parent is not None:
            return getattr(parent, "name", None) or "project"
        name = getattr(channel, "name", None)
        if name:
            return name
        display = getattr(author, "display_name", None) or "researcher"
        return f"DM · {display}"

    def _ch(self, message: discord.Message) -> discord.abc.Messageable:
        """Reply target: the thread opened this turn, or the original channel."""
        return self._msg_dest.get(message.id) or message.channel

    async def _to_thread(self, message: discord.Message) -> discord.abc.Messageable:
        """Create a public thread on a text-channel message and return it.

        Falls back silently to the channel if the bot lacks permission.
        """
        content = message.content
        if self.user:
            for token in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
                content = content.replace(token, "")
        content = content.strip()
        name = content[:80] if content else f"Research · {message.author.display_name}"
        try:
            return await message.create_thread(name=name or "Research", auto_archive_duration=1440)
        except (discord.Forbidden, discord.HTTPException):
            logger.warning("Could not create thread in %s, replying in channel", message.channel)
            return message.channel

    async def _ensure_project(self, message: discord.Message) -> None:
        """Create (or reuse) the project for this channel/thread (idempotent).

        Threads share their parent channel's project so all conversations inside
        #sgprotonet appear under one research project in the dashboard.
        """
        if self.projects is None:
            return
        project_key = self._project_channel_id(self._ch(message))
        if project_key in self._ensured_projects:
            return
        try:
            await self.projects.ensure(
                project_key,
                name=self._project_name(self._ch(message), message.author),
            )
            self._ensured_projects.add(project_key)
        except Exception:  # noqa: BLE001 — a project hiccup must not drop the message
            logger.exception("Failed to ensure project for channel %s", project_key)

    async def _ingest_attachments(self, message: discord.Message) -> tuple[str, str]:
        """Download uploaded files → (inline_text, notes).

        PDFs are text-extracted via pypdf; recognized text files are decoded as
        UTF-8. Each file's full text is saved as a project 'upload' artifact, and
        a length-capped copy is returned for inline context. `notes` collects
        user-facing warnings about files we couldn't read.
        """
        atts = getattr(message, "attachments", None) or []
        if not atts:
            return "", ""
        blocks: list[str] = []
        notes: list[str] = []
        for att in atts:
            name = att.filename or "attachment"
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            ctype = (att.content_type or "").lower()
            if (att.size or 0) > settings.attachment_max_bytes:
                notes.append(
                    f"⚠️ `{name}` is too large to read "
                    f"({(att.size or 0) // (1024 * 1024)} MB)."
                )
                continue
            try:
                data = await att.read()
            except Exception:  # noqa: BLE001
                logger.exception("Attachment download failed: %s", name)
                notes.append(f"⚠️ Couldn't download `{name}`.")
                continue

            if ext == "pdf" or "pdf" in ctype:
                text = self._extract_pdf_text(data)
                if len(text.strip()) < 40:
                    notes.append(
                        f"⚠️ Couldn't extract text from `{name}` — is it a scanned/image "
                        "PDF? Paste the `.tex` or text directly and I'll use it."
                    )
                    continue
            elif ext in TEXT_EXTS or ctype.startswith("text/"):
                text = data.decode("utf-8", errors="replace")
            elif ext in IMAGE_EXTS or ctype.startswith("image/"):
                await self._save_binary_artifact(message, name, data)
                notes.append(f"📎 `{name}` saved to the project.")
                continue
            else:
                notes.append(
                    f"ℹ️ `{name}` ({ctype or ext or 'unknown type'}) isn't a readable "
                    "doc — I can read PDF, LaTeX, plain-text, and image files."
                )
                continue

            saved_ref = await self._save_upload_artifact(message, name, text)
            cap = settings.attachment_max_chars
            if len(text) > cap:
                more = len(text) - cap
                tail = (
                    f"\n…[truncated {more} chars; full text saved to `{saved_ref}`]"
                    if saved_ref else f"\n…[truncated {more} chars]"
                )
                inline = text[:cap] + tail
            else:
                inline = text
            head = (
                f"[Attached file: {name} — full text saved to `{saved_ref}`]"
                if saved_ref else f"[Attached file: {name}]"
            )
            blocks.append(f"{head}\n{inline}\n[end of {name}]")
        return "\n\n".join(blocks), "\n".join(notes)

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        """Best-effort text from a PDF's bytes. Returns '' on failure/scanned PDF."""
        import io

        try:
            from pypdf import PdfReader
        except Exception:  # noqa: BLE001 — dependency missing
            logger.error("pypdf is not installed; cannot read PDF attachments.")
            return ""
        try:
            reader = PdfReader(io.BytesIO(data))
            parts: list[str] = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001 — skip an unreadable page
                    continue
            return "\n".join(parts).strip()
        except Exception:  # noqa: BLE001 — corrupt/encrypted PDF
            logger.exception("PDF extraction failed")
            return ""

    @staticmethod
    def _safe_upload_path(folder, original: str, forced_suffix: str | None = None):
        """Return a path under folder that is safe and unique.

        Strips directory components from the original filename (path traversal
        guard) and appends a counter if the target already exists.
        """
        from pathlib import Path
        safe_name = Path(original).name or "upload"
        stem = Path(safe_name).stem or "upload"
        suffix = forced_suffix if forced_suffix is not None else (Path(safe_name).suffix or ".bin")
        candidate = folder / f"{stem}{suffix}"
        i = 2
        while candidate.exists():
            candidate = folder / f"{stem}-{i}{suffix}"
            i += 1
        return candidate

    async def _save_upload_artifact(
        self, message: discord.Message, name: str, text: str
    ) -> str:
        """Save full attachment text as a project 'upload' artifact.

        Returns the outputs-relative path (usable with `!getfile`) or '' if it
        couldn't be saved.
        """
        if self.projects is None:
            return ""
        from pathlib import Path

        try:
            project_key = self._project_channel_id(self._ch(message))
            project = await self.projects.ensure(
                project_key,
                name=self._project_name(self._ch(message), message.author),
            )
            self._ensured_projects.add(project_key)
            folder = self.projects.kind_dir(project["slug"], "uploads")
            out = self._safe_upload_path(folder, name, ".txt")
            out.write_text(text)
            rel = str(out.resolve().relative_to(Path(settings.output_dir).resolve()))
            if project.get("id"):
                await self.projects.add_artifact(
                    project["id"], "upload", out.stem, rel, {"original": name},
                )
            return rel
        except Exception:  # noqa: BLE001 — saving must not drop the message
            logger.exception("Failed to save upload artifact for %s", name)
            return ""

    async def _save_binary_artifact(
        self, message: discord.Message, name: str, data: bytes
    ) -> None:
        """Save binary attachment (image, etc.) directly to the project uploads folder."""
        if self.projects is None:
            return
        from pathlib import Path

        try:
            project_key = self._project_channel_id(self._ch(message))
            project = await self.projects.ensure(
                project_key,
                name=self._project_name(self._ch(message), message.author),
            )
            self._ensured_projects.add(project_key)
            folder = self.projects.kind_dir(project["slug"], "uploads")
            out = self._safe_upload_path(folder, name)
            out.write_bytes(data)
            rel = str(out.resolve().relative_to(Path(settings.output_dir).resolve()))
            if project.get("id"):
                await self.projects.add_artifact(
                    project["id"], "upload", out.stem, rel, {"original": name},
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save binary artifact for %s", name)

    async def _handle_chat(self, message, content, config, thread_id) -> None:
        try:
            # Serialize per channel so concurrent messages in the same thread
            # don't race on shared checkpoint/memory state.
            async with self._channel_locks.get(thread_id):
                async with self._ch(message).typing():
                    result = await self.graph.ainvoke(
                        {"messages": [("user", content)]}, config=config
                    )
            reply = _flatten(result["messages"][-1].content)
            cumulative = result.get("cumulative_tokens", 0)
            summary = result.get("summary") or ""
        except Exception:  # noqa: BLE001
            logger.exception("Error handling message")
            await self._ch(message).send(
                "Something went wrong while I was thinking. Check the logs."
            )
            return

        for chunk in _chunk(reply):
            await self._ch(message).send(chunk)

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
            await self._ch(message).send(HELP_TEXT)
        elif cmd in {"checkpoint", "summarize"}:
            await self._checkpoint(message, config)
        elif cmd == "remember":
            arg = arg.strip()
            if not arg:
                await self._ch(message).send("Usage: `!remember <text>`")
            elif self.memory is not None:
                await self.memory.procedural.add(arg, kind="preference")
                await self._ch(message).send("Noted — I'll remember that.")
            else:
                await self._ch(message).send("Memory isn't configured, so I can't store that.")
        elif cmd in {"runs", "approve", "cancel"}:
            await self._handle_experiment_command(message, cmd, arg.strip())
        elif cmd == "gpu":
            await self._attach_gpu(message, arg.strip())
        elif cmd == "getfile":
            await self._send_file(message, arg.strip())
        elif cmd == "ideate":
            await self._ideate(message, arg.strip(), config)
        elif cmd in {"tasks", "task", "trace"}:
            await self._handle_task_command(message, cmd, arg.strip())
        elif cmd == "feedback":
            await self._handle_feedback(message, arg.strip())
        elif cmd == "project":
            await self._handle_project_command(message, arg.strip())
        elif cmd in {"mute", "stop", "quiet"}:
            self._engaged.get(config["configurable"]["thread_id"], set()).discard(message.author.id)
            await self._ch(message).send(
                "Muted here — I'll stay quiet until you @-mention me or reply to me again."
            )
        else:
            await self._ch(message).send(f"Unknown command `!{cmd}`.\n\n{HELP_TEXT}")

    async def _handle_project_command(self, message, arg) -> None:
        """`!project` shows this chat's project; `!project name <x>` renames it."""
        if self.projects is None:
            await self._ch(message).send("Projects aren't configured (needs the DB).")
            return
        channel_id = self._project_channel_id(self._ch(message))
        if arg.lower().startswith("name "):
            new_name = arg[5:].strip()
            proj = await self.projects.rename(channel_id, new_name)
            await self._ch(message).send(f"Project renamed to **{proj['name']}** (`{proj['slug']}`).")
            return
        proj = await self.projects.ensure(channel_id)
        arts = await self.projects.list_artifacts(proj["id"]) if proj.get("id") else []
        by_kind: dict[str, int] = {}
        for a in arts:
            by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in by_kind.items()) or "no artifacts yet"
        await self._ch(message).send(
            f"**Project:** {proj['name']} (`{proj['slug']}`)\n"
            f"Artifacts — {summary}\n"
            f"Folder: `outputs/projects/{proj['slug']}/`  ·  rename with `!project name <x>`"
        )

    async def _handle_task_command(self, message, cmd, arg) -> None:
        if self.tasks is None:
            await self._ch(message).send("The task dashboard isn't configured (needs the DB).")
            return

        if cmd == "tasks":
            rows = await self.tasks.list_recent(limit=15)
            if not rows:
                await self._ch(message).send("No tasks yet.")
                return
            lines = [
                f"#{r['id']} [{r['status']}] {r['agent']} — {r['input'][:60]}"
                for r in rows
            ]
            await self._ch(message).send("**Tasks (recent)**\n" + "\n".join(lines))
            return

        if not arg.isdigit():
            await self._ch(message).send(f"Usage: `!{cmd} <task_id>`")
            return
        task = await self.tasks.get(int(arg))
        if task is None:
            await self._ch(message).send(f"No task #{arg}.")
            return

        if cmd == "task":
            body = (
                f"**Task #{task['id']}** — {task['agent']} [{task['status']}]\n"
                f"Input: {task['input'][:300]}\n\n"
                f"{(task.get('result') or task.get('error') or '(no result yet)')}"
            )
            for chunk in _chunk(body):
                await self._ch(message).send(chunk)
            return

        # cmd == "trace": export the full trace as a JSON file via outputs/
        import json
        from pathlib import Path

        traces_dir = Path(settings.output_dir) / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        out = traces_dir / f"task_{task['id']}.json"
        out.write_text(json.dumps(task.get("trace") or [], indent=2, default=str))
        await self._ch(message).send(
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
            await self._ch(message).send("The task dashboard isn't configured (needs the DB).")
            return
        parts = arg.split(maxsplit=2)
        if len(parts) < 2 or not parts[0].isdigit() or parts[1].lower() not in {"good", "bad"}:
            await self._ch(message).send("Usage: `!feedback <task_id> <good|bad> [note]`")
            return
        task_id, quality = int(parts[0]), parts[1].lower()
        note = parts[2].strip() if len(parts) > 2 else ""

        task = await self.tasks.get(task_id)
        if task is None:
            await self._ch(message).send(f"No task #{task_id}.")
            return

        ok = await self.tasks.set_feedback(task_id, quality, note or None)
        agent = task.get("agent") or "subagent"
        banked = False
        if ok and self.memory is not None:
            try:
                project_key = self._project_channel_id(self._ch(message))
                project = (
                    await self.projects.get_by_channel(project_key)
                    if self.projects is not None else None
                )
                verdict = "was good — reuse this approach" if quality == "good" else "needs correction"
                lesson = (
                    f"[user feedback] A '{agent}' result for "
                    f"\"{(task.get('input') or '')[:160]}\" {verdict}."
                    + (f" Specifically: {note}" if note else "")
                )
                await self.memory.record_lesson(
                    lesson, kind=agent, channel_id=project_key,
                    status=quality, project=(project["slug"] if project else None),
                )
                banked = True
            except Exception:  # noqa: BLE001 — label is saved; banking a lesson is best-effort
                logger.exception("Failed to bank feedback lesson for task #%s", task_id)

        if ok:
            tail = " and banked a lesson for next time." if banked else "."
            await self._ch(message).send(f"Logged **{quality}** feedback on task #{task_id}{tail}")
        else:
            await self._ch(message).send(f"Couldn't update task #{task_id}.")

    async def _ideate(self, message, arg, config) -> None:
        """Drive the consortium.

        `!ideate <topic>` runs round 1 (independent + debated proposals, then
        scoring) and posts the top 5; reply with numbers (or `!ideate pick 2,4`) to
        develop them; `!ideate again <notes>` runs another polish+vote round;
        `!ideate done` finalizes; `!ideate cancel` drops the session.
        """
        if self.consortium is None:
            await self._ch(message).send(
                "The consortium isn't configured (needs OPENROUTER_API_KEY)."
            )
            return

        thread_id = config["configurable"]["thread_id"]
        session = self.consortium_sessions.get(thread_id)
        sub, _, rest = arg.strip().partition(" ")
        sub, rest = sub.lower(), rest.strip()

        if sub in {"done", "finish", "finalize"}:
            if session is not None:
                await self._finalize_consortium(message, session, config)
            else:
                await self._ch(message).send("No active session. Start one with `!ideate <topic>`.")
            return
        if sub in {"cancel", "stop", "abort"}:
            if session is not None:
                self.consortium_sessions.pop(thread_id, None)
                await self._ch(message).send("Consortium session cancelled.")
            else:
                await self._ch(message).send("No active consortium session to cancel.")
            return
        if sub in {"again", "refine", "another"}:
            if session is not None and session.phase == "polished":
                await self._consortium_polish(message, session, picks=None, comments=rest)
            else:
                await self._ch(message).send("Nothing to refine yet — pick ideas from a round-1 top-5 first.")
            return
        if sub == "pick":
            picks = _parse_picks(rest)
            if session is not None and session.phase == "scored" and picks:
                await self._consortium_polish(message, session, picks=picks)
            else:
                await self._ch(message).send("`!ideate pick <numbers>` works on a round-1 top-5.")
            return

        # An active session: route by phase.
        if session is not None and not session.finalized:
            picks = _parse_picks(arg)
            if session.phase == "scored" and picks:
                await self._consortium_polish(message, session, picks=picks)
            else:
                await self._ch(message).send(self._ideate_hint(session))
            return

        # No active session: start a fresh one on the given topic.
        if not arg.strip():
            await self._ch(message).send("Usage: `!ideate <topic>`")
            return
        session = self.consortium.new_session(arg.strip())
        self.consortium_sessions[thread_id] = session
        await self._ch(message).send(
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
            await self._ch(message).send("The panel is still deliberating — one moment…")
            return
        session.busy = True
        try:
            async with self._ch(message).typing():
                await session.run_round1()
        except Exception:  # noqa: BLE001
            logger.exception("Consortium round 1 failed")
            await self._ch(message).send("The consortium hit an error this round. Check the logs.")
            return
        finally:
            session.busy = False

        for chunk in _chunk("**Round 1 — top 5 (scored 0–10)**\n\n" + session.render_top()):
            await self._ch(message).send(chunk)
        await self._ch(message).send(
            "Reply with the numbers to develop (e.g. `2,4`), or `!ideate done` to finalize."
        )

    async def _consortium_polish(self, message, session, picks: list[int] | None, comments: str = "") -> None:
        if session.busy:
            await self._ch(message).send("The panel is still deliberating — one moment…")
            return
        session.busy = True
        try:
            async with self._ch(message).typing():
                await session.select_and_polish(picks=picks, comments=comments)
        except Exception:  # noqa: BLE001
            logger.exception("Consortium polish round failed")
            await self._ch(message).send("The panel hit an error polishing. Check the logs.")
            return
        finally:
            session.busy = False

        for chunk in _chunk(f"**Round {session.round_no} — polished & voted**\n\n" + session.render_top()):
            await self._ch(message).send(chunk)
        await self._ch(message).send(
            "Reply `!ideate again <notes>` for another round, or `!ideate done` to finalize."
        )

    async def _finalize_consortium(self, message, session, config) -> None:
        if session.busy:
            await self._ch(message).send("The panel is still deliberating — one moment…")
            return
        session.busy = True
        try:
            async with self._ch(message).typing():
                result = await session.finalize()
        except Exception:  # noqa: BLE001
            logger.exception("Consortium finalize failed")
            await self._ch(message).send("The chair hit an error finalizing. Check the logs.")
            session.busy = False
            return
        session.busy = False
        thread_id = config["configurable"]["thread_id"]
        self.consortium_sessions.pop(thread_id, None)

        # Save the chosen proposal(s) into the project's council folder.
        council_rel = ""
        project_key = self._project_channel_id(self._ch(message))
        if self.projects is not None:
            from ..projects import save_council_proposal

            project = await self.projects.ensure(project_key)
            council_rel = await save_council_proposal(
                self.projects, project, session.topic, result["ideas"]
            )

        # Capture the session (incl. the debate) to memory so future ideation's
        # debate track recalls it (scoped to the project, not the individual thread).
        from ..consortium import capture_council

        await capture_council(
            self.memory, project_key, session.topic,
            result["ideas"], result["rel_path"], rounds=result["rounds"],
        )

        for chunk in _chunk(result["ideas"]):
            await self._ch(message).send(chunk)
        tail = (
            f"Done after {result['rounds']} round(s). "
            f"Full transcript (ideas + debates): `!getfile {result['rel_path']}`\n"
        )
        if council_rel:
            tail += f"Saved for methodology: `!getfile {council_rel}`\n"
        tail += "Want me to hand a chosen idea to the methodology writer? Just say the word."
        await self._ch(message).send(tail)

    async def _send_file(self, message, relpath: str) -> None:
        if not relpath:
            await self._ch(message).send("Usage: `!getfile <path>` (relative to outputs/)")
            return
        from pathlib import Path

        base = Path(settings.output_dir).resolve()
        target = (base / relpath).resolve()
        # Confine to the outputs directory; reject traversal / absolute escapes.
        if target != base and base not in target.parents:
            await self._ch(message).send("Path is outside the outputs directory.")
            return
        if not target.is_file():
            await self._ch(message).send(f"No such file: `{relpath}`")
            return
        # Discord's default non-boosted upload limit is 25 MB; stay well under.
        if target.stat().st_size > 8 * 1024 * 1024:
            await self._ch(message).send("File is too large to upload (>8 MB).")
            return
        await self._ch(message).send(file=discord.File(str(target)))

    async def _attach_gpu(self, message, arg) -> None:
        """Attach a fresh GPU box (`!gpu user@ip`) and provision it in the background."""
        if self.experiments is None:
            await self._ch(message).send(
                "Experiment runner isn't configured (needs a database)."
            )
            return
        if not arg:
            await self._ch(message).send("Usage: `!gpu <user@ip>` (bare Ubuntu, your SSH key).")
            return

        msg = await self.experiments.set_compute(arg)
        await self._ch(message).send(
            f"{msg}\nProvisioning (Docker + NVIDIA toolkit + MLflow) — I'll report back."
        )
        notify_id = str(self._ch(message).id)

        async def _provision_and_report() -> None:
            try:
                report = await self.experiments.provision()
                survey = await self.experiments.survey()
            except Exception as exc:  # noqa: BLE001
                await self._notify_channel(notify_id, f"GPU provisioning failed: {exc}")
                return
            tail = "\n".join(report.splitlines()[-15:])
            await self._notify_channel(
                notify_id,
                f"✅ GPU box ready.\n```\n{survey}\n```\n_provision log (tail):_\n```\n{tail}\n```",
            )

        self._spawn_background(_provision_and_report(), "gpu.provision")

    async def _handle_experiment_command(self, message, cmd, arg) -> None:
        if self.experiments is None:
            await self._ch(message).send("Experiment runner isn't configured.")
            return

        if cmd == "runs":
            rows = await self.memory.episodic.list_experiments(limit=15)
            if not rows:
                await self._ch(message).send("No experiments yet.")
                return
            lines = [f"#{r['id']} [{r['status']}] {r['title']}" for r in rows]
            await self._ch(message).send("**Experiments**\n" + "\n".join(lines))
            return

        if not arg.isdigit():
            await self._ch(message).send(f"Usage: `!{cmd} <experiment_id>`")
            return
        exp_id = int(arg)
        async with self._ch(message).typing():
            if cmd == "approve":
                result = await self.experiments.approve_and_launch(exp_id)
            else:  # cancel
                result = await self.experiments.cancel(exp_id)
        await self._ch(message).send(result)

    async def _checkpoint(self, message, config) -> None:
        from langchain_core.messages import RemoveMessage

        thread_id = config["configurable"]["thread_id"]

        # Hold the channel lock across read-summarize-reset so it can't
        # interleave with a concurrent chat turn on the same thread.
        async with self._channel_locks.get(thread_id):
            snapshot = await self.graph.aget_state(config)
            messages = snapshot.values.get("messages", []) if snapshot else []
            if not messages:
                await self._ch(message).send("Nothing to checkpoint yet.")
                return

            semantic_saved = False
            async with self._ch(message).typing():
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

        await self._ch(message).send(
            checkpoint_result_message(self.memory is not None, semantic_saved)
        )
