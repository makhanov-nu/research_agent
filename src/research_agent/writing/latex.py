"""Shared helpers for LaTeX research-artifact writers.

Every writer (literature review, methodology, paper draft) follows the same
shape: run a bounded tool-using loop over the literature tools, expect the model
to return a ```latex block (+ an optional ```bibtex block), then parse and save
both to disk. The parsing/saving helpers here are pure and unit-tested; the
gathering/writing step uses the LLM via the `LatexWriter` base class.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def flatten_content(content) -> str:
    """Collapse a message's content (str or list-of-blocks) into plain text."""
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content)


def slugify(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "draft"


def extract_code_block(text: str, lang: str) -> str:
    """Return the contents of the first ```<lang> fenced block, or ""."""
    m = re.search(rf"```{lang}\b[ \t]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def count_bib_entries(bibtex: str) -> int:
    return len(re.findall(r"^\s*@\w+\s*\{", bibtex, re.MULTILINE))


def parse_latex_artifact(text: str) -> tuple[str, str, int]:
    """Split model output into (latex, bibtex, n_refs).

    Falls back to treating the whole text as LaTeX if no fenced block is present.
    """
    latex = extract_code_block(text, "latex") or text.strip()
    bibtex = extract_code_block(text, "bibtex")
    return latex, bibtex, count_bib_entries(bibtex)


# `\cite{a,b}`, `\citep[p.~5]{c}`, `\textcite{d}`, ... — capture the brace keys.
_CITE_RE = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
# `@article{smith2020,` -> the entry key `smith2020`.
_BIBKEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.MULTILINE)


def cited_keys(latex: str) -> set[str]:
    """Every key referenced by a `\\cite*`-family command in the LaTeX."""
    keys: set[str] = set()
    for m in _CITE_RE.finditer(latex or ""):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def undefined_citations(latex: str, bibtex: str) -> list[str]:
    """`\\cite` keys used in `latex` that have no matching BibTeX entry (sorted).

    Catches the common failure mode where the model writes `\\cite{foo}` but never
    emits the corresponding `@...{foo}` — a dangling reference that would not
    compile — so callers can flag it instead of shipping a broken draft.
    """
    defined = {m.group(1).strip() for m in _BIBKEY_RE.finditer(bibtex or "")}
    return sorted(cited_keys(latex) - defined)


def _serialize_trace(messages) -> list[dict]:
    """Serialize an agent's message history into the dashboard trace shape.

    Imported lazily to keep `writing` importable without the agents package
    (and to avoid an import cycle: agents -> writing.tools -> writing.latex).
    """
    from ..agents.middleware import serialize_messages

    return serialize_messages(messages)


def timestamped(base: str) -> str:
    """Append a UTC timestamp so repeated drafts never clobber each other."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{stamp}"


def write_tex_bib(directory: Path, name: str, latex: str, bibtex: str) -> tuple[str, str]:
    """Write `<name>.tex` (+ `<name>.bib` if bibtex is non-empty); return paths."""
    directory.mkdir(parents=True, exist_ok=True)
    tex_path = directory / f"{name}.tex"
    tex_path.write_text(latex)
    bib_path = directory / f"{name}.bib"
    if bibtex:
        bib_path.write_text(bibtex)
    return str(tex_path), (str(bib_path) if bibtex else "")


class LatexWriter:
    """Base for subagents that research the literature and write a LaTeX artifact.

    Subclasses set `system_prompt` and `subdir` and build the task string in
    `draft(...)`. The shared `_draft(...)` runs the loop, parses the output, saves
    the files, and returns a result dict (`tex_path`, `bib_path`, `n_refs`,
    `latex`). `llm=None`/`tools=[]` is supported for unit-testing the save path.
    """

    system_prompt: str = ""
    subdir: str = "drafts"

    def __init__(self, llm, tools, output_dir: str):
        self.llm = llm
        self.tools = tools
        self.dir = Path(output_dir) / self.subdir

    def save(self, name: str, latex: str, bibtex: str) -> tuple[str, str]:
        return write_tex_bib(self.dir, name, latex, bibtex)

    async def _generate(self, task: str, recursion_limit: int = 40) -> tuple[str, list]:
        """Run the tool-using loop; return (final_text, full_message_history).

        The message history (reasoning + literature tool calls/results) is handed
        back so callers can persist it as the task's dashboard trace.
        """
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(self.llm, self.tools, prompt=self.system_prompt)
        result = await agent.ainvoke(
            {"messages": [("user", task)]}, config={"recursion_limit": recursion_limit}
        )
        messages = result["messages"]
        return flatten_content(messages[-1].content), messages

    async def _draft(
        self, task: str, slug_source: str, save_name: str = "", dirpath=None
    ) -> dict:
        text, messages = await self._generate(task)
        latex, bibtex, n_refs = parse_latex_artifact(text)
        missing = undefined_citations(latex, bibtex)
        name = timestamped(slugify(save_name or slug_source))
        directory = Path(dirpath) if dirpath else self.dir
        tex_path, bib_path = write_tex_bib(directory, name, latex, bibtex)
        logger.info(
            "Wrote %s %s (%d refs%s)", self.subdir, tex_path, n_refs,
            f", {len(missing)} undefined cite(s)" if missing else "",
        )
        return {
            "tex_path": tex_path,
            "bib_path": bib_path,
            "n_refs": n_refs,
            "latex": latex,
            "missing_citations": missing,
            "trace": _serialize_trace(messages),
        }
