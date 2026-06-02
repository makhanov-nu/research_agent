"""Tests for the shared LaTeX writer helpers and the writer bundle."""

from __future__ import annotations

from research_agent.writing import (
    MethodologyWriter,
    PaperWriter,
    Writers,
    build_writers,
)
from research_agent.writing.latex import (
    flatten_content,
    parse_latex_artifact,
    slugify,
    timestamped,
    write_tex_bib,
)

SAMPLE = """Here is the section.

```latex
\\section{Methodology}
We evaluate on standard benchmarks~\\cite{a2024}.
```

```bibtex
@article{a2024, title={A}, year={2024}}
@inproceedings{b2025, title={B}, year={2025}}
```

Key decision: two-stage design.
"""


def test_parse_latex_artifact_splits_blocks():
    latex, bibtex, n_refs = parse_latex_artifact(SAMPLE)
    assert latex.startswith("\\section{Methodology}")
    assert "@inproceedings{b2025" in bibtex
    assert n_refs == 2


def test_parse_latex_artifact_fallback():
    latex, bibtex, n_refs = parse_latex_artifact("plain text, no fences")
    assert latex == "plain text, no fences"
    assert bibtex == "" and n_refs == 0


def test_slugify_default():
    assert slugify("A Novel Method!") == "a-novel-method"
    assert slugify("") == "draft"


def test_flatten_content_handles_blocks():
    assert flatten_content([{"text": "a"}, {"text": "b"}]) == "ab"
    assert flatten_content("plain") == "plain"


def test_timestamped_appends_suffix():
    out = timestamped("mybase")
    assert out.startswith("mybase-") and len(out) > len("mybase-")


def test_write_tex_bib(tmp_path):
    tex, bib = write_tex_bib(tmp_path / "methodology", "m1", "\\section{M}", "@a{k,}")
    assert tex.endswith("methodology/m1.tex")
    assert (tmp_path / "methodology" / "m1.tex").read_text() == "\\section{M}"
    assert (tmp_path / "methodology" / "m1.bib").read_text() == "@a{k,}"


def test_write_tex_bib_skips_empty_bibtex(tmp_path):
    tex, bib = write_tex_bib(tmp_path / "papers", "p1", "\\section{P}", "")
    assert bib == ""
    assert not (tmp_path / "papers" / "p1.bib").exists()


def test_writers_use_distinct_subdirs(tmp_path):
    m = MethodologyWriter(llm=None, tools=[], output_dir=str(tmp_path))
    p = PaperWriter(llm=None, tools=[], output_dir=str(tmp_path))
    assert m.dir.name == "methodology"
    assert p.dir.name == "papers"


def test_build_writers_bundle(tmp_path):
    writers = build_writers(llm=None, tools=[], output_dir=str(tmp_path))
    assert isinstance(writers, Writers)
    assert writers.methodologist.subdir == "methodology"
    assert writers.paper_writer.subdir == "papers"
    assert writers.reviewer.subdir == "lit_reviews"
