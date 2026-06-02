"""Tests for the literature-review drafting helpers (pure logic + saving)."""

from __future__ import annotations

from research_agent.writing.lit_review import (
    LiteratureReviewer,
    count_bib_entries,
    extract_code_block,
    parse_review,
    slugify,
)

SAMPLE = """Here is the review.

```latex
\\section{Related Work}
Speculative decoding~\\cite{chen2023} accelerates inference.
```

```bibtex
@article{chen2023, title={Speculative Sampling}, year={2023}}
@inproceedings{xia2025, title={Tutorial}, year={2025}}
```

Covered 2 papers.
"""


def test_slugify():
    assert slugify("Speculative Decoding for LLMs!") == "speculative-decoding-for-llms"
    assert slugify("") == "draft"  # shared fallback across all LaTeX writers


def test_extract_code_block():
    assert "\\section{Related Work}" in extract_code_block(SAMPLE, "latex")
    assert "@article{chen2023" in extract_code_block(SAMPLE, "bibtex")
    assert extract_code_block(SAMPLE, "python") == ""


def test_count_bib_entries():
    assert count_bib_entries(extract_code_block(SAMPLE, "bibtex")) == 2
    assert count_bib_entries("") == 0


def test_parse_review_splits_blocks():
    latex, bibtex, n_refs = parse_review(SAMPLE)
    assert latex.startswith("\\section{Related Work}")
    assert "@inproceedings{xia2025" in bibtex
    assert n_refs == 2


def test_parse_review_fallback_to_whole_text():
    latex, bibtex, n_refs = parse_review("just some latex with no fences")
    assert latex == "just some latex with no fences"
    assert bibtex == ""
    assert n_refs == 0


def test_save_review_writes_files(tmp_path):
    reviewer = LiteratureReviewer(llm=None, tools=[], output_dir=str(tmp_path))
    tex_path, bib_path = reviewer.save_review("mytopic", "\\section{X}", "@a{k,}")
    assert tex_path.endswith("lit_reviews/mytopic.tex")
    assert (tmp_path / "lit_reviews" / "mytopic.tex").read_text() == "\\section{X}"
    assert (tmp_path / "lit_reviews" / "mytopic.bib").read_text() == "@a{k,}"


def test_save_review_skips_empty_bibtex(tmp_path):
    reviewer = LiteratureReviewer(llm=None, tools=[], output_dir=str(tmp_path))
    tex_path, bib_path = reviewer.save_review("t", "\\section{X}", "")
    assert bib_path == ""
    assert not (tmp_path / "lit_reviews" / "t.bib").exists()
