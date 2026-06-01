"""Memory subsystem.

Three memory types (per the semantic/episodic/procedural framework):

- semantic.py   -> facts (mem0 over pgvector, with citation/provenance)
- episodic.py   -> experiences (Postgres lab notebook + experiment registry)
- procedural.py -> instructions (learned preferences/procedures, prepended)

Plus operational machinery: tokens.py (counting + nudge boundaries),
summarize.py (rolling summarization), maintenance.py (idle archival +
consolidation/reflection).
"""
