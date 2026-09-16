"""ATS Checker: análisis de compatibilidad CV ↔ oferta con LangGraph.

La capa LLM es multi-proveedor (Claude, OpenAI, Gemini o compatible) con cadena
de fallback; ver `ats.providers`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    audit,
    config,
    graph,
    llm,
    matching,
    pipeline,
    prompts,
    providers,
    scoring,
    schemas,
    textio,
)

__all__ = [
    "audit",
    "config",
    "graph",
    "llm",
    "matching",
    "pipeline",
    "prompts",
    "providers",
    "scoring",
    "schemas",
    "textio",
]
