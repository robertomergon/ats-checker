"""Grafo LangGraph del análisis ATS.

    ingest_resume ─┬─> profile_job ─────┐
                   ├─> profile_resume ──┼─> match_requirements ─> reconcile_semantics
                   └─> audit_format ────┘                                │
                                                                         v
                                       END <── advise <── score_candidate

Los tres nodos de la primera oleada corren en paralelo y escriben claves
distintas del estado; `warnings` y `usage` llevan reducer (`operator.add`)
precisamente por eso. Todo nodo con LLM degrada a heurística determinista si la
llamada falla, de modo que el grafo siempre llega a `END` con un score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langgraph.graph import END, START, StateGraph

from . import audit, llm, matching, prompts, scoring
from .schemas import (
    ATSState,
    JobProfile,
    RecommendationSet,
    ResumeProfile,
    SemanticMatchResult,
)

#: Tope de requisitos que se envían al nodo semántico (control de coste).
MAX_SEMANTIC_ITEMS = 40

NODE_DESCRIPTIONS: Dict[str, str] = {
    "ingest_resume": "Valida el texto extraído y registra avisos de extracción",
    "profile_job": "El LLM estructura la oferta en requisitos con exigencia y alias",
    "profile_resume": "El LLM normaliza el CV como lo haría el parser de un ATS",
    "audit_format": "Auditoría determinista de legibilidad e integridad del documento",
    "match_requirements": "Match literal de requisitos y keywords sobre el texto del CV",
    "reconcile_semantics": "El LLM revisa solo los requisitos no encontrados literalmente",
    "score_candidate": "Score ponderado por dimensiones, determinista y reproducible",
    "advise": "El LLM redacta el veredicto y las recomendaciones priorizadas",
}


def build_graph(chain: Optional[Any] = None):
    """Compila el grafo.

    `chain` es una `ProviderChain` (Claude, OpenAI, Gemini o compatible). Sin
    cadena, o si está vacía, los nodos LLM usan su versión heurística.
    """

    def ingest_resume(state: ATSState) -> Dict[str, Any]:
        document = state.get("document", {}) or {}
        text = document.get("text", "") or ""
        warnings: List[str] = list(document.get("notes", []))
        if not text.strip():
            warnings.append(
                "No hay texto que analizar: el score de contenido será 0 por construcción."
            )
        elif document.get("words", 0) < 80:
            warnings.append("El CV extraído tiene muy poco texto; revisa si la extracción falló.")
        return {"resume_text": text, "warnings": warnings}

    def profile_job(state: ATSState) -> Dict[str, Any]:
        job_description = state.get("job_description", "")
        # En comparación de varios CVs la oferta se analiza una sola vez y se
        # inyecta en el estado inicial: aquí solo se reutiliza.
        if state.get("job_profile"):
            return {}
        if not _llm_enabled(state, chain):
            return {"job_profile": matching.heuristic_job_profile(job_description)}
        try:
            result = chain.structured(
                system=prompts.JOB_SYSTEM,
                user=prompts.job_message(job_description),
                output_format=JobProfile,
                effort=state.get("effort", "medium"),
                label="profile_job",
            )
        except llm.LLMError as exc:
            return {
                "job_profile": matching.heuristic_job_profile(job_description),
                "warnings": [f"Oferta analizada sin LLM ({exc})."],
            }
        profile = result.parsed.model_dump()
        profile["_source"] = "llm"
        return {"job_profile": profile, "usage": [result.usage]}

    def profile_resume(state: ATSState) -> Dict[str, Any]:
        text = (state.get("document", {}) or {}).get("text", "") or ""
        if not text.strip():
            return {"resume_profile": matching.heuristic_resume_profile("")}
        if not _llm_enabled(state, chain):
            return {"resume_profile": matching.heuristic_resume_profile(text)}
        try:
            result = chain.structured(
                system=prompts.RESUME_SYSTEM,
                user=prompts.resume_message(text),
                output_format=ResumeProfile,
                effort=state.get("effort", "medium"),
                label="profile_resume",
            )
        except llm.LLMError as exc:
            return {
                "resume_profile": matching.heuristic_resume_profile(text),
                "warnings": [f"CV estructurado sin LLM ({exc})."],
            }
        profile = result.parsed.model_dump()
        profile["_source"] = "llm"
        return {"resume_profile": profile, "usage": [result.usage]}

    def audit_format(state: ATSState) -> Dict[str, Any]:
        document = state.get("document", {}) or {}
        return {"format_audit": audit.audit_format(document, document.get("text", "") or "")}

    def match_requirements(state: ATSState) -> Dict[str, Any]:
        job_profile = state.get("job_profile", {}) or {}
        text = state.get("resume_text", "") or ""
        matches = matching.match_requirements(job_profile.get("requirements", []), text)
        coverage = matching.keyword_coverage(job_profile.get("ats_keywords", []), text)
        warnings: List[str] = []
        if not matches:
            warnings.append(
                "No se extrajo ningún requisito de la oferta: revisa que el texto pegado "
                "sea la descripción completa del puesto."
            )
        return {"matches": matches, "keyword_coverage": coverage, "warnings": warnings}

    def reconcile_semantics(state: ATSState) -> Dict[str, Any]:
        matches = list(state.get("matches", []) or [])
        pending = [m for m in matches if m["status"] != "match"][:MAX_SEMANTIC_ITEMS]
        if not pending or not _llm_enabled(state, chain):
            return {}
        try:
            result = chain.structured(
                system=prompts.SEMANTIC_SYSTEM,
                user=prompts.semantic_message(
                    state.get("resume_text", ""),
                    pending,
                    (state.get("job_profile", {}) or {}).get("job_title", ""),
                ),
                output_format=SemanticMatchResult,
                effort=state.get("effort", "medium"),
                label="reconcile_semantics",
            )
        except llm.LLMError as exc:
            return {"warnings": [f"Sin revisión semántica de equivalencias ({exc})."]}

        verdicts = {
            matching.normalize(v.requirement): v for v in result.parsed.verdicts if v.requirement
        }
        updated = []
        for match in matches:
            verdict = verdicts.get(matching.normalize(match["name"]))
            # Solo se promociona: un match literal nunca se degrada por el LLM.
            if verdict and match["status"] != "match" and verdict.status != "missing":
                match = dict(
                    match,
                    status=verdict.status,
                    evidence=verdict.evidence or match.get("evidence", ""),
                    source="semantic",
                    reasoning=verdict.reasoning,
                )
            updated.append(match)
        return {"matches": updated, "usage": [result.usage]}

    def score_candidate(state: ATSState) -> Dict[str, Any]:
        return {
            "scores": scoring.compute(
                matches=state.get("matches", []) or [],
                coverage=state.get("keyword_coverage", {}) or {},
                format_audit=state.get("format_audit", {}) or {},
                job_profile=state.get("job_profile", {}) or {},
                resume_profile=state.get("resume_profile", {}) or {},
                weights=state.get("weights"),
            )
        }

    def advise(state: ATSState) -> Dict[str, Any]:
        if not _llm_enabled(state, chain):
            return {"recommendations": _heuristic_advice(state)}
        try:
            result = chain.structured(
                system=prompts.ADVICE_SYSTEM,
                user=prompts.advice_message(
                    state.get("job_profile", {}) or {},
                    state.get("matches", []) or [],
                    state.get("keyword_coverage", {}) or {},
                    state.get("format_audit", {}) or {},
                    state.get("scores", {}) or {},
                    state.get("resume_text", ""),
                ),
                output_format=RecommendationSet,
                effort=state.get("effort", "medium"),
                label="advise",
            )
        except llm.LLMError as exc:
            return {
                "recommendations": _heuristic_advice(state),
                "warnings": [f"Recomendaciones generadas sin LLM ({exc})."],
            }
        payload = result.parsed.model_dump()
        payload["_source"] = "llm"
        return {"recommendations": payload, "usage": [result.usage]}

    graph = StateGraph(ATSState)
    graph.add_node("ingest_resume", ingest_resume)
    graph.add_node("profile_job", profile_job)
    graph.add_node("profile_resume", profile_resume)
    graph.add_node("audit_format", audit_format)
    graph.add_node("match_requirements", match_requirements)
    graph.add_node("reconcile_semantics", reconcile_semantics)
    graph.add_node("score_candidate", score_candidate)
    graph.add_node("advise", advise)

    graph.add_edge(START, "ingest_resume")
    # Primera oleada en paralelo: oferta, CV y formato son independientes.
    for node in ("profile_job", "profile_resume", "audit_format"):
        graph.add_edge("ingest_resume", node)
        graph.add_edge(node, "match_requirements")
    graph.add_edge("match_requirements", "reconcile_semantics")
    graph.add_edge("reconcile_semantics", "score_candidate")
    graph.add_edge("score_candidate", "advise")
    graph.add_edge("advise", END)

    return graph.compile()


# --- Helpers --------------------------------------------------------------


def _llm_enabled(state: ATSState, chain: Optional[Any]) -> bool:
    """Hay LLM si el usuario lo quiere y la cadena tiene algún proveedor usable."""
    return bool(state.get("use_llm", True)) and bool(chain)


def _heuristic_advice(state: ATSState) -> Dict[str, Any]:
    """Recomendaciones deterministas: traduce hallazgos y huecos en acciones."""
    scores = state.get("scores", {}) or {}
    matches = state.get("matches", []) or []
    coverage = state.get("keyword_coverage", {}) or {}
    findings = (state.get("format_audit", {}) or {}).get("findings", [])

    recommendations: List[Dict[str, Any]] = []
    severity_to_priority = {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }
    for finding in findings[:6]:
        recommendations.append(
            {
                "priority": severity_to_priority.get(finding["severity"], "medium"),
                "area": "format",
                "issue": finding["title"],
                "action": finding.get("fix") or finding["detail"],
                "example": "",
            }
        )

    missing_required = [m["name"] for m in matches if m["importance"] == "required" and m["status"] == "missing"]
    if missing_required:
        recommendations.append(
            {
                "priority": "critical",
                "area": "skills",
                "issue": "Requisitos excluyentes sin evidencia en el CV: "
                + ", ".join(missing_required[:6]),
                "action": "Si tienes esa experiencia, nómbrala explícitamente en la sección de "
                "skills y en la viñeta del puesto donde la usaste.",
                "example": "",
            }
        )

    partials = [m["name"] for m in matches if m["status"] == "partial"]
    if partials:
        recommendations.append(
            {
                "priority": "high",
                "area": "keywords",
                "issue": "Requisitos con evidencia parcial: " + ", ".join(partials[:6]),
                "action": "Usa el término exacto del anuncio junto al que ya empleas, para que "
                "el match literal del ATS lo encuentre.",
                "example": "",
            }
        )

    missing_keywords = coverage.get("missing", [])[:12]
    if missing_keywords:
        recommendations.append(
            {
                "priority": "medium",
                "area": "keywords",
                "issue": f"{len(coverage.get('missing', []))} keywords del anuncio no aparecen en el CV.",
                "action": "Incorpora las que correspondan a experiencia real, en la frase donde "
                "la demuestras (no en una lista suelta).",
                "example": ", ".join(missing_keywords),
            }
        )

    return {
        "verdict_summary": (
            f"Score {scores.get('total', 0)}/100 — {scores.get('band', '')}. "
            f"{scores.get('band_detail', '')} Análisis generado sin LLM: "
            "las equivalencias entre tecnologías no se han evaluado semánticamente."
        ),
        "recommendations": recommendations,
        "keywords_to_add": missing_keywords,
        "_source": "heuristic",
    }


def graph_structure(compiled: Any) -> Dict[str, Sequence[Any]]:
    """Nodos y aristas del grafo compilado, para dibujarlo en la UI."""
    try:
        drawable = compiled.get_graph()
        nodes = [n for n in drawable.nodes]
        edges = [(e.source, e.target) for e in drawable.edges]
        return {"nodes": nodes, "edges": edges}
    except Exception:  # noqa: BLE001 - la UI no debe romperse por el diagrama
        return {"nodes": list(NODE_DESCRIPTIONS), "edges": []}
