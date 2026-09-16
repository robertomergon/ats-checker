"""Orquestación de alto nivel: de bytes de fichero a resultado completo.

Envuelve al grafo para que la UI (o un script) no tenga que conocer el estado
interno: extrae el documento, construye el estado inicial, ejecuta el grafo con
progreso por nodo y reduce las actualizaciones al estado final.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config, graph as graph_mod, textio

#: Claves del estado cuyas actualizaciones se acumulan (tienen reducer en el grafo).
_ACCUMULATED = ("warnings", "usage")

ProgressCallback = Optional[Callable[[str], None]]


def extract_document(data: bytes, filename: str) -> Dict[str, Any]:
    """Extrae texto y señales de maquetación de un CV."""
    return textio.extract(data, filename)


def initial_state(
    document: Dict[str, Any],
    job_description: str,
    *,
    use_llm: bool = True,
    effort: str = config.DEFAULT_EFFORT,
    weights: Optional[Dict[str, float]] = None,
    job_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "resume_filename": document.get("filename", ""),
        "document": document,
        "job_description": job_description,
        "use_llm": use_llm,
        "effort": effort,
        "weights": dict(weights or config.DEFAULT_WEIGHTS),
        "warnings": [],
        "usage": [],
    }
    if job_profile:
        state["job_profile"] = job_profile
    return state


def run(compiled: Any, state: Dict[str, Any], on_node: ProgressCallback = None) -> Dict[str, Any]:
    """Ejecuta el grafo en streaming y devuelve el estado final reducido.

    Se reducen las actualizaciones a mano (en vez de pedir el estado completo)
    para poder notificar el avance nodo a nodo a la UI sin depender del modo de
    streaming de la versión de LangGraph instalada.
    """
    final: Dict[str, Any] = dict(state)
    for chunk in compiled.stream(state, stream_mode="updates"):
        for node_name, update in (chunk or {}).items():
            if on_node:
                on_node(node_name)
            if not isinstance(update, dict):
                continue
            for key, value in update.items():
                if key in _ACCUMULATED:
                    final[key] = list(final.get(key, [])) + list(value or [])
                else:
                    final[key] = value
    return final


def analyze(
    data: bytes,
    filename: str,
    job_description: str,
    *,
    chain: Optional[Any] = None,
    use_llm: bool = True,
    effort: str = config.DEFAULT_EFFORT,
    weights: Optional[Dict[str, float]] = None,
    on_node: ProgressCallback = None,
    compiled: Optional[Any] = None,
) -> Dict[str, Any]:
    """Analiza un CV contra una oferta y devuelve el estado final del grafo."""
    compiled = compiled or graph_mod.build_graph(chain)
    document = extract_document(data, filename)
    state = initial_state(
        document, job_description, use_llm=use_llm, effort=effort, weights=weights
    )
    return run(compiled, state, on_node=on_node)


def compare(
    files: Sequence[Tuple[str, bytes]],
    job_description: str,
    *,
    chain: Optional[Any] = None,
    use_llm: bool = True,
    effort: str = config.DEFAULT_EFFORT,
    weights: Optional[Dict[str, float]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Analiza varios CVs contra la misma oferta y los devuelve ordenados por score.

    La oferta se estructura una única vez (en el primer CV) y su perfil se
    reutiliza: el coste por candidato adicional baja y, sobre todo, todos se
    puntúan contra exactamente los mismos requisitos.
    """
    compiled = graph_mod.build_graph(chain)
    results: List[Dict[str, Any]] = []
    shared_job_profile: Optional[Dict[str, Any]] = None

    for index, (filename, data) in enumerate(files, start=1):
        if on_progress:
            on_progress(index, len(files), filename)
        document = extract_document(data, filename)
        state = initial_state(
            document,
            job_description,
            use_llm=use_llm,
            effort=effort,
            weights=weights,
            job_profile=shared_job_profile,
        )
        final = run(compiled, state)
        shared_job_profile = shared_job_profile or final.get("job_profile")
        results.append(final)

    return sorted(results, key=lambda s: (s.get("scores", {}) or {}).get("total", 0), reverse=True)


def summary_row(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fila compacta de un resultado, para la tabla comparativa y el CSV."""
    scores = state.get("scores", {}) or {}
    dimensions = scores.get("dimensions", {})
    matches = state.get("matches", []) or []
    profile = state.get("resume_profile", {}) or {}

    row: Dict[str, Any] = {
        "CV": state.get("resume_filename", ""),
        "Candidato": profile.get("candidate_name", "") or "—",
        "Score": scores.get("total", 0),
        "Banda": scores.get("band", ""),
    }
    for name, label in config.DIMENSION_LABELS.items():
        dim = dimensions.get(name, {})
        row[label] = round(dim.get("score", 0), 1) if dim.get("applicable") else None
    row["Requisitos excluyentes sin cubrir"] = len(scores.get("missing_required", []))
    row["Requisitos cubiertos"] = sum(1 for m in matches if m["status"] == "match")
    row["Años experiencia"] = profile.get("total_years_experience", 0)
    row["Bloqueantes de formato"] = len(scores.get("blockers", []))
    usage = total_usage([state])
    row["Proveedor"] = usage["providers"] or "—"
    row["Coste USD"] = usage["cost_usd"] if usage["cost_usd"] is not None else None
    return row


def total_usage(states: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Agregado de tokens, proveedores y coste estimado de una o varias ejecuciones.

    El coste es `None` cuando algún proveedor no tiene tarifa conocida ni
    configurada: mejor no dar una cifra que dar una inventada.
    """
    records = [u for state in states for u in (state.get("usage", []) or [])]
    costs = [r.get("cost_usd") for r in records]
    known = [c for c in costs if c is not None]
    used = []
    for record in records:
        name = record.get("provider", "")
        if name and name not in used:
            used.append(name)

    return {
        "calls": len(records),
        "input_tokens": sum(r.get("input_tokens", 0) for r in records),
        "output_tokens": sum(r.get("output_tokens", 0) for r in records),
        "cost_usd": round(sum(known), 4) if len(known) == len(costs) else None,
        "partial_cost_usd": round(sum(known), 4),
        "providers": ", ".join(used),
        "models": ", ".join(
            dict.fromkeys(r.get("model", "") for r in records if r.get("model"))
        ),
        "fallbacks_used": sorted(
            {p for r in records for p in (r.get("fallback_from") or [])}
        ),
    }
