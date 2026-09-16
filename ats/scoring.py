"""Cálculo del score ATS a partir de los resultados de los nodos anteriores.

Todas las dimensiones se normalizan a 0-100 y se combinan con los pesos de
`config.DEFAULT_WEIGHTS`. Si una dimensión no aplica (p.ej. la oferta no pide
ninguna certificación), su peso se redistribuye entre las demás en lugar de
contar como un cero injusto.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import config, matching
from .config import IMPORTANCE_WEIGHT, STATUS_CREDIT
from .data import SENIORITY_ORDER

_HARD_CATEGORIES = {"hard_skill", "tool", "domain", "responsibility"}
_EDU_CATEGORIES = {"education", "certification"}


def compute(
    matches: Sequence[Dict[str, Any]],
    coverage: Dict[str, Any],
    format_audit: Dict[str, Any],
    job_profile: Dict[str, Any],
    resume_profile: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Devuelve el score global, el detalle por dimensión y los huecos críticos."""
    weights = dict(weights or config.DEFAULT_WEIGHTS)
    bullet_stats = format_audit.get("bullet_stats", {})

    dimensions: Dict[str, Dict[str, Any]] = {
        "hard_skills": _requirement_dimension(matches, _HARD_CATEGORIES),
        "soft_skills": _requirement_dimension(matches, {"soft_skill"}),
        "education": _requirement_dimension(matches, _EDU_CATEGORIES),
        "keywords": _keyword_dimension(coverage),
        "experience": _experience_dimension(job_profile, resume_profile),
        "format": {
            "score": float(format_audit.get("score", 0.0)),
            "applicable": True,
            "detail": f"{len(format_audit.get('findings', []))} hallazgos de formato",
        },
        "impact": _impact_dimension(bullet_stats, resume_profile),
    }

    applicable = {k: v for k, v in dimensions.items() if v["applicable"]}
    weight_sum = sum(weights.get(k, 0.0) for k in applicable) or 1.0

    total = 0.0
    for name, dim in dimensions.items():
        raw = weights.get(name, 0.0)
        effective = (raw / weight_sum) if dim["applicable"] else 0.0
        dim["weight"] = raw
        dim["effective_weight"] = effective
        dim["label"] = config.DIMENSION_LABELS.get(name, name)
        total += dim["score"] * effective

    total = round(max(0.0, min(100.0, total)), 1)
    label, explanation = config.band_for(total)

    return {
        "total": total,
        "band": label,
        "band_detail": explanation,
        "dimensions": dimensions,
        "missing_required": [
            m["name"] for m in matches if m["importance"] == "required" and m["status"] == "missing"
        ],
        "partial_required": [
            m["name"] for m in matches if m["importance"] == "required" and m["status"] == "partial"
        ],
        "matched": [m["name"] for m in matches if m["status"] == "match"],
        "blockers": _blockers(format_audit),
    }


# --- Dimensiones ----------------------------------------------------------


def _requirement_dimension(
    matches: Sequence[Dict[str, Any]], categories: set
) -> Dict[str, Any]:
    subset = [m for m in matches if m.get("category") in categories]
    if not subset:
        return {"score": 0.0, "applicable": False, "detail": "La oferta no pide nada de este tipo"}

    earned = 0.0
    possible = 0.0
    for m in subset:
        w = IMPORTANCE_WEIGHT.get(m.get("importance", "preferred"), 1.0)
        possible += w
        earned += w * STATUS_CREDIT.get(m.get("status", "missing"), 0.0)

    hits = sum(1 for m in subset if m["status"] == "match")
    return {
        "score": round(100.0 * earned / possible, 1) if possible else 0.0,
        "applicable": True,
        "detail": f"{hits}/{len(subset)} cubiertos (ponderado por exigencia)",
    }


def _keyword_dimension(coverage: Dict[str, Any]) -> Dict[str, Any]:
    present = len(coverage.get("present", []))
    total = present + len(coverage.get("missing", []))
    if not total:
        return {"score": 0.0, "applicable": False, "detail": "Sin keywords extraídas"}
    return {
        "score": round(100.0 * coverage.get("ratio", 0.0), 1),
        "applicable": True,
        "detail": f"{present}/{total} keywords del anuncio presentes en el CV",
    }


def _experience_dimension(job: Dict[str, Any], resume: Dict[str, Any]) -> Dict[str, Any]:
    required = int(job.get("min_years_experience") or 0)
    actual = float(resume.get("total_years_experience") or 0.0)

    if required > 0:
        years_score = min(1.0, actual / required) * 100 if actual else 0.0
        years_detail = f"{actual:.0f} de {required} años exigidos"
    elif actual > 0:
        years_score, years_detail = 100.0, f"{actual:.0f} años detectados (la oferta no exige mínimo)"
    else:
        years_score, years_detail = 60.0, "Años de experiencia no determinados"

    seniority_score, seniority_detail = _seniority_fit(
        job.get("seniority", "unspecified"), resume
    )
    title_score, title_detail = _title_fit(job.get("job_title", ""), resume)

    score = 0.40 * years_score + 0.30 * seniority_score + 0.30 * title_score
    return {
        "score": round(score, 1),
        "applicable": True,
        "detail": "; ".join([years_detail, seniority_detail, title_detail]),
    }


def _seniority_fit(job_seniority: str, resume: Dict[str, Any]) -> Any:
    candidate = matching.detect_seniority(
        " ".join([resume.get("current_title", "")] + list(resume.get("titles", []) or []))
    )
    if job_seniority == "unspecified" or candidate == "unspecified":
        return 70.0, "Seniority no comparable"
    gap = abs(SENIORITY_ORDER.index(job_seniority) - SENIORITY_ORDER.index(candidate))
    score = {0: 100.0, 1: 75.0, 2: 45.0}.get(gap, 25.0)
    return score, f"seniority {candidate} frente a {job_seniority} solicitado"


def _title_fit(job_title: str, resume: Dict[str, Any]) -> Any:
    titles: List[str] = [t for t in [resume.get("current_title", "")] + list(resume.get("titles", []) or []) if t]
    if not job_title or not titles:
        return 60.0, "Puestos no comparables"
    target = {
        w for w in matching.normalize(job_title).split() if len(w) > 2 and w not in {"para", "the", "and"}
    }
    if not target:
        return 60.0, "Puesto de la oferta sin términos comparables"
    best = 0.0
    for title in titles:
        words = set(matching.normalize(title).split())
        best = max(best, len(target & words) / len(target))
    return round(best * 100, 1), f"solapamiento de puesto {best:.0%}"


def _impact_dimension(bullet_stats: Dict[str, Any], resume: Dict[str, Any]) -> Dict[str, Any]:
    bullets = int(bullet_stats.get("bullets") or resume.get("bullet_count") or 0)
    quantified = int(bullet_stats.get("quantified") or resume.get("quantified_bullets") or 0)
    verbs = int(bullet_stats.get("action_verbs") or 0)
    if bullets == 0:
        return {
            "score": 20.0,
            "applicable": True,
            "detail": "Sin viñetas: no hay logros identificables",
        }
    # Con ~50 % de viñetas cuantificadas se considera óptimo.
    quant_score = min(100.0, (quantified / bullets) * 200)
    verb_score = min(100.0, (verbs / bullets) * 125)
    return {
        "score": round(0.70 * quant_score + 0.30 * verb_score, 1),
        "applicable": True,
        "detail": f"{quantified}/{bullets} viñetas con métrica, {verbs} empiezan por verbo de acción",
    }


def _blockers(format_audit: Dict[str, Any]) -> List[str]:
    """Hallazgos que por sí solos pueden tumbar la candidatura."""
    return [f["title"] for f in format_audit.get("findings", []) if f["severity"] == "critical"]
