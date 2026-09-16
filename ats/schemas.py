"""Esquemas de datos: salidas estructuradas de Claude y estado del grafo.

Los modelos Pydantic se pasan a `client.beta.messages.parse(output_format=...)`,
así que evitan `Optional`: cada campo usa un valor centinela documentado
(`0`, `""`, `"unspecified"`) para que el JSON Schema sea plano y estricto.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict

# --- Salidas estructuradas del LLM ---------------------------------------

Importance = Literal["required", "preferred", "nice_to_have"]
RequirementCategory = Literal[
    "hard_skill", "soft_skill", "tool", "certification", "education", "responsibility", "domain"
]
Seniority = Literal[
    "intern", "junior", "mid", "senior", "lead", "manager", "director", "executive", "unspecified"
]


class JobRequirement(BaseModel):
    """Un requisito atómico extraído de la oferta."""

    name: str = Field(description="El requisito en su forma canónica y corta, p.ej. 'Kubernetes'.")
    category: RequirementCategory
    importance: Importance = Field(
        description="'required' si la oferta lo marca como imprescindible o excluyente."
    )
    # Sin valor por defecto: las salidas estructuradas estrictas exigen que todas
    # las propiedades estén en `required`.
    aliases: List[str] = Field(
        description=(
            "Sinónimos, abreviaturas y traducciones con las que este requisito puede "
            "aparecer en un CV (p.ej. para 'Kubernetes': ['k8s']). Lista vacía si no hay."
        ),
    )


class JobProfile(BaseModel):
    """Perfil estructurado de la oferta de empleo."""

    job_title: str
    seniority: Seniority
    min_years_experience: int = Field(
        description="Años mínimos de experiencia exigidos. 0 si la oferta no lo especifica."
    )
    requirements: List[JobRequirement]
    key_responsibilities: List[str] = Field(
        description="Responsabilidades principales, máximo 8, en frases cortas."
    )
    ats_keywords: List[str] = Field(
        description=(
            "Entre 15 y 30 términos literales que un filtro ATS buscaría en el CV, "
            "tal y como aparecen en el anuncio."
        )
    )
    language: str = Field(description="Idioma del anuncio: 'es', 'en' u otro código ISO 639-1.")


class ResumeProfile(BaseModel):
    """Perfil estructurado del CV."""

    candidate_name: str = Field(description="Nombre del candidato, o '' si no se identifica.")
    current_title: str = Field(description="Puesto más reciente, o '' si no se identifica.")
    total_years_experience: float = Field(
        description="Años totales de experiencia profesional estimados a partir de las fechas. 0 si no hay datos."
    )
    titles: List[str] = Field(description="Puestos ocupados, del más reciente al más antiguo.")
    companies: List[str]
    skills: List[str] = Field(description="Skills técnicos explícitos en el CV.")
    tools: List[str] = Field(description="Herramientas, frameworks y plataformas mencionadas.")
    education: List[str] = Field(description="Titulaciones, una por entrada.")
    certifications: List[str]
    languages: List[str] = Field(description="Idiomas con su nivel si consta.")
    bullet_count: int = Field(description="Número total de viñetas de logros/responsabilidades.")
    quantified_bullets: int = Field(
        description="Viñetas que incluyen una métrica concreta (%, €, volumen, tiempo)."
    )


class RequirementVerdict(BaseModel):
    """Veredicto del LLM sobre un requisito que el match literal no encontró."""

    requirement: str = Field(description="El requisito evaluado, copiado literalmente.")
    status: Literal["match", "partial", "missing"]
    evidence: str = Field(
        description="Cita textual del CV que lo respalda, o '' si el estado es 'missing'."
    )
    reasoning: str = Field(description="Una frase explicando la decisión.")


class SemanticMatchResult(BaseModel):
    verdicts: List[RequirementVerdict]


class Recommendation(BaseModel):
    priority: Literal["critical", "high", "medium", "low"]
    area: Literal["keywords", "experience", "format", "skills", "impact", "education", "other"]
    issue: str = Field(description="Qué está mal, en una frase.")
    action: str = Field(description="Qué hacer exactamente, en imperativo.")
    example: str = Field(
        description=(
            "Texto listo para pegar en el CV (viñeta reescrita, línea de skills...). "
            "'' si la acción no requiere un ejemplo."
        )
    )


class RecommendationSet(BaseModel):
    verdict_summary: str = Field(
        description="Dos o tres frases sobre el encaje real del candidato con la oferta."
    )
    recommendations: List[Recommendation] = Field(
        description="Entre 4 y 10 recomendaciones ordenadas por impacto en el score."
    )
    keywords_to_add: List[str] = Field(
        description="Keywords ausentes que el candidato puede añadir con honestidad."
    )


# --- Estado del grafo -----------------------------------------------------


class ATSState(TypedDict, total=False):
    """Estado compartido entre los nodos de LangGraph.

    Los valores son dicts/listas (no modelos Pydantic) para que el estado sea
    serializable y se pueda guardar en `st.session_state` o en un checkpointer.
    `warnings` y `usage` llevan reducer porque los escriben nodos paralelos.
    """

    # Entradas
    resume_filename: str
    job_description: str
    use_llm: bool
    effort: str
    weights: Dict[str, float]

    # Nodo parse_resume
    resume_text: str
    document: Dict[str, Any]

    # Nodos LLM / heurísticos
    job_profile: Dict[str, Any]
    resume_profile: Dict[str, Any]

    # Matching y auditoría
    matches: List[Dict[str, Any]]
    keyword_coverage: Dict[str, Any]
    format_audit: Dict[str, Any]

    # Salidas
    scores: Dict[str, Any]
    recommendations: Dict[str, Any]

    # Telemetría acumulada por nodos en paralelo
    warnings: Annotated[List[str], operator.add]
    usage: Annotated[List[Dict[str, Any]], operator.add]
