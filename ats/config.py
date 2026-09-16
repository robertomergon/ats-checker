"""Configuración central del ATS checker."""

from __future__ import annotations

from typing import Dict, List, Tuple

# --- Modelo ---------------------------------------------------------------

#: Niveles de esfuerzo de la app. Cada proveedor los traduce a su parámetro
#: equivalente (`output_config.effort`, `reasoning.effort`, `thinking_level`).
EFFORT_LEVELS: List[str] = ["low", "medium", "high", "xhigh", "max"]

#: `medium` da buena relación calidad/latencia para extracción estructurada.
#: Se puede subir desde la UI cuando interesa más la precisión que el coste.
DEFAULT_EFFORT = "medium"

# --- Pesos del score -----------------------------------------------------

#: Pesos por dimensión. Suman 1.0. Editables desde la UI.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "hard_skills": 0.30,
    "keywords": 0.15,
    "experience": 0.20,
    "format": 0.15,
    "impact": 0.10,
    "education": 0.05,
    "soft_skills": 0.05,
}

DIMENSION_LABELS: Dict[str, str] = {
    "hard_skills": "Skills técnicos requeridos",
    "keywords": "Cobertura de keywords del anuncio",
    "experience": "Experiencia y seniority",
    "format": "Legibilidad para el ATS",
    "impact": "Logros cuantificados",
    "education": "Formación y certificaciones",
    "soft_skills": "Competencias blandas",
}

#: Peso relativo de cada requisito según su importancia en la oferta.
IMPORTANCE_WEIGHT: Dict[str, float] = {
    "required": 3.0,
    "preferred": 2.0,
    "nice_to_have": 1.0,
}

#: Crédito otorgado a cada estado de match.
STATUS_CREDIT: Dict[str, float] = {
    "match": 1.0,
    "partial": 0.5,
    "missing": 0.0,
}

#: Bandas de interpretación del score final (mínimo inclusivo).
SCORE_BANDS: List[Tuple[int, str, str]] = [
    (85, "Muy alta compatibilidad", "El CV debería pasar el filtro ATS y llegar a revisión humana."),
    (70, "Compatibilidad buena", "Pasa la mayoría de filtros; quedan mejoras claras de keywords."),
    (55, "Compatibilidad media", "Riesgo real de descarte automático. Requiere ajustes antes de enviar."),
    (0, "Compatibilidad baja", "Muy probable descarte en el cribado automático."),
]

# --- Extracción de ficheros ---------------------------------------------

SUPPORTED_EXTENSIONS = ("pdf", "docx", "txt", "md")

#: Por debajo de esta densidad de texto por página asumimos PDF escaneado.
MIN_CHARS_PER_PAGE = 200

#: Rango de palabras razonable para un CV de una a dos páginas.
WORD_COUNT_OK = (300, 1000)


def band_for(score: float) -> Tuple[str, str]:
    """Devuelve (etiqueta, explicación) para un score 0-100."""
    for threshold, label, detail in SCORE_BANDS:
        if score >= threshold:
            return label, detail
    return SCORE_BANDS[-1][1], SCORE_BANDS[-1][2]

