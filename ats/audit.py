"""Auditoría de legibilidad del CV para un ATS.

Cada comprobación devuelve un hallazgo con severidad y penalización. El score de
formato es `100 - suma(penalizaciones)`, acotado a [0, 100].
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from . import config, matching
from .data import (
    CORE_SECTIONS,
    INJECTION_PATTERNS,
    RISKY_GLYPHS,
    SKILL_DICTIONARY,
    STOPWORDS,
)

_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

SECTION_LABELS = {"experiencia": "Experiencia", "educacion": "Educación", "skills": "Skills"}


def _finding(
    check: str, severity: str, penalty: int, title: str, detail: str, fix: str = ""
) -> Dict[str, Any]:
    return {
        "check": check,
        "severity": severity,
        "penalty": penalty,
        "title": title,
        "detail": detail,
        "fix": fix,
    }


def audit_format(document: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
    """Ejecuta todas las comprobaciones de formato sobre el CV extraído."""
    findings: List[Dict[str, Any]] = []
    sections = matching.detect_sections(resume_text)
    contact = matching.contact_signals(resume_text)
    stats = matching.bullet_stats(resume_text)

    findings += _check_extraction(document)
    findings += _check_contact(contact)
    findings += _check_sections(sections)
    findings += _check_layout(document)
    findings += _check_length(document)
    findings += _check_bullets(stats)
    findings += _check_glyphs(resume_text)
    findings += _check_dates(resume_text)
    integrity = _check_integrity(document, resume_text)
    findings += integrity["findings"]

    penalty = sum(f["penalty"] for f in findings)
    score = max(0.0, min(100.0, 100.0 - penalty))

    return {
        "score": score,
        "findings": sorted(findings, key=lambda f: -f["penalty"]),
        "sections": sections,
        "contact": contact,
        "bullet_stats": stats,
        "integrity": {k: v for k, v in integrity.items() if k != "findings"},
    }


# --- Comprobaciones -------------------------------------------------------


def _check_extraction(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not doc.get("extraction_ok"):
        out.append(
            _finding(
                "no_text", "critical", 60,
                "El documento no devuelve texto seleccionable",
                "Ningún parser pudo extraer texto: el ATS verá un CV vacío.",
                "Exporta el CV a PDF desde el editor de texto (no escaneado ni imagen).",
            )
        )
        return out

    pages = max(1, doc.get("pages", 1))
    if doc["chars"] / pages < config.MIN_CHARS_PER_PAGE:
        out.append(
            _finding(
                "low_text_density", "critical", 35,
                "Densidad de texto muy baja",
                f"Solo {doc['chars'] // pages} caracteres por página: gran parte del "
                "contenido probablemente está dentro de imágenes.",
                "Sustituye cualquier bloque de texto incrustado en imagen por texto real.",
            )
        )
    if doc.get("ext") not in ("pdf", "docx"):
        out.append(
            _finding(
                "file_format", "low", 4,
                f"Formato .{doc.get('ext')} poco habitual en portales de empleo",
                "La mayoría de ATS esperan PDF o DOCX.",
                "Exporta el CV a PDF manteniendo el texto seleccionable.",
            )
        )
    return out


def _check_contact(contact: Dict[str, bool]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not contact["email"]:
        out.append(
            _finding(
                "no_email", "critical", 20,
                "No se detecta email",
                "Sin email el ATS no puede crear la ficha de candidato.",
                "Escribe el email como texto plano en la cabecera, no dentro de un icono.",
            )
        )
    if not contact["phone"]:
        out.append(
            _finding(
                "no_phone", "medium", 6,
                "No se detecta teléfono",
                "Muchos formularios ATS dejan el campo vacío y penalizan la ficha.",
                "Añade el teléfono en formato internacional: +34 600 000 000.",
            )
        )
    if not contact["linkedin"]:
        out.append(
            _finding(
                "no_linkedin", "low", 3,
                "No se detecta perfil de LinkedIn",
                "Es un campo que muchos ATS rellenan automáticamente.",
                "Incluye la URL completa de tu perfil como texto.",
            )
        )
    return out


def _check_sections(sections: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    missing = [s for s in CORE_SECTIONS if s not in sections["found"]]
    for name in missing:
        out.append(
            _finding(
                f"missing_section_{name}", "high", 10,
                f"Falta un encabezado reconocible de «{SECTION_LABELS[name]}»",
                "El parser segmenta el CV por encabezados estándar; sin ellos mete "
                "todo el texto en un único bloque sin estructurar.",
                f"Usa literalmente «{SECTION_LABELS[name]}» como título de sección.",
            )
        )
    if sections["unknown_headings"]:
        out.append(
            _finding(
                "creative_headings", "medium", 6,
                "Encabezados no estándar",
                "Títulos que el ATS no sabe clasificar: "
                + ", ".join(f"«{h}»" for h in sections["unknown_headings"][:4]),
                "Renombra a los términos convencionales (Experiencia, Educación, Skills).",
            )
        )
    return out


def _check_layout(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if doc.get("tables"):
        out.append(
            _finding(
                "tables", "high", 12,
                f"{doc['tables']} tabla(s) en el documento",
                "Los parsers leen las tablas celda a celda y rompen el orden de lectura.",
                "Pasa el contenido de las tablas a párrafos y viñetas normales.",
            )
        )
    if doc.get("columns_suspected"):
        out.append(
            _finding(
                "multicolumn", "high", 12,
                "Maquetación a dos columnas",
                "El texto extraído mezcla líneas de ambas columnas: el ATS leerá frases partidas.",
                "Reorganiza el CV en una sola columna de ancho completo.",
            )
        )
    if doc.get("images", 0) > 2:
        out.append(
            _finding(
                "images", "low", 4,
                f"{doc['images']} imágenes incrustadas",
                "Iconos y gráficos no aportan nada al ATS y pueden desplazar el texto.",
                "Elimina los iconos decorativos; deja el texto de contacto como texto.",
            )
        )
    if doc.get("has_header_footer"):
        out.append(
            _finding(
                "header_footer", "medium", 6,
                "Hay contenido en cabecera o pie de página",
                "Buena parte de los parsers ignoran cabeceras y pies.",
                "Mueve esa información (sobre todo el contacto) al cuerpo del documento.",
            )
        )
    if doc.get("pages", 0) > 3:
        out.append(
            _finding(
                "too_many_pages", "medium", 6,
                f"{doc['pages']} páginas",
                "Por encima de 2-3 páginas el cribado humano posterior se vuelve improbable.",
                "Recorta a 1-2 páginas priorizando los últimos 10 años.",
            )
        )
    return out


def _check_length(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    low, high = config.WORD_COUNT_OK
    words = doc.get("words", 0)
    if words and words < low:
        return [
            _finding(
                "too_short", "medium", 8,
                f"CV muy corto ({words} palabras)",
                "Hay poco texto del que extraer keywords, así que el match será bajo por construcción.",
                f"Amplía logros y stack técnico hasta las {low}-{high} palabras.",
            )
        ]
    if words > high * 1.6:
        return [
            _finding(
                "too_long", "low", 4,
                f"CV muy extenso ({words} palabras)",
                "Diluye las keywords relevantes entre contenido secundario.",
                "Condensa responsabilidades repetidas y elimina puestos de hace más de 10-15 años.",
            )
        ]
    return []


def _check_bullets(stats: Dict[str, int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if stats["bullets"] == 0:
        out.append(
            _finding(
                "no_bullets", "medium", 8,
                "No hay viñetas",
                "Los bloques de texto corrido dificultan extraer logros y responsabilidades.",
                "Convierte cada responsabilidad en una viñeta que empiece por verbo de acción.",
            )
        )
    elif stats["action_verbs"] / stats["bullets"] < 0.4:
        out.append(
            _finding(
                "weak_verbs", "low", 4,
                "Pocas viñetas empiezan por verbo de acción",
                f"{stats['action_verbs']} de {stats['bullets']} viñetas.",
                "Reescribe en pasado y en primera persona implícita: «Lideré…», «Reduje…».",
            )
        )
    if stats["longest_bullet"] > 320:
        out.append(
            _finding(
                "long_bullets", "low", 3,
                "Viñetas demasiado largas",
                f"La más larga tiene {stats['longest_bullet']} caracteres.",
                "Máximo dos líneas por viñeta; divide las que mezclen varias ideas.",
            )
        )
    return out


def _check_glyphs(text: str) -> List[Dict[str, Any]]:
    risky = {ch for ch in text if ch in RISKY_GLYPHS}
    if len(risky) >= 2:
        return [
            _finding(
                "risky_glyphs", "low", 3,
                "Símbolos decorativos poco estándar",
                "Caracteres detectados: " + " ".join(sorted(risky)),
                "Usa un guion o una viñeta simple; algunos parsers los convierten en basura.",
            )
        ]
    return []


def _check_dates(text: str) -> List[Dict[str, Any]]:
    ranges = len(matching._DATE_RANGE.findall(text))
    if ranges == 0:
        return [
            _finding(
                "no_date_ranges", "high", 10,
                "No se detectan rangos de fechas",
                "Sin fechas el ATS no puede calcular antigüedad ni años de experiencia.",
                "Indica cada puesto como «Ene 2021 – Actualidad» con años de 4 dígitos.",
            )
        ]
    return []


def _check_integrity(doc: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Detecta texto oculto, relleno de keywords e intentos de inyección de prompt.

    El contenido de un CV es dato no confiable: si alguien esconde instrucciones
    dirigidas al modelo, se reportan aquí y nunca se ejecutan.
    """
    findings: List[Dict[str, Any]] = []
    injections = [
        m.group(0)
        for pattern in _INJECTION_RE
        for m in [pattern.search(text)]
        if m
    ]
    stuffing = _keyword_stuffing(text)

    if doc.get("hidden_text_runs"):
        findings.append(
            _finding(
                "hidden_text", "critical", 30,
                f"{doc['hidden_text_runs']} fragmento(s) de texto oculto",
                "Texto en blanco, oculto o de tamaño microscópico. Los ATS modernos lo "
                "detectan y marcan la candidatura como manipulada.",
                "Elimina todo el texto invisible del documento.",
            )
        )
    if injections:
        findings.append(
            _finding(
                "prompt_injection", "critical", 25,
                "El CV contiene instrucciones dirigidas a un sistema automático",
                "Fragmentos detectados: " + " | ".join(f"«{i[:70]}»" for i in injections[:3]),
                "Elimínalos: se tratan como texto sospechoso y nunca se obedecen.",
            )
        )
    if stuffing:
        findings.append(
            _finding(
                "keyword_stuffing", "high", 12,
                "Posible relleno de keywords",
                "Términos repetidos de forma anómala: "
                + ", ".join(f"{t} (×{n})" for t, n in stuffing[:4]),
                "Menciona cada tecnología donde la usaste de verdad, no en listas repetidas.",
            )
        )

    return {
        "findings": findings,
        "hidden_text_runs": doc.get("hidden_text_runs", 0),
        "hidden_text_samples": doc.get("hidden_text_samples", []),
        "injection_hits": injections,
        "stuffed_terms": stuffing,
    }


#: Términos de una sola palabra del diccionario de skills, normalizados.
_SKILL_TOKENS = {
    matching.normalize(term)
    for canonical, aliases in SKILL_DICTIONARY.items()
    for term in [canonical, *aliases]
    if " " not in matching.normalize(term)
}


def _keyword_stuffing(text: str) -> List[Any]:
    """Términos repetidos con frecuencia anómala para el tamaño del CV.

    El umbral es más estricto para palabras que no son tecnologías: un CV de
    data engineer repite «datos» con naturalidad, pero nadie escribe «Python»
    doce veces si no está rellenando keywords.
    """
    from collections import Counter

    words = [w for w in matching.normalize(text).split() if len(w) > 3 and w not in STOPWORDS]
    if len(words) < 150:
        return []

    skill_threshold = max(8, int(len(words) * 0.02))
    generic_threshold = max(14, int(len(words) * 0.035))
    counts = Counter(words)

    flagged = [
        (word, count)
        for word, count in counts.most_common(20)
        if count >= (skill_threshold if word in _SKILL_TOKENS else generic_threshold)
    ]
    return flagged[:6]
