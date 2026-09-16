"""Normalización de texto y matching determinista de requisitos.

El objetivo es imitar cómo un ATS busca términos: coincidencia literal tolerante
a separadores (`node.js` ≈ `nodejs` ≈ `node js`), insensible a acentos y
mayúsculas, pero con límites de palabra para no dar falsos positivos con siglas
cortas (`R`, `Go`, `AWS`).
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .data import (
    ACTION_VERBS,
    SECTION_ALIASES,
    SENIORITY_SIGNALS,
    SKILL_DICTIONARY,
    SOFT_SKILL_TERMS,
    STOPWORDS,
)

_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789+#./&- ")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9+#]+")
_JOIN = r"[\s./_\-]{0,2}"
_BULLET_LINE = re.compile(r"^\s*([\-\*•▪●◦‣⁃·∙]|\d+[\.\)])\s+")
_METRIC = re.compile(
    r"(\d+\s?%"                                        # porcentajes
    r"|[€$£]\s?\d|\d+\s?(?:€|\$|eur|usd)"              # importes
    r"|\d+\s?(?:k|m|mm|bn|millones|million|tb|gb|mb)\b"  # magnitudes
    r"|\bx\s?\d+\b"                                    # multiplicadores
    r"|\d+\s?(?:d[ií]as?|horas?|semanas?|meses|a[nñ]os?|days?|hours?|weeks?|months?)\b"
    r"|\d{2,})",                                       # cifras de dos o más dígitos
    re.I,
)
_YEARS_REQ = re.compile(r"(\d{1,2})\s*(?:\+|o m[aá]s)?\s*(?:a[nñ]os|years?)", re.I)
_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().\-]{6,}\d")
_DATE_RANGE = re.compile(
    r"(19|20)\d{2}\s*(?:-|–|—|a|to|hasta)\s*((19|20)\d{2}|actualidad|presente|present|current|hoy)",
    re.I,
)
_REQUIRED_MARKERS = re.compile(
    r"(imprescindible|requisito|requerid|obligatori|excluyente|must[-\s]?have|required|necesario|"
    r"se requiere|deber[aá]s|essential)",
    re.I,
)
_NICE_MARKERS = re.compile(
    r"(deseable|valorable|se valorar[aá]|plus|nice[-\s]?to[-\s]?have|preferred|bonus|opcional)",
    re.I,
)


# --- Normalización --------------------------------------------------------


def normalize(text: str) -> str:
    """Texto en minúsculas, sin acentos, con espacios colapsados."""
    return normalize_with_map(text)[0]


def normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Normaliza conservando, por cada carácter de salida, su índice original.

    El mapa permite recuperar una cita literal del documento original a partir
    de la posición de un match hecho sobre el texto normalizado.
    """
    out: List[str] = []
    index_map: List[int] = []
    prev_space = True  # evita espacio inicial
    for i, ch in enumerate(text):
        decomposed = unicodedata.normalize("NFKD", ch)
        folded = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
        for c in folded or " ":
            if c not in _ALLOWED:
                c = " "
            if c == " ":
                if prev_space:
                    continue
                prev_space = True
            else:
                prev_space = False
            out.append(c)
            index_map.append(i)
    while out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map


@functools.lru_cache(maxsize=4096)
def term_pattern(term: str) -> Optional[re.Pattern]:
    """Compila el patrón de búsqueda de un término ya normalizado."""
    parts = [p for p in _TOKEN_SPLIT.split(term.strip()) if p]
    if not parts:
        return None
    body = _JOIN.join(re.escape(p) for p in parts)
    return re.compile(r"(?<![a-z0-9+#])" + body + r"(?![a-z0-9+#])")


def find_term(term: str, haystack_norm: str) -> Optional[Tuple[int, int]]:
    """Posición (inicio, fin) del término en el texto normalizado, o None."""
    pattern = term_pattern(normalize(term))
    if pattern is None:
        return None
    match = pattern.search(haystack_norm)
    return (match.start(), match.end()) if match else None


def token_presence(term: str, haystack_norm: str) -> float:
    """Fracción de tokens del término presentes por separado (match parcial)."""
    parts = [p for p in _TOKEN_SPLIT.split(normalize(term)) if p and p not in STOPWORDS]
    if not parts:
        return 0.0
    hits = sum(1 for p in parts if term_pattern(p) and term_pattern(p).search(haystack_norm))
    return hits / len(parts)


def snippet(original: str, index_map: Sequence[int], span: Tuple[int, int], width: int = 70) -> str:
    """Cita del documento original alrededor de un match normalizado."""
    if not index_map:
        return ""
    start = index_map[max(0, min(span[0], len(index_map) - 1))]
    end = index_map[max(0, min(span[1] - 1, len(index_map) - 1))] + 1
    left = max(0, start - width)
    right = min(len(original), end + width)
    text = original[left:right].replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return ("…" if left > 0 else "") + text + ("…" if right < len(original) else "")


# --- Matching de requisitos ----------------------------------------------


def match_requirements(
    requirements: Iterable[Dict[str, Any]], resume_text: str
) -> List[Dict[str, Any]]:
    """Evalúa cada requisito contra el texto del CV.

    Devuelve un match por requisito con `status` (match/partial/missing),
    la evidencia literal y el término que disparó la coincidencia.
    """
    norm, index_map = normalize_with_map(resume_text)
    results: List[Dict[str, Any]] = []

    for req in requirements:
        name = str(req.get("name", "")).strip()
        if not name:
            continue
        aliases = [a for a in req.get("aliases", []) or [] if str(a).strip()]
        status, evidence, hit_term, source = "missing", "", "", "literal"

        for candidate in [name] + aliases:
            span = find_term(candidate, norm)
            if span:
                status = "match"
                hit_term = candidate
                evidence = snippet(resume_text, index_map, span)
                break

        if status == "missing":
            best = max((token_presence(c, norm), c) for c in [name] + aliases)
            if best[0] >= 0.5 and len(_TOKEN_SPLIT.split(normalize(best[1]))) > 1:
                status, hit_term = "partial", best[1]

        results.append(
            {
                "name": name,
                "category": req.get("category", "hard_skill"),
                "importance": req.get("importance", "preferred"),
                "aliases": aliases,
                "status": status,
                "evidence": evidence,
                "matched_term": hit_term,
                "source": source,
            }
        )
    return results


def keyword_coverage(keywords: Sequence[str], resume_text: str) -> Dict[str, Any]:
    """Cobertura de las keywords literales del anuncio en el CV."""
    norm, _ = normalize_with_map(resume_text)
    present, missing = [], []
    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue
        (present if find_term(kw, norm) else missing).append(kw)
    total = len(present) + len(missing)
    return {
        "present": present,
        "missing": missing,
        "ratio": (len(present) / total) if total else 0.0,
    }


# --- Secciones ------------------------------------------------------------


def detect_sections(text: str) -> Dict[str, Any]:
    """Detecta encabezados de sección reconocibles y los que no lo son."""
    found: Dict[str, int] = {}
    unknown: List[str] = []
    lines = text.split("\n")

    alias_to_section = {
        normalize(alias): section
        for section, aliases in SECTION_ALIASES.items()
        for alias in aliases
    }

    for idx, raw in enumerate(lines):
        line = raw.strip().rstrip(":").strip()
        if not line or len(line) > 60 or line.endswith("."):
            continue
        key = normalize(line)
        if key in alias_to_section:
            found.setdefault(alias_to_section[key], idx)
        elif _looks_like_heading(raw, lines, idx):
            unknown.append(line)

    return {"found": found, "unknown_headings": unknown[:8]}


#: Las primeras líneas son la cabecera (nombre y contacto), nunca encabezados
#: de sección: sin esto el nombre del candidato se reporta como sección inventada.
_HEADER_LINES = 3


def _looks_like_heading(raw: str, lines: List[str], idx: int) -> bool:
    """Línea corta, sin puntuación final, en mayúsculas o capitalizada, con contenido debajo."""
    if idx < _HEADER_LINES:
        return False
    line = raw.strip()
    if not (2 <= len(line.split()) <= 5):
        return False
    if not line.replace(" ", "").isalpha():
        return False
    if not (line.isupper() or line.istitle()):
        return False
    following = "".join(lines[idx + 1 : idx + 3]).strip()
    return len(following) > 30


def section_text(text: str, sections: Dict[str, int], name: str) -> str:
    """Texto perteneciente a una sección, hasta el siguiente encabezado."""
    if name not in sections:
        return ""
    lines = text.split("\n")
    start = sections[name] + 1
    later = [i for i in sections.values() if i > sections[name]]
    end = min(later) if later else len(lines)
    return "\n".join(lines[start:end]).strip()


# --- Señales de contenido ------------------------------------------------


def group_bullets(text: str) -> List[str]:
    """Agrupa cada viñeta con sus líneas de continuación.

    Importa porque la métrica de un logro suele caer en la segunda línea
    («…reduciendo\\n  los fallos un 68 %»): contando línea a línea se perdería.
    """
    bullets: List[str] = []
    current: Optional[str] = None
    for raw in text.split("\n"):
        if _BULLET_LINE.match(raw):
            if current is not None:
                bullets.append(current)
            current = raw.strip()
            continue
        if current is None:
            continue
        stripped = raw.strip()
        # Continúa la viñeta si va sangrada o empieza en minúscula; cualquier
        # otra línea (encabezado, nuevo puesto, línea vacía) la cierra.
        if stripped and (raw[:1].isspace() or stripped[:1].islower()):
            current += " " + stripped
        else:
            bullets.append(current)
            current = None
    if current is not None:
        bullets.append(current)
    return bullets


def bullet_stats(text: str) -> Dict[str, int]:
    """Cuenta viñetas, cuántas llevan métrica y cuántas empiezan por verbo de acción."""
    bullets = group_bullets(text)
    quantified = sum(1 for b in bullets if _METRIC.search(b))
    action = 0
    for b in bullets:
        words = normalize(_BULLET_LINE.sub("", b)).split()
        if words and words[0] in ACTION_VERBS:
            action += 1
    longest = max((len(b) for b in bullets), default=0)
    return {
        "bullets": len(bullets),
        "quantified": quantified,
        "action_verbs": action,
        "longest_bullet": longest,
    }


def _has_phone(text: str) -> bool:
    """Teléfono real, no un rango de fechas.

    Un «2019 - 2020» encaja en cualquier patrón laxo de teléfono, así que se
    exige además un recuento de dígitos plausible.
    """
    for candidate in _PHONE_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", candidate.group(0))
        if 9 <= len(digits) <= 15:
            return True
    return False


def contact_signals(text: str) -> Dict[str, bool]:
    norm = text.lower()
    return {
        "email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]{2,}", text)),
        "phone": _has_phone(text),
        "linkedin": "linkedin." in norm,
        "location": bool(re.search(r"\b(madrid|barcelona|valencia|sevilla|bilbao|remoto|remote|m[eé]xico|bogot[aá]|lima|santiago|buenos aires)\b", norm)),
    }


def detect_seniority(text: str) -> str:
    """Seniority más alto mencionado en el texto."""
    norm = normalize(text)
    best = "unspecified"
    for level, signals in SENIORITY_SIGNALS.items():
        if any(find_term(s, norm) for s in signals):
            best = level
    return best


def estimate_years(text: str) -> float:
    """Años de experiencia a partir de rangos de fechas; si no hay, de '<n> años'."""
    years: List[int] = []
    for match in _DATE_RANGE.finditer(text):
        start = int(match.group(0)[:4])
        years.append(start)
    if years:
        import datetime

        current = datetime.date.today().year
        ends = [
            current if re.search(r"(actualidad|presente|present|current|hoy)", m.group(0), re.I)
            else int(m.group(2)[:4])
            for m in _DATE_RANGE.finditer(text)
        ]
        return float(max(0, max(ends) - min(years)))
    stated = [int(m.group(1)) for m in _YEARS_REQ.finditer(text)]
    return float(max(stated)) if stated else 0.0


def required_years(text: str) -> int:
    """Años mínimos exigidos en una oferta (el menor valor mencionado)."""
    stated = [int(m.group(1)) for m in _YEARS_REQ.finditer(text) if 0 < int(m.group(1)) < 30]
    return min(stated) if stated else 0


# --- Perfiles heurísticos (camino sin LLM) --------------------------------


_IMPORTANCE_RANK = {"required": 3, "preferred": 2, "nice_to_have": 1}


def heuristic_job_profile(job_description: str) -> Dict[str, Any]:
    """Perfil de la oferta sin LLM: diccionario de skills + n-gramas frecuentes.

    La exigencia se deduce del bloque en que aparece cada término («Requisitos
    imprescindibles» frente a «Se valorará»), que es como están redactadas las
    ofertas reales; un marcador en la propia línea tiene prioridad.
    """
    found: Dict[str, str] = {}
    block_importance = "preferred"

    for raw_line in job_description.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _NICE_MARKERS.search(line):
            importance = "nice_to_have"
        elif _REQUIRED_MARKERS.search(line):
            importance = "required"
        else:
            importance = block_importance

        # Un encabezado corto con marcador fija la exigencia de las líneas que
        # vienen debajo. La línea se sigue escaneando: «Se valorará Kafka»
        # es a la vez encabezado y requisito.
        if importance != block_importance and len(line) < 80 and len(line.split()) <= 6:
            block_importance = importance

        line_norm = normalize(line)
        for term, aliases in SKILL_DICTIONARY.items():
            if not (
                find_term(term, line_norm)
                or any(find_term(alias, line_norm) for alias in aliases)
            ):
                continue
            previous = found.get(term)
            if previous is None or _IMPORTANCE_RANK[importance] > _IMPORTANCE_RANK[previous]:
                found[term] = importance

    requirements = [
        {
            "name": term,
            "category": "soft_skill" if term in SOFT_SKILL_TERMS else "hard_skill",
            "importance": importance,
            "aliases": list(SKILL_DICTIONARY[term]),
        }
        for term, importance in found.items()
    ]

    title = next((ln.strip() for ln in job_description.split("\n") if ln.strip()), "")
    return {
        "job_title": title[:120],
        "seniority": detect_seniority(job_description),
        "min_years_experience": required_years(job_description),
        "requirements": requirements,
        "key_responsibilities": [],
        "ats_keywords": top_keywords(job_description, limit=25),
        "language": "es" if _looks_spanish(normalize(job_description)) else "en",
        "_source": "heuristic",
    }


def heuristic_resume_profile(resume_text: str) -> Dict[str, Any]:
    """Perfil del CV sin LLM: secciones, diccionario de skills y fechas."""
    norm, _ = normalize_with_map(resume_text)
    sections = detect_sections(resume_text)["found"]
    skills = [
        term
        for term, aliases in SKILL_DICTIONARY.items()
        if find_term(term, norm) or any(find_term(a, norm) for a in aliases)
    ]
    stats = bullet_stats(resume_text)
    experience_block = section_text(resume_text, sections, "experiencia")
    titles = [
        ln.strip()
        for ln in experience_block.split("\n")
        if _DATE_RANGE.search(ln) and len(ln.strip()) < 120
    ]
    # Los años se cuentan solo sobre la sección de experiencia: incluir las
    # fechas de la carrera universitaria infla el total varios años.
    years = estimate_years(experience_block or resume_text)
    return {
        "candidate_name": next((ln.strip() for ln in resume_text.split("\n") if ln.strip()), "")[:80],
        "current_title": titles[0] if titles else "",
        "total_years_experience": years,
        "titles": titles[:10],
        "companies": [],
        "skills": [s for s in skills if s not in SOFT_SKILL_TERMS],
        "tools": [],
        "education": _lines(section_text(resume_text, sections, "educacion"))[:6],
        "certifications": _lines(section_text(resume_text, sections, "certificaciones"))[:6],
        "languages": _lines(section_text(resume_text, sections, "idiomas"))[:5],
        "bullet_count": stats["bullets"],
        "quantified_bullets": stats["quantified"],
        "_source": "heuristic",
    }


def top_keywords(text: str, limit: int = 25) -> List[str]:
    """Unigramas y bigramas más frecuentes, descartando stopwords."""
    from collections import Counter

    words = [w for w in normalize(text).split() if len(w) > 2 and w not in STOPWORDS]
    counter = Counter(words)
    bigrams = Counter(
        f"{a} {b}" for a, b in zip(words, words[1:]) if a not in STOPWORDS and b not in STOPWORDS
    )
    ranked = [w for w, c in counter.most_common(limit * 2) if c > 1]
    ranked += [b for b, c in bigrams.most_common(limit) if c > 1]
    seen, out = set(), []
    for term in ranked:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out[:limit]


def _lines(text: str) -> List[str]:
    return [ln.strip(" -•\t") for ln in text.split("\n") if len(ln.strip()) > 3]


def _looks_spanish(norm: str) -> bool:
    markers = ("experiencia", "requisitos", "empresa", "equipo", "conocimientos", "buscamos")
    return sum(1 for m in markers if m in norm) >= 2
