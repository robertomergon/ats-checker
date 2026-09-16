from __future__ import annotations

from ats import config, scoring


def _match(name, status, importance="required", category="hard_skill"):
    return {
        "name": name,
        "category": category,
        "importance": importance,
        "status": status,
        "evidence": "",
        "aliases": [],
    }


FORMAT_OK = {
    "score": 100.0,
    "findings": [],
    "bullet_stats": {"bullets": 10, "quantified": 5, "action_verbs": 8, "longest_bullet": 150},
}
JOB = {"job_title": "Senior Data Engineer", "seniority": "senior", "min_years_experience": 5}
RESUME = {
    "current_title": "Senior Data Engineer",
    "titles": ["Senior Data Engineer"],
    "total_years_experience": 7,
}


def _compute(matches, coverage_ratio=1.0):
    coverage = {
        "present": ["a"] * int(coverage_ratio * 10),
        "missing": ["b"] * int((1 - coverage_ratio) * 10),
        "ratio": coverage_ratio,
    }
    return scoring.compute(matches, coverage, FORMAT_OK, JOB, RESUME)


def test_perfect_candidate_scores_high():
    result = _compute([_match("Python", "match"), _match("Airflow", "match")])
    assert result["total"] >= 90
    assert result["band"] == "Muy alta compatibilidad"
    assert result["missing_required"] == []


def test_missing_required_lowers_score_more_than_optional():
    required_gap = _compute([_match("Python", "match"), _match("Airflow", "missing")])
    optional_gap = _compute(
        [_match("Python", "match"), _match("Airflow", "missing", importance="nice_to_have")]
    )
    assert required_gap["total"] < optional_gap["total"]
    assert required_gap["missing_required"] == ["Airflow"]


def test_partial_match_gets_half_credit():
    full = _compute([_match("Kubernetes", "match")])
    partial = _compute([_match("Kubernetes", "partial")])
    missing = _compute([_match("Kubernetes", "missing")])
    assert full["total"] > partial["total"] > missing["total"]


def test_inapplicable_dimensions_redistribute_weight():
    # Solo hay requisitos técnicos: soft skills y educación no aplican.
    result = _compute([_match("Python", "match")])
    dimensions = result["dimensions"]
    assert dimensions["soft_skills"]["applicable"] is False
    assert dimensions["soft_skills"]["effective_weight"] == 0.0
    total_weight = sum(d["effective_weight"] for d in dimensions.values())
    assert abs(total_weight - 1.0) < 1e-9


def test_weights_are_configurable():
    matches = [_match("Python", "missing")]
    default = _compute(matches)
    format_heavy = scoring.compute(
        matches,
        {"present": [], "missing": ["x"], "ratio": 0.0},
        FORMAT_OK,
        JOB,
        RESUME,
        weights={**config.DEFAULT_WEIGHTS, "hard_skills": 0.05, "format": 0.40},
    )
    assert format_heavy["total"] > default["total"]


def test_experience_penalizes_missing_years():
    junior = scoring.compute(
        [_match("Python", "match")],
        {"present": ["a"], "missing": [], "ratio": 1.0},
        FORMAT_OK,
        JOB,
        {"current_title": "Data Engineer", "titles": ["Data Engineer"], "total_years_experience": 1},
    )
    assert junior["dimensions"]["experience"]["score"] < 70


def test_blockers_are_surfaced():
    audit_result = dict(
        FORMAT_OK,
        score=40.0,
        findings=[{"severity": "critical", "title": "No se detecta email", "penalty": 20}],
    )
    result = scoring.compute(
        [_match("Python", "match")],
        {"present": ["a"], "missing": [], "ratio": 1.0},
        audit_result,
        JOB,
        RESUME,
    )
    assert result["blockers"] == ["No se detecta email"]
