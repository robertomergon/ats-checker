from __future__ import annotations

from ats import matching


def test_normalize_folds_accents_and_spaces():
    assert matching.normalize("  Ingeniería   INFORMÁTICA ") == "ingenieria informatica"


def test_find_term_tolerates_separators():
    text = matching.normalize("Experiencia con NodeJS y Node.js en producción")
    assert matching.find_term("node.js", text)
    assert matching.find_term("nodejs", text)


def test_find_term_keeps_symbols():
    text = matching.normalize("Stack: C++, C# y .NET")
    assert matching.find_term("c++", text)
    assert matching.find_term("c#", text)


def test_short_terms_respect_word_boundaries():
    assert matching.find_term("r", matching.normalize("R y Python")) is not None
    assert matching.find_term("r", matching.normalize("Rust y Python")) is None
    assert matching.find_term("go", matching.normalize("Google Cloud")) is None


def test_match_requirements_statuses(resume_strong):
    requirements = [
        {"name": "Apache Airflow", "category": "tool", "importance": "required", "aliases": ["airflow"]},
        {"name": "Spark", "category": "tool", "importance": "required", "aliases": ["pyspark"]},
        {"name": "Kafka", "category": "tool", "importance": "nice_to_have", "aliases": []},
        {"name": "Google Cloud Platform", "category": "tool", "importance": "preferred", "aliases": ["gcp"]},
    ]
    by_name = {m["name"]: m for m in matching.match_requirements(requirements, resume_strong)}

    assert by_name["Apache Airflow"]["status"] == "match"
    assert by_name["Spark"]["status"] == "match"
    assert by_name["Kafka"]["status"] == "missing"
    # "Cloud" aparece suelto en el CV, pero no "Google Cloud Platform".
    assert by_name["Google Cloud Platform"]["status"] in ("partial", "missing")


def test_match_returns_literal_evidence(resume_strong):
    requirements = [{"name": "dbt", "category": "tool", "importance": "required", "aliases": []}]
    match = matching.match_requirements(requirements, resume_strong)[0]
    assert match["status"] == "match"
    assert "dbt" in match["evidence"].lower()


def test_keyword_coverage(resume_strong):
    coverage = matching.keyword_coverage(["Python", "SQL", "Kafka", "Snowflake"], resume_strong)
    assert set(coverage["present"]) == {"Python", "SQL"}
    assert coverage["ratio"] == 0.5


def test_detect_sections(resume_strong, resume_weak):
    strong = matching.detect_sections(resume_strong)["found"]
    assert {"experiencia", "educacion", "skills"} <= set(strong)

    weak = matching.detect_sections(resume_weak)["found"]
    assert "experiencia" not in weak  # usa encabezados inventados


def test_bullet_stats_counts_metrics(resume_strong):
    stats = matching.bullet_stats(resume_strong)
    assert stats["bullets"] >= 7
    assert stats["quantified"] >= 4
    assert stats["action_verbs"] >= 4


def test_required_years_from_offer(job_description):
    assert matching.required_years(job_description) == 5


def test_estimate_years_from_date_ranges(resume_strong, resume_weak):
    assert matching.estimate_years(resume_strong) >= 8
    assert matching.estimate_years(resume_weak) >= 5


def test_heuristic_job_profile_marks_importance(job_description):
    profile = matching.heuristic_job_profile(job_description)
    by_name = {r["name"]: r for r in profile["requirements"]}

    assert "airflow" in by_name
    assert by_name["airflow"]["importance"] == "required"
    assert by_name["kafka"]["importance"] == "nice_to_have"
    assert by_name["terraform"]["importance"] == "nice_to_have"
    assert profile["language"] == "es"


def test_contact_signals(resume_strong, resume_weak):
    strong = matching.contact_signals(resume_strong)
    assert strong["email"] and strong["phone"] and strong["linkedin"]

    weak = matching.contact_signals(resume_weak)
    assert weak["email"] and not weak["phone"]
