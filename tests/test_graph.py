"""El grafo completo en modo determinista (sin LLM ni red)."""

from __future__ import annotations

from ats import graph as graph_mod
from ats import pipeline


def _analyze(text: str, filename: str, job_description: str, visited=None):
    return pipeline.analyze(
        text.encode("utf-8"),
        filename,
        job_description,
        chain=None,
        use_llm=False,
        on_node=(visited.append if visited is not None else None),
    )


def test_graph_runs_offline_and_visits_every_node(resume_strong, job_description):
    visited = []
    state = _analyze(resume_strong, "cv_a.txt", job_description, visited)

    assert set(visited) == set(graph_mod.NODE_DESCRIPTIONS)
    assert 0 < state["scores"]["total"] <= 100
    assert state["usage"] == []  # sin LLM no hay llamadas ni coste
    assert state["recommendations"]["_source"] == "heuristic"


def test_parallel_nodes_accumulate_warnings(job_description):
    state = _analyze("", "vacio.txt", job_description)
    # El reducer de `warnings` permite que varios nodos escriban sin pisarse.
    assert len(state["warnings"]) >= 2
    assert state["scores"]["total"] < 40


def test_stronger_candidate_ranks_higher(resume_strong, resume_weak, job_description):
    strong = _analyze(resume_strong, "cv_a.txt", job_description)
    weak = _analyze(resume_weak, "cv_b.txt", job_description)
    assert strong["scores"]["total"] > weak["scores"]["total"] + 15


def test_compare_reuses_job_profile_and_sorts(resume_strong, resume_weak, job_description):
    results = pipeline.compare(
        [("cv_b.txt", resume_weak.encode()), ("cv_a.txt", resume_strong.encode())],
        job_description,
        chain=None,
        use_llm=False,
    )
    assert [r["resume_filename"] for r in results] == ["cv_a.txt", "cv_b.txt"]
    # Mismo perfil de oferta para todos: los requisitos evaluados son idénticos.
    assert [m["name"] for m in results[0]["matches"]] == [
        m["name"] for m in results[1]["matches"]
    ]


def test_summary_row_shape(resume_strong, job_description):
    state = _analyze(resume_strong, "cv_a.txt", job_description)
    row = pipeline.summary_row(state)
    assert row["CV"] == "cv_a.txt"
    assert row["Score"] == state["scores"]["total"]
    assert "Requisitos excluyentes sin cubrir" in row


def test_graph_structure_has_parallel_fanout():
    structure = graph_mod.graph_structure(graph_mod.build_graph(None))
    fanout = [t for s, t in structure["edges"] if s == "ingest_resume"]
    assert set(fanout) == {"profile_job", "profile_resume", "audit_format"}


def test_unsupported_file_degrades_gracefully(job_description):
    state = _analyze("contenido", "cv.rtf", job_description)
    assert state["scores"]["total"] >= 0
    assert any("no soportada" in w for w in state["warnings"])
