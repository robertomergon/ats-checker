from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"


def _read(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def job_description() -> str:
    return _read("oferta_data_engineer.txt")


@pytest.fixture(scope="session")
def resume_strong() -> str:
    return _read("cv_candidata_a.txt")


@pytest.fixture(scope="session")
def resume_weak() -> str:
    return _read("cv_candidato_b.txt")
