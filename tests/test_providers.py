"""Cadena de proveedores: orden, fallback y contrato común.

Los proveedores reales se sustituyen por dobles, así que no hay red ni claves.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

import pytest
from pydantic import BaseModel

from ats import graph as graph_mod
from ats import pipeline, providers
from ats.providers import (
    AllProvidersFailed,
    LLMError,
    LLMProvider,
    ProviderChain,
    ProviderUnavailable,
    TransientLLMError,
)
from ats.schemas import JobProfile, RecommendationSet, ResumeProfile, SemanticMatchResult


class Echo(BaseModel):
    value: str


class FakeProvider(LLMProvider):
    """Proveedor de prueba: responde, falla o se declara no disponible."""

    name = "fake"
    label = "Fake"
    default_model = "fake-1"
    env_key = ""
    package = "pytest"
    price_per_mtok = (1.0, 2.0)

    def __init__(self, *, fail=None, unavailable=False, name=None, fail_times=0, **kwargs):
        super().__init__(**kwargs)
        self._fail = fail
        self._fail_times = fail_times  # falla las N primeras llamadas y luego responde
        self._unavailable = unavailable
        self.calls = 0
        if name:
            self.name = name
            self.label = name

    def availability(self) -> Tuple[bool, str]:
        return (False, "de prueba") if self._unavailable else (True, "")

    def _call(self, *, system, user, output_format, effort):
        self.calls += 1
        if self._fail_times and self.calls <= self._fail_times:
            raise TransientLLMError("503 temporal")
        if self._fail:
            raise self._fail
        return output_format(value="ok"), {"input_tokens": 1000, "output_tokens": 500}


# --- Contrato de la interfaz ---------------------------------------------


@pytest.mark.parametrize("cls", list(providers.PROVIDERS.values()))
def test_every_provider_implements_the_interface(cls: Type[LLMProvider]):
    assert issubclass(cls, LLMProvider)
    assert cls.name and cls.label and cls.package
    provider = cls()
    # Sin credenciales debe declararse no disponible, nunca reventar.
    available, reason = provider.availability()
    assert isinstance(available, bool)
    assert available or reason


@pytest.mark.parametrize("cls", list(providers.PROVIDERS.values()))
def test_every_provider_maps_all_effort_levels(cls: Type[LLMProvider]):
    mapping = getattr(cls, "EFFORT_MAP", {})
    assert set(providers.base.CANONICAL_EFFORTS) <= set(mapping)


def test_unavailable_provider_raises_before_calling():
    provider = FakeProvider(unavailable=True)
    with pytest.raises(ProviderUnavailable):
        provider.structured(system="s", user="u", output_format=Echo)
    assert provider.calls == 0


def test_usage_record_carries_provider_and_cost():
    provider = FakeProvider()
    result = provider.structured(system="s", user="u", output_format=Echo, label="nodo")
    assert result.parsed.value == "ok"
    assert result.usage["provider"] == "fake"
    assert result.usage["node"] == "nodo"
    assert result.usage["model"] == "fake-1"
    # 1000 in * 1$/M + 500 out * 2$/M
    assert result.usage["cost_usd"] == pytest.approx(0.002)


def test_cost_is_none_without_known_price(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(type(provider), "price_per_mtok", None)
    result = provider.structured(system="s", user="u", output_format=Echo)
    assert result.usage["cost_usd"] is None


def test_price_override_from_environment(monkeypatch):
    monkeypatch.setenv("FAKE_PRICE_IN", "10")
    monkeypatch.setenv("FAKE_PRICE_OUT", "20")
    result = FakeProvider().structured(system="s", user="u", output_format=Echo)
    assert result.usage["cost_usd"] == pytest.approx(0.02)


# --- Reintentos ante fallos transitorios ---------------------------------


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Sin esperas reales entre reintentos."""
    monkeypatch.setenv("LLM_RETRY_DELAY", "0")


def test_transient_error_is_retried_and_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    provider = FakeProvider(fail_times=2)
    result = provider.structured(system="s", user="u", output_format=Echo)
    assert provider.calls == 3  # dos 503 y a la tercera responde
    assert result.usage["retries"] == 2


def test_default_policy_is_one_retry(monkeypatch):
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    provider = FakeProvider(fail_times=1)
    provider.structured(system="s", user="u", output_format=Echo)
    assert provider.calls == 2


def test_transient_error_gives_up_after_the_limit(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    provider = FakeProvider(fail_times=5)
    with pytest.raises(TransientLLMError):
        provider.structured(system="s", user="u", output_format=Echo)
    assert provider.calls == 2  # intento inicial + un reintento


def test_permanent_error_is_not_retried():
    provider = FakeProvider(fail=LLMError("modelo no encontrado"))
    with pytest.raises(LLMError):
        provider.structured(system="s", user="u", output_format=Echo)
    assert provider.calls == 1


def test_retries_are_configurable(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    provider = FakeProvider(fail_times=1)
    with pytest.raises(TransientLLMError):
        provider.structured(system="s", user="u", output_format=Echo)
    assert provider.calls == 1


def test_chain_moves_on_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    flaky = FakeProvider(name="inestable", fail_times=9)
    healthy = FakeProvider(name="sano")
    result = ProviderChain([flaky, healthy]).structured(
        system="s", user="u", output_format=Echo
    )
    assert flaky.calls == 2
    assert result.usage["provider"] == "sano"


# --- Cadena de fallback ---------------------------------------------------


def test_chain_uses_first_working_provider():
    first, second = FakeProvider(name="uno"), FakeProvider(name="dos")
    result = ProviderChain([first, second]).structured(
        system="s", user="u", output_format=Echo
    )
    assert (first.calls, second.calls) == (1, 0)
    assert result.usage["attempt"] == 1
    assert result.usage["fallback_from"] == []


def test_chain_falls_back_on_error():
    broken = FakeProvider(name="roto", fail=LLMError("429 rate limit"))
    healthy = FakeProvider(name="sano")
    result = ProviderChain([broken, healthy]).structured(
        system="s", user="u", output_format=Echo
    )
    assert (broken.calls, healthy.calls) == (1, 1)
    assert result.usage["provider"] == "sano"
    assert result.usage["attempt"] == 2
    assert result.usage["fallback_from"] == ["roto (fake-1)"]


def test_chain_skips_unavailable_provider():
    missing = FakeProvider(name="sin_clave", unavailable=True)
    healthy = FakeProvider(name="sano")
    result = ProviderChain([missing, healthy]).structured(
        system="s", user="u", output_format=Echo
    )
    assert result.usage["provider"] == "sano"


def test_chain_raises_when_everything_fails():
    chain = ProviderChain(
        [
            FakeProvider(name="a", fail=LLMError("caído")),
            FakeProvider(name="b", fail=LLMError("sin cuota")),
        ]
    )
    with pytest.raises(AllProvidersFailed) as excinfo:
        chain.structured(system="s", user="u", output_format=Echo)
    assert "caído" in str(excinfo.value) and "sin cuota" in str(excinfo.value)


def test_empty_chain_is_falsy_and_raises():
    chain = ProviderChain([])
    assert not chain
    with pytest.raises(AllProvidersFailed):
        chain.structured(system="s", user="u", output_format=Echo)


# --- Configuración por entorno -------------------------------------------


def test_chain_order_defaults_to_anthropic_first(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACKS", raising=False)
    order = providers.chain_order()
    assert order[0] == "anthropic"
    # Sin configurar respaldos, el resto entra automáticamente.
    assert set(order) == set(providers.PROVIDERS)


def test_chain_order_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACKS", "openai, anthropic")
    assert providers.chain_order() == ["gemini", "openai", "anthropic"]


def test_unknown_provider_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "inventado")
    monkeypatch.setenv("LLM_FALLBACKS", "")
    assert providers.chain_order() == ["anthropic"]


def test_arguments_win_over_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert providers.chain_order("openai", ["anthropic"]) == ["openai", "anthropic"]


def test_build_chain_drops_providers_without_credentials(monkeypatch):
    for row in providers.inventory():
        if row["env_key"]:
            monkeypatch.delenv(row["env_key"], raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert providers.build_chain("anthropic", ["openai", "gemini"]).names == []


def test_build_chain_keeps_provider_with_key(monkeypatch):
    chain = providers.build_chain("openai", [], api_keys={"openai": "sk-test"})
    assert chain.names == ["openai"]
    assert "OpenAI" in chain.describe()


def test_inventory_reports_reason_when_unusable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    row = next(r for r in providers.inventory() if r["name"] == "openai")
    assert row["available"] is False
    assert "OPENAI_API_KEY" in row["reason"]


# --- Integración con el grafo --------------------------------------------


class ScriptedProvider(FakeProvider):
    """Devuelve una instancia mínima válida de cada esquema del grafo."""

    def _call(self, *, system, user, output_format, effort):
        self.calls += 1
        if self._fail:
            raise self._fail
        payloads: Dict[Any, Any] = {
            JobProfile: {
                "job_title": "Data Engineer",
                "seniority": "senior",
                "min_years_experience": 5,
                "requirements": [
                    {
                        "name": "Airflow",
                        "category": "tool",
                        "importance": "required",
                        "aliases": ["apache airflow"],
                    }
                ],
                "key_responsibilities": ["Pipelines"],
                "ats_keywords": ["Airflow", "Python"],
                "language": "es",
            },
            ResumeProfile: {
                "candidate_name": "Lucía",
                "current_title": "Senior Data Engineer",
                "total_years_experience": 7.0,
                "titles": ["Senior Data Engineer"],
                "companies": ["Envialia"],
                "skills": ["Python"],
                "tools": ["Airflow"],
                "education": ["Grado"],
                "certifications": [],
                "languages": ["Español"],
                "bullet_count": 10,
                "quantified_bullets": 6,
            },
            SemanticMatchResult: {"verdicts": []},
            RecommendationSet: {
                "verdict_summary": "Buen encaje.",
                "recommendations": [
                    {
                        "priority": "high",
                        "area": "keywords",
                        "issue": "Falta X",
                        "action": "Añade X",
                        "example": "",
                    }
                ],
                "keywords_to_add": ["Kafka"],
            },
        }
        return output_format(**payloads[output_format]), {
            "input_tokens": 2000,
            "output_tokens": 800,
        }


def test_graph_runs_through_the_chain(resume_strong, job_description):
    served = ScriptedProvider(name="secundario")
    chain = ProviderChain([ScriptedProvider(name="principal", fail=LLMError("503")), served])

    state = pipeline.analyze(
        resume_strong.encode(),
        "cv.txt",
        job_description,
        chain=chain,
        use_llm=True,
    )

    assert state["job_profile"]["_source"] == "llm"
    assert state["recommendations"]["_source"] == "llm"
    assert state["scores"]["total"] > 0
    # Un registro de uso por cada nodo con LLM, todos servidos por el respaldo.
    assert {u["provider"] for u in state["usage"]} == {"secundario"}
    assert served.calls == len(state["usage"])

    usage = pipeline.total_usage([state])
    assert usage["fallbacks_used"] == ["principal (fake-1)"]
    assert usage["providers"] == "secundario"


def test_graph_degrades_when_whole_chain_fails(resume_strong, job_description):
    chain = ProviderChain([ScriptedProvider(name="a", fail=LLMError("caído"))])
    state = pipeline.analyze(
        resume_strong.encode(), "cv.txt", job_description, chain=chain, use_llm=True
    )

    assert state["job_profile"]["_source"] == "heuristic"
    assert state["usage"] == []
    assert any("sin LLM" in w for w in state["warnings"])
    assert state["scores"]["total"] > 0  # el grafo termina igualmente


def test_graph_ignores_empty_chain(resume_strong, job_description):
    state = pipeline.analyze(
        resume_strong.encode(),
        "cv.txt",
        job_description,
        chain=ProviderChain([]),
        use_llm=True,
    )
    assert state["job_profile"]["_source"] == "heuristic"
    assert graph_mod.NODE_DESCRIPTIONS  # el grafo sigue siendo el mismo
