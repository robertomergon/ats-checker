"""Interfaz común de proveedores LLM y cadena de fallback.

Todo proveedor expone la misma operación: dada una instrucción de sistema, un
mensaje de usuario y un modelo Pydantic, devuelve una instancia validada de ese
modelo más un registro de uso. Los nodos del grafo no saben qué vendor hay
detrás; la cadena decide y, si uno falla, pasa al siguiente.
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel


class LLMError(RuntimeError):
    """Fallo recuperable: la cadena prueba el siguiente proveedor."""


class ProviderUnavailable(LLMError):
    """El proveedor no se puede usar (falta el paquete o la clave)."""


class TransientLLMError(LLMError):
    """Fallo temporal (429, 503, 5xx, timeout): merece reintento."""


class AllProvidersFailed(LLMError):
    """Ningún proveedor de la cadena pudo responder."""


def retry_policy() -> Tuple[int, float]:
    """(reintentos, retardo base en segundos), configurables por entorno."""
    try:
        retries = int(os.environ.get("LLM_MAX_RETRIES", "1"))
    except ValueError:
        retries = 1
    try:
        delay = float(os.environ.get("LLM_RETRY_DELAY", "1.5"))
    except ValueError:
        delay = 1.5
    return max(0, retries), max(0.0, delay)


#: Niveles de esfuerzo de la app. Cada proveedor los traduce a lo suyo.
CANONICAL_EFFORTS: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


@dataclass
class LLMResult:
    """Salida estructurada más la telemetría de la llamada."""

    parsed: BaseModel
    usage: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Contrato que implementa cada vendor."""

    #: Identificador corto usado en `.env` y en la UI.
    name: str = ""
    #: Nombre legible.
    label: str = ""
    #: Modelo por defecto si no se configura otro.
    default_model: str = ""
    #: Variables de entorno que lee este proveedor.
    env_key: str = ""
    env_model: str = ""
    env_base_url: str = ""
    #: Paquete pip necesario, para el mensaje de error si falta.
    package: str = ""
    #: Precio por millón de tokens (entrada, salida). None = desconocido.
    price_per_mtok: Optional[Tuple[float, float]] = None

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model or os.environ.get(self.env_model, "") or self.default_model
        self.api_key = api_key or (os.environ.get(self.env_key, "") if self.env_key else "")
        self.base_url = base_url or (
            os.environ.get(self.env_base_url, "") if self.env_base_url else ""
        )
        self._client: Any = None

    # --- Disponibilidad ---------------------------------------------------

    @classmethod
    def installed(cls) -> bool:
        """¿Está instalado el SDK de este proveedor?"""
        import importlib.util

        root = cls.package.split("[")[0].replace("-", "_")
        return importlib.util.find_spec(root) is not None

    def has_credentials(self) -> bool:
        return bool(self.api_key)

    def availability(self) -> Tuple[bool, str]:
        """(disponible, motivo si no lo está)."""
        if not self.installed():
            return False, f"falta el paquete `{self.package}`"
        if not self.has_credentials():
            return False, f"falta {self.env_key}"
        return True, ""

    # --- Llamada ----------------------------------------------------------

    @abstractmethod
    def _call(
        self, *, system: str, user: str, output_format: Type[BaseModel], effort: str
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Llamada concreta al SDK. Devuelve (instancia validada, uso en bruto)."""

    def structured(
        self,
        *,
        system: str,
        user: str,
        output_format: Type[BaseModel],
        effort: str = "medium",
        label: str = "",
    ) -> LLMResult:
        available, reason = self.availability()
        if not available:
            raise ProviderUnavailable(f"{self.label} no disponible: {reason}")

        started = time.perf_counter()
        retries, base_delay = retry_policy()

        # Los 429/503 son casi siempre picos momentáneos: reintentar aquí sale
        # más barato en calidad que saltar de eslabón (o caer a la heurística).
        for attempt in range(retries + 1):
            try:
                parsed, usage = self._call(
                    system=system, user=user, output_format=output_format, effort=effort
                )
                break
            except TransientLLMError:
                if attempt == retries:
                    raise
                time.sleep(base_delay * (2**attempt) + random.uniform(0, 0.3))

        usage.update(
            {
                "node": label,
                "provider": self.name,
                "model": usage.get("model") or self.model,
                "effort": effort,
                "retries": attempt,
                "latency_s": round(time.perf_counter() - started, 2),
            }
        )
        usage["cost_usd"] = self.estimate_cost(usage)
        return LLMResult(parsed=parsed, usage=usage)

    # --- Coste ------------------------------------------------------------

    def prices(self) -> Optional[Tuple[float, float]]:
        """Precio por millón de tokens, con override por entorno.

        Las tarifas cambian a menudo y no todas son públicas de forma estable,
        así que solo se estima coste cuando hay tarifa conocida o configurada:
        `<PROVEEDOR>_PRICE_IN` / `<PROVEEDOR>_PRICE_OUT` en el `.env`.
        """
        prefix = self.name.upper()
        raw_in = os.environ.get(f"{prefix}_PRICE_IN")
        raw_out = os.environ.get(f"{prefix}_PRICE_OUT")
        if raw_in and raw_out:
            try:
                return float(raw_in), float(raw_out)
            except ValueError:
                return self.price_per_mtok
        return self.price_per_mtok

    def estimate_cost(self, usage: Dict[str, Any]) -> Optional[float]:
        prices = self.prices()
        if not prices:
            return None
        price_in, price_out = prices
        cost = (
            usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out
        ) / 1_000_000
        return round(cost, 5)

    # --- Utilidades para las subclases ------------------------------------

    def _describe(self) -> str:
        return f"{self.label} ({self.model})"

    def __repr__(self) -> str:  # pragma: no cover - ayuda en depuración
        return f"<{type(self).__name__} {self.model}>"


class ProviderChain:
    """Prueba los proveedores en orden hasta que uno responde."""

    def __init__(self, providers: List[LLMProvider]) -> None:
        self.providers = providers

    def __bool__(self) -> bool:
        return bool(self.providers)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.providers]

    def describe(self) -> str:
        return " → ".join(p._describe() for p in self.providers) or "sin proveedores"

    def structured(
        self,
        *,
        system: str,
        user: str,
        output_format: Type[BaseModel],
        effort: str = "medium",
        label: str = "",
    ) -> LLMResult:
        if not self.providers:
            raise AllProvidersFailed("No hay ningún proveedor LLM configurado.")

        # Cada fallo guarda (descriptor, motivo). El descriptor lleva el modelo
        # porque la cadena puede tener dos eslabones del mismo proveedor.
        failures: List[Tuple[str, str]] = []
        for provider in self.providers:
            try:
                result = provider.structured(
                    system=system,
                    user=user,
                    output_format=output_format,
                    effort=effort,
                    label=label,
                )
            except LLMError as exc:
                failures.append((provider._describe(), str(exc)))
                continue
            result.usage["fallback_from"] = [descriptor for descriptor, _ in failures]
            result.usage["attempt"] = len(failures) + 1
            return result

        raise AllProvidersFailed(
            "Todos los proveedores fallaron — "
            + " | ".join(f"{descriptor}: {reason}" for descriptor, reason in failures)
        )
