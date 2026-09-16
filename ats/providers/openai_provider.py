"""Proveedores OpenAI y cualquier endpoint compatible con su API.

`OpenAIProvider` usa la Responses API (`client.responses.parse`).
`OpenAICompatibleProvider` usa Chat Completions (`client.chat.completions.parse`),
que es la superficie que implementan los servicios compatibles: OpenRouter,
Groq, Together, DeepSeek, vLLM, Ollama…
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from .base import LLMError, LLMProvider, TransientLLMError

MAX_OUTPUT_TOKENS = 16_000


class OpenAIProvider(LLMProvider):
    name = "openai"
    label = "OpenAI"
    default_model = "gpt-5.5"
    env_key = "OPENAI_API_KEY"
    env_model = "OPENAI_MODEL"
    env_base_url = "OPENAI_BASE_URL"
    package = "openai"

    #: `reasoning.effort` admite los mismos niveles que usa la app.
    EFFORT_MAP = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
    #: Los modelos sin razonamiento rechazan el parámetro `reasoning`.
    use_reasoning = True

    def client(self) -> Any:
        if self._client is None:
            import openai

            kwargs: Dict[str, Any] = {"api_key": self.api_key or "not-needed"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _call(
        self, *, system: str, user: str, output_format: Type[BaseModel], effort: str
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        client = self.client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "text_format": output_format,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        if self.use_reasoning:
            kwargs["reasoning"] = {"effort": self.EFFORT_MAP.get(effort, "medium")}

        try:
            response = client.responses.parse(**kwargs)
        except TypeError:
            # Un endpoint o SDK que no acepte `reasoning` no debe tumbar la llamada.
            kwargs.pop("reasoning", None)
            try:
                response = client.responses.parse(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise self._translate(exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc

        return self._read(response)

    def _read(self, response: Any) -> Tuple[BaseModel, Dict[str, Any]]:
        status = getattr(response, "status", None)
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "desconocido")
            raise LLMError(f"Respuesta incompleta ({reason}).")

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMError("La respuesta no contenía salida estructurada válida.")

        usage = getattr(response, "usage", None)
        return parsed, {
            "model": getattr(response, "model", self.model),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_tokens": int(
                getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
            ),
        }

    def _translate(self, exc: Exception) -> LLMError:
        import openai

        if isinstance(exc, openai.AuthenticationError):
            return LLMError(f"Credenciales inválidas ({self.env_key}).")
        if isinstance(exc, openai.PermissionDeniedError):
            return LLMError("La API key no tiene permiso para usar este modelo.")
        if isinstance(exc, openai.NotFoundError):
            return LLMError(f"Modelo no encontrado: {self.model}.")
        if isinstance(exc, openai.RateLimitError):
            return TransientLLMError("Límite de rate o cuota agotada.")
        if isinstance(exc, openai.BadRequestError):
            return LLMError(f"Petición rechazada: {exc}")
        if isinstance(exc, openai.APITimeoutError):
            return TransientLLMError("Tiempo de espera agotado.")
        if isinstance(exc, openai.APIConnectionError):
            return TransientLLMError(
                f"No se pudo conectar con {self.base_url or 'la API de OpenAI'}."
            )
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 0) or 0
            message = f"Error HTTP {status or '?'}: {exc}"
            return TransientLLMError(message) if status >= 500 else LLMError(message)
        return LLMError(f"Error inesperado: {type(exc).__name__}: {exc}")


class OpenAICompatibleProvider(OpenAIProvider):
    """Endpoint compatible con OpenAI: OpenRouter, Groq, Ollama, vLLM…

    Usa Chat Completions porque la Responses API todavía no es universal fuera
    de OpenAI, y no envía `reasoning` salvo que se active explícitamente.
    """

    name = "custom"
    label = "Compatible OpenAI"
    default_model = ""
    env_key = "CUSTOM_API_KEY"
    env_model = "CUSTOM_MODEL"
    env_base_url = "CUSTOM_BASE_URL"
    use_reasoning = False

    def availability(self) -> Tuple[bool, str]:
        if not self.installed():
            return False, f"falta el paquete `{self.package}`"
        if not self.base_url:
            return False, f"falta {self.env_base_url}"
        if not self.model:
            return False, f"falta {self.env_model}"
        return True, ""

    def _call(
        self, *, system: str, user: str, output_format: Type[BaseModel], effort: str
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        client = self.client()
        try:
            response = client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=output_format,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise LLMError("La respuesta no traía ninguna alternativa.")
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError("Respuesta truncada por límite de tokens.")

        parsed = getattr(choice.message, "parsed", None)
        if parsed is None:
            raise LLMError(
                "El endpoint no devolvió salida estructurada válida. "
                "Comprueba que el modelo soporta `response_format` con JSON Schema."
            )

        usage = getattr(response, "usage", None)
        return parsed, {
            "model": getattr(response, "model", self.model),
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cached_tokens": 0,
        }
