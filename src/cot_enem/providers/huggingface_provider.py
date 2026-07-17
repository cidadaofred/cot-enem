"""Optional local/Colab Hugging Face text-generation provider."""

import json
from typing import Any

from cot_enem.providers.base import LLMProvider, LLMResponse, Message
from cot_enem.providers.errors import ProviderConfigurationError
from cot_enem.providers.structured import parse_json_object, require_schema_keys


class HuggingFaceProvider(LLMProvider):
    """Lazy `transformers.pipeline` adapter; suitable for later GPU experiments."""

    def __init__(
        self,
        model: str,
        *,
        pipeline_instance: Any | None = None,
        device: str | int | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = pipeline_instance

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ProviderConfigurationError(
                    "install the optional 'huggingface' dependencies to use this provider"
                ) from exc
            self._pipeline = pipeline("text-generation", model=self.model, device=self.device)
        return self._pipeline

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if response_schema:
            prompt += "\nReturn only a JSON object matching this schema:\n"
            prompt += json.dumps(response_schema, ensure_ascii=False)
        output = self._get_pipeline()(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            return_full_text=False,
        )
        content = output[0]["generated_text"]
        parsed = parse_json_object(content) if response_schema is not None else None
        if parsed is not None:
            require_schema_keys(parsed, response_schema)
        return LLMResponse(content=content, parsed=parsed, model=self.model)
