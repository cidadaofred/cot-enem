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
        precision: str = "auto",
        quantization: str = "none",
        max_new_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.device = device
        self.precision = precision
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self._pipeline = pipeline_instance

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                    pipeline,
                )
            except ImportError as exc:
                raise ProviderConfigurationError(
                    "install the optional 'huggingface' dependencies to use this provider"
                ) from exc
            if self.quantization != "none" and self.device != "cuda":
                raise ProviderConfigurationError(
                    "4-bit/8-bit quantization requires a CUDA runtime in this project"
                )
            dtype_by_name = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            dtype = dtype_by_name.get(self.precision)
            model_kwargs: dict[str, Any] = {}
            if dtype is not None:
                model_kwargs["dtype"] = dtype
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            if self.quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype or torch.float16,
                )
            elif self.quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
            pipeline_kwargs: dict[str, Any] = {
                "task": "text-generation",
                "model": model,
                "tokenizer": tokenizer,
            }
            if self.device != "cuda":
                pipeline_kwargs["device"] = self.device
            self._pipeline = pipeline(**pipeline_kwargs)
        return self._pipeline

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        conversation = [dict(message) for message in messages]
        if response_schema:
            schema_instruction = (
                "\nResponda somente com um objeto JSON válido, sem Markdown ou texto adicional, "
                "seguindo este JSON Schema:\n"
                + json.dumps(response_schema, ensure_ascii=False)
            )
            conversation[-1]["content"] += schema_instruction
        generation_pipeline = self._get_pipeline()
        tokenizer = getattr(generation_pipeline, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{message['role']}: {message['content']}" for message in conversation
            )
        output = generation_pipeline(
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
