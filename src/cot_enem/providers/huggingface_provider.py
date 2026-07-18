"""Optional local/Colab Hugging Face text-generation provider."""

import json
from typing import Any

from cot_enem.providers.base import LLMProvider, LLMResponse, Message
from cot_enem.providers.errors import ProviderConfigurationError, StructuredResponseError
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
        max_format_attempts: int = 3,
    ) -> None:
        self.model = model
        self.device = device
        self.precision = precision
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.max_format_attempts = max_format_attempts
        self._pipeline = pipeline_instance

    def unload(self) -> None:
        """Release model references before loading the next sequential Colab judge."""

        self._pipeline = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

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
        last_error: StructuredResponseError | None = None
        for attempt in range(1, self.max_format_attempts + 1):
            tokenizer = getattr(generation_pipeline, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as exc:
                    if len(conversation) < 2 or conversation[0]["role"] != "system":
                        raise ProviderConfigurationError(
                            f"model chat template rejected the conversation: {self.model}"
                        ) from exc
                    compatible_conversation = [
                        {
                            "role": "user",
                            "content": (
                                conversation[0]["content"]
                                + "\n\n"
                                + conversation[1]["content"]
                            ),
                        },
                        *conversation[2:],
                    ]
                    prompt = tokenizer.apply_chat_template(
                        compatible_conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                prompt = "\n".join(
                    f"{message['role']}: {message['content']}" for message in conversation
                )
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": temperature > 0,
                "return_full_text": False,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            output = generation_pipeline(prompt, **generation_kwargs)
            content = output[0]["generated_text"]
            try:
                parsed = parse_json_object(content) if response_schema is not None else None
                if parsed is not None:
                    parsed = self._normalize_judge_response(parsed, response_schema)
                    require_schema_keys(parsed, response_schema)
                return LLMResponse(content=content, parsed=parsed, model=self.model)
            except StructuredResponseError as exc:
                last_error = exc
                if attempt == self.max_format_attempts:
                    break
                conversation.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                f"A resposta anterior foi rejeitada: {exc}. Corrija-a e "
                                "responda somente com o objeto JSON exigido pelo schema. "
                                "Para julgamentos, use exatamente o formato "
                                '{"approved": true, "reasons": ["justificativa"]}. '
                                "Escape barras invertidas dentro de strings como \\\\ e não "
                                "use blocos Markdown."
                            ),
                        },
                    ]
                )
        preview = content[:200].replace("\n", " ") if "content" in locals() else ""
        raise StructuredResponseError(
            f"{self.model} did not return the required structured JSON; "
            f"last_error={last_error}; preview={preview!r}"
        )

    @staticmethod
    def _normalize_judge_response(
        value: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Map common binary-judge aliases to the canonical majority-vote contract."""

        required = set(schema.get("required", []))
        if required != {"approved", "reasons"}:
            return value
        normalized = dict(value)
        if "approved" not in normalized:
            decision_keys = (
                "answer",
                "verdict",
                "decision",
                "result",
                "is_correct",
                "correct",
                "success",
                "evolution_success",
                "correctness_verified",
            )
            decision = next(
                (normalized[key] for key in decision_keys if key in normalized),
                None,
            )
            if isinstance(decision, bool):
                normalized["approved"] = decision
            elif isinstance(decision, str):
                token = decision.strip().casefold().rstrip(".")
                if token in {"yes", "sim", "true", "approved", "correct", "pass"}:
                    normalized["approved"] = True
                elif token in {"no", "não", "nao", "false", "rejected", "incorrect", "fail"}:
                    normalized["approved"] = False
        if "reasons" not in normalized and "approved" in normalized:
            reason_keys = ("reason", "explanation", "justification", "rationale")
            reason = next(
                (normalized[key] for key in reason_keys if key in normalized),
                None,
            )
            if isinstance(reason, list):
                normalized["reasons"] = [str(item) for item in reason]
            elif reason is None:
                normalized["reasons"] = []
            else:
                normalized["reasons"] = [str(reason)]
        return normalized
