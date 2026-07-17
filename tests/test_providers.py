import json

import pytest

from cot_enem.providers.errors import ProviderConfigurationError, StructuredResponseError
from cot_enem.providers.huggingface_provider import HuggingFaceProvider
from cot_enem.providers.mock_provider import MockLLMProvider
from cot_enem.providers.openai_compatible import OpenAICompatibleProvider
from cot_enem.providers.structured import parse_json_object
from cot_enem.utils.retry import retry_call

SCHEMA = {
    "type": "object",
    "properties": {"final_answer": {"type": "string"}},
    "required": ["final_answer"],
}
MESSAGES = [{"role": "user", "content": "Resolva."}]


class FakeHTTPResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.value).encode()


def test_mock_returns_structured_deterministic_response():
    provider = MockLLMProvider([{"final_answer": "C"}])
    response = provider.generate(MESSAGES, temperature=0.2, response_schema=SCHEMA)
    assert response.parsed == {"final_answer": "C"}
    assert provider.calls[0]["temperature"] == 0.2


def test_mock_rejects_missing_required_key():
    provider = MockLLMProvider([{"answer": "C"}])
    with pytest.raises(StructuredResponseError, match="missing keys"):
        provider.generate(MESSAGES, response_schema=SCHEMA)


def test_structured_parser_accepts_json_fence():
    assert parse_json_object('```json\n{"final_answer": "A"}\n```')["final_answer"] == "A"


def test_retry_uses_exponential_backoff():
    attempts, delays = [], []

    def unstable():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("temporary")
        return "ok"

    result = retry_call(
        unstable, max_attempts=3, initial_delay=0.25, sleep=delays.append, retry_on=(OSError,)
    )
    assert result == "ok"
    assert delays == [0.25, 0.5]


def test_openai_compatible_builds_request_without_leaking_key():
    observed = {}

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.headers["Authorization"]
        observed["payload"] = json.loads(request.data)
        observed["timeout"] = timeout
        return FakeHTTPResponse(
            {
                "model": "test-model",
                "choices": [
                    {
                        "message": {"content": '{"final_answer": "B"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
        )

    provider = OpenAICompatibleProvider(
        base_url="https://llm.example/v1",
        api_key="top-secret",
        model="test-model",
        opener=opener,
    )
    response = provider.generate(MESSAGES, response_schema=SCHEMA)
    assert observed["url"] == "https://llm.example/v1/chat/completions"
    assert observed["authorization"] == "Bearer top-secret"
    assert observed["payload"]["response_format"]["type"] == "json_schema"
    assert response.parsed == {"final_answer": "B"}
    assert response.usage["prompt_tokens"] == 10
    assert "top-secret" not in repr(provider)


def test_openai_provider_requires_configuration():
    with pytest.raises(ProviderConfigurationError, match="LLM_BASE_URL"):
        OpenAICompatibleProvider(base_url="", model="model")


def test_huggingface_provider_accepts_injected_pipeline():
    calls = []

    def pipeline(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return [{"generated_text": '{"final_answer": "D"}'}]

    provider = HuggingFaceProvider("local-model", pipeline_instance=pipeline)
    response = provider.generate(MESSAGES, response_schema=SCHEMA)
    assert response.parsed == {"final_answer": "D"}
    assert "user: Resolva." in calls[0][0]


def test_huggingface_provider_keeps_remote_runtime_options_lazy():
    provider = HuggingFaceProvider(
        "Qwen/Qwen2.5-7B-Instruct",
        pipeline_instance=lambda *_args, **_kwargs: [],
        device="cuda",
        precision="fp16",
        quantization="4bit",
        max_new_tokens=512,
    )
    assert provider.device == "cuda"
    assert provider.precision == "fp16"
    assert provider.quantization == "4bit"
    assert provider.max_new_tokens == 512
