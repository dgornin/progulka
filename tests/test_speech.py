"""Voice input: the audio goes to the speech model as an input_audio part, the
transcript and its cost come back, and a failure is an error the traveller can read."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from app.agent import speech
from app.agent.prompts import TRANSCRIBE
from app.agent.speech import transcribe
from app.llm import LLM, LLMError
from app.pipeline import PipelineError

AUDIO = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="  # an empty WAV


@pytest.fixture(autouse=True)
def stt_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speech, "settings", SimpleNamespace(stt_model="test/omni"))


def provider(sent: list[dict[str, Any]]) -> AsyncOpenAI:
    """An OpenAI-compatible endpoint that records the requests and answers one transcript."""

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test/omni",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "  Три дня в Казани\n"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 6,
                    "total_tokens": 406,
                    "cost": 0.00021,
                },
            },
        )

    return AsyncOpenAI(
        base_url="https://llm.test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )


async def test_the_audio_goes_to_the_speech_model_and_the_transcript_comes_back() -> None:
    sent: list[dict[str, Any]] = []
    llm = LLM(client=provider(sent), model="test/planner")

    text, cost = await transcribe(llm, AUDIO)

    assert (text, cost) == ("Три дня в Казани", 0.00021)
    [request] = sent
    assert (request["model"], request["temperature"]) == ("test/omni", 0)
    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIBE},
                {"type": "input_audio", "input_audio": {"data": AUDIO, "format": "wav"}},
            ],
        }
    ]
    # The run's counters include the call: it counts toward the daily budget.
    assert (llm.usage.calls, llm.model) == (1, "test/planner")


async def test_a_slow_or_failing_speech_model_is_an_error_the_traveller_can_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowLLM:
        usage = SimpleNamespace(cost_usd=0.0)

        async def chat(self, *_: Any, **__: Any) -> None:
            await asyncio.sleep(5)

    class FailingLLM(SlowLLM):
        async def chat(self, *_: Any, **__: Any) -> None:
            raise LLMError("No endpoints found that support input audio", 404)

    monkeypatch.setattr(speech, "STT_TIMEOUT_S", 0.01)

    with pytest.raises(PipelineError, match="не ответило вовремя"):
        await transcribe(SlowLLM(), AUDIO)
    with pytest.raises(PipelineError, match="Не удалось распознать речь: No endpoints"):
        await transcribe(FailingLLM(), AUDIO)
