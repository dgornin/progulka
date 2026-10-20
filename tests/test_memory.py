"""Long-term memory extraction: what the model proposes, code trims and deduplicates."""

import asyncio
import json

import pytest

from app.agent import memory
from app.agent.memory import extract_memory
from app.agent.prompts import MEMORY
from app.llm import LLMError
from tests.test_agent import FakeLLM


async def test_only_new_short_facts_are_kept() -> None:
    proposed = [
        "Не любит музеи.",
        "  вегетарианец ",  # known, in other case
        "любит   кофе третьей волны",
        "Любит кофе третьей волны!",  # the same fact twice
        "пересказ запроса " * 10,  # too long to be a fact
        "",
    ]
    llm = FakeLLM(structured=[{"facts": proposed}])

    facts = await extract_memory(llm, ["Терпеть не могу музеи", "  "], known=["Вегетарианец"])

    assert facts == ["Не любит музеи", "любит кофе третьей волны"]
    system, user = llm.seen[0]
    assert system["content"] == MEMORY
    # The traveller's words go in as JSON data next to what is already known.
    assert json.loads(user["content"]) == {
        "known_facts": ["Вегетарианец"],
        "messages": ["Терпеть не могу музеи"],
    }


async def test_at_most_three_facts_per_run() -> None:
    llm = FakeLLM(structured=[{"facts": ["раз", "два", "три", "четыре"]}])
    assert await extract_memory(llm, ["много о себе"], known=[]) == ["раз", "два", "три"]


async def test_nothing_durable_and_provider_failure_learn_nothing() -> None:
    assert await extract_memory(FakeLLM(structured=[{"facts": []}]), ["Казань, суббота"], []) == []
    failing = FakeLLM(structured=[LLMError("rate limited", 429)])
    assert await extract_memory(failing, ["я вегетарианец"], []) == []


async def test_a_hung_provider_does_not_hold_back_the_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowLLM:
        async def structured(self, messages, schema, **_):
            await asyncio.sleep(5)

    monkeypatch.setattr(memory, "MEMORY_TIMEOUT_S", 0.01)
    assert await extract_memory(SlowLLM(), ["я вегетарианец"], []) == []


async def test_no_text_makes_no_call() -> None:
    llm = FakeLLM()
    assert await extract_memory(llm, ["", "   "], known=["вегетарианец"]) == []
    assert llm.seen == []
