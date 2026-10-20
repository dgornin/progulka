"""Long-term memory: durable facts about the traveller learned from what they write.

Two kinds of memory (lecture 07). Working memory is a day's ResearchState: the
places found, the shortlist, the rejected stops; it lives with the trip and dies with
it. Long-term memory is a short list of facts about the person ("не любит музеи",
"гуляет с ребёнком 8 лет") that outlives the trip and shapes every next one.

What is written, and when: after a successful run, one structured LLM call reads only
the traveller's own words of that run (the request, or the feedback) and returns at
most three NEW durable facts: never trip details (city, dates, this route), never what
the profile or the memory already says. Place texts from the sources are never read
here, so a web page cannot plant a "fact". Code then trims, drops the duplicates the
model missed and caps the count. Memory is best-effort: a provider failure, or an
answer slower than MEMORY_TIMEOUT_S, learns nothing instead of failing the walk. The
call runs before the finished plan is sent, so a hung provider must not hold it back
for the minutes the LLM client's retries can take.

Facts are written without asking: the traveller sees them in the profile and can
delete any of them, so a wrong fact costs one click. We prefer reversibility to
approval (lecture 04): a confirmation dialog after every walk would be ignored.

How memory reaches the context: the caller loads the facts and passes them to the
pipeline, which puts them into the research task and the writer's input, marked as
data about the traveller, never as instructions.
"""

import asyncio
import json
import re

from pydantic import BaseModel, Field

from app.agent.prompts import MEMORY
from app.llm import LLM, LLMError

MAX_NEW_FACTS = 3
MAX_FACT_CHARS = 120
MEMORY_TIMEOUT_S = 20


class LearnedFacts(BaseModel):
    facts: list[str] = Field(
        default_factory=list,
        description="0-3 new durable facts about the traveller, short phrases in Russian",
    )


async def extract_memory(llm: LLM, texts: list[str], known: list[str]) -> list[str]:
    """New durable facts from the traveller's texts; [] when there are none or on failure."""
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return []
    payload = {"known_facts": known, "messages": texts}
    messages = [
        {"role": "system", "content": MEMORY},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        async with asyncio.timeout(MEMORY_TIMEOUT_S):
            learned = await llm.structured(messages, LearnedFacts, max_tokens=400)
    except (LLMError, TimeoutError):
        return []

    seen = {fact_key(fact) for fact in known}
    facts = []
    for fact in learned.facts:
        fact = " ".join(fact.split()).rstrip(".")
        # A long "fact" is a retelling of the request, which is exactly what not to keep.
        if not fact or len(fact) > MAX_FACT_CHARS or fact_key(fact) in seen:
            continue
        seen.add(fact_key(fact))
        facts.append(fact)
    return facts[:MAX_NEW_FACTS]


def fact_key(fact: str) -> str:
    """Compare facts by their words only: case and punctuation do not make a new fact."""
    return " ".join(re.findall(r"\w+", fact.casefold()))
