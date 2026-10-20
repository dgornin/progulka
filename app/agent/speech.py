"""Voice input: a recorded message -> its transcript, by an audio-capable LLM.

Why on the server: the browser's Web Speech API works only in Chrome and Safari (and
sends the audio to Google or Apple anyway); Yandex Browser and Firefox have no working
recognizer. So the browser only records and converts the message to 16 kHz mono WAV,
which every audio model accepts and which keeps a minute under 2 MB, and the server
sends it to an audio-capable model (STT_MODEL) through the same OpenAI-compatible
client: one more model, not one more provider or key, and its cost is counted with
the traveller's other LLM calls.

The transcript only fills the text box: the traveller reads and edits it before
sending, so a misheard word costs a correction, not a wrong trip.
"""

import asyncio

from app.agent.prompts import TRANSCRIBE
from app.config import settings
from app.llm import LLM, LLMError
from app.pipeline import PipelineError

STT_TIMEOUT_S = 30
# A minute of speech is about 150 words; the cap only stops a model that rambles on.
MAX_TRANSCRIPT_TOKENS = 1000


async def transcribe(llm: LLM, audio_wav_b64: str) -> tuple[str, float]:
    """The transcript and its cost in USD; PipelineError when the model fails or is slow."""
    spent = llm.usage.cost_usd
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIBE},
                {"type": "input_audio", "input_audio": {"data": audio_wav_b64, "format": "wav"}},
            ],
        }
    ]
    try:
        async with asyncio.timeout(STT_TIMEOUT_S):
            turn = await llm.chat(
                messages,
                temperature=0,
                max_tokens=MAX_TRANSCRIPT_TOKENS,
                model=settings.stt_model,
            )
    except TimeoutError as e:
        raise PipelineError("Распознавание речи не ответило вовремя, попробуйте ещё раз") from e
    except LLMError as e:
        raise PipelineError(f"Не удалось распознать речь: {e}") from e
    return (turn.content or "").strip(), llm.usage.cost_usd - spent
