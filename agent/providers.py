"""LLM providers behind one small interface, so the eval grid can swap models.

The rest of the agent only sees `ChatSession` and `ModelTurn`, never an SDK type.
"""

from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-3.6-flash"
TIMEOUT_MS = 30_000
MAX_ATTEMPTS = 3  # the first try plus 2 retries on 429 and 5xx errors


@dataclass(frozen=True)
class ModelTurn:
    """One model reply, in a provider-neutral shape."""

    text: str
    input_tokens: int
    output_tokens: int


class ChatSession(Protocol):
    def send(self, text: str) -> ModelTurn: ...


class GeminiChat:
    def __init__(
        self, system: str, model: str = GEMINI_MODEL, attempts: int = MAX_ATTEMPTS
    ) -> None:
        retry = types.HttpRetryOptions(attempts=attempts, max_delay=10.0)
        options = types.HttpOptions(timeout=TIMEOUT_MS, retry_options=retry)
        self._client = genai.Client(
            http_options=options
        )  # Reads the GEMINI_API_KEY environment variable
        config = types.GenerateContentConfig(system_instruction=system)
        self._chat = self._client.chats.create(model=model, config=config)

    def send(self, text: str) -> ModelTurn:
        return to_model_turn(self._chat.send_message(text))


def to_model_turn(response: types.GenerateContentResponse) -> ModelTurn:
    usage = response.usage_metadata
    return ModelTurn(
        text=response.text or "",
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
    )
