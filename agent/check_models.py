"""Smoke-check models through our own provider code: one short prompt, no retries.

Usage: uv run python -m agent.check_models [model ...]
"""

import sys
import time

from agent.providers import GEMINI_MODEL, GeminiChat

PROMPT = "Hola, responde en una frase."


def check(model: str) -> str:
    start = time.perf_counter()
    try:
        chat = GeminiChat(system="Responde en español.", model=model, attempts=1)
        turn = chat.send(PROMPT)
    except Exception as error:
        result = f"ERROR {type(error).__name__}: {str(error)[:60]}"
    else:
        result = f"{turn.input_tokens}+{turn.output_tokens} tokens {turn.text[:40]!r}"
    return f"{model:24} {time.perf_counter() - start:5.1f}s {result}"


def main() -> None:
    for model in sys.argv[1:] or [GEMINI_MODEL]:
        print(check(model))


if __name__ == "__main__":
    main()
