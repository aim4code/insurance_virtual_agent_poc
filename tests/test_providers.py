from google.genai import types

from agent.providers import ModelTurn, to_model_turn


def test_to_model_turn_reads_text_and_tokens() -> None:
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Hola")])
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=3
        ),
    )

    turn = to_model_turn(response)

    assert turn.text == "Hola"
    assert (turn.input_tokens, turn.output_tokens) == (12, 3)


def test_to_model_turn_handles_missing_usage() -> None:
    turn = to_model_turn(types.GenerateContentResponse())

    assert turn == ModelTurn(text="", input_tokens=0, output_tokens=0)
