from google.genai import types

from agent.providers import (
    ModelTurn,
    ToolCall,
    ToolResult,
    function_responses,
    gemini_config,
    to_model_turn,
)
from agent.retrieval import TfidfIndex
from agent.tools import make_search_kb


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


SEARCH = ToolCall(id="call-1", name="search_kb", arguments={"query": "agua"})


def test_to_model_turn_reads_tool_calls() -> None:
    call = types.FunctionCall(id="call-1", name="search_kb", args={"query": "agua"})
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part(function_call=call)]
                )
            )
        ]
    )

    turn = to_model_turn(response)

    assert turn.tool_calls == (SEARCH,)
    assert turn.text == ""


def test_gemini_config_declares_tools_and_disables_auto_calling() -> None:
    search_kb = make_search_kb(TfidfIndex([]))

    config = gemini_config("Eres un asistente.", [search_kb])

    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "search_kb"
    assert declaration.parameters_json_schema == search_kb.parameters
    assert config.automatic_function_calling.disable is True


def test_function_responses_keep_the_call_id() -> None:
    result = ToolResult(call=SEARCH, output={"articles": []})

    [part] = function_responses([result])

    assert part.function_response.id == "call-1"
    assert part.function_response.response == {"articles": []}
