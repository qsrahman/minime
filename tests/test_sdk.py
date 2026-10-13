"""Run the loop through the real OpenAI client against canned HTTP responses.

The other loop tests use stand-ins for responses and stream events; these catch
a change in the SDK itself (request shape, event names, response parsing).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest

import config
import main
import tools

USAGE = {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 7,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 127,
}


def message_item(text: str, item_id: str = "msg_1") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call_item(name: str, call_id: str = "call_1", **arguments: Any) -> dict[str, Any]:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def reasoning_item(item_id: str = "rs_1") -> dict[str, Any]:
    """What the API returns for a reasoning step when the request has store=False."""
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "gAAAA-opaque-reasoning-state",
    }


def response_body(
    *output: dict[str, Any], status: str = "completed", **extra: Any
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": "test-model",
        "output": list(output),
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "usage": USAGE,
        **extra,
    }


def stream_events(*output: dict[str, Any]) -> list[dict[str, Any]]:
    """The event sequence the API sends when streaming a response with this output."""
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": response_body(status="in_progress") | {"usage": None},
        }
    ]
    for index, item in enumerate(output):
        at = {"output_index": index, "item_id": item["id"]}
        if item["type"] == "message":
            text = item["content"][0]["text"]
            part = {"type": "output_text", "text": "", "annotations": []}
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item | {"content": [], "status": "in_progress"},
                },
                {
                    "type": "response.content_part.added",
                    **at,
                    "content_index": 0,
                    "part": part,
                },
                *(
                    {
                        "type": "response.output_text.delta",
                        **at,
                        "content_index": 0,
                        "delta": chunk,
                        "logprobs": [],
                    }
                    for chunk in (text[: len(text) // 2], text[len(text) // 2 :])
                ),
                {
                    "type": "response.output_text.done",
                    **at,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
                {
                    "type": "response.content_part.done",
                    **at,
                    "content_index": 0,
                    "part": item["content"][0],
                },
            ]
        elif item["type"] == "reasoning":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item | {"encrypted_content": None},
                }
            )
        else:
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item | {"arguments": "", "status": "in_progress"},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    **at,
                    "delta": item["arguments"],
                },
                {
                    "type": "response.function_call_arguments.done",
                    **at,
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            ]
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    events.append({"type": "response.completed", "response": response_body(*output)})
    return [event | {"sequence_number": n} for n, event in enumerate(events)]


class Api:
    """A fake Responses endpoint: replies from a script and records request bodies."""

    def __init__(self) -> None:
        self.replies: list[Any] = []
        self.requests: list[dict[str, Any]] = []
        self.client = openai.OpenAI(
            api_key="test-key",
            base_url="http://api.test/v1",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, httpx2.Response):
            return reply
        if body.get("stream"):
            lines = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in stream_events(*reply["output"])
            )
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, text=lines
            )
        return httpx2.Response(200, json=reply)


@pytest.fixture
def api() -> Api:
    return Api()


@pytest.fixture(params=[False, True], ids=["create", "stream"])
def streaming(request: pytest.FixtureRequest, settings: config.Config) -> bool:
    settings.streaming = request.param
    return request.param


def input_kinds(body: dict[str, Any]) -> list[str]:
    return [item.get("type") or item["role"] for item in body["input"]]


Turn = Callable[..., main.Conversation]


@pytest.fixture
def turn(api: Api) -> Turn:
    def run(
        prompt: str, conversation: main.Conversation | None = None
    ) -> main.Conversation:
        conversation = conversation or main.Conversation()
        main.run_turn(
            api.client, conversation, prompt, tools.build_tool_schemas(), "Be brief."
        )
        return conversation

    return run


def test_answer(
    api: Api, turn: Turn, streaming: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    api.replies = [response_body(message_item("Hello there."))]
    conversation = turn("hi")

    assert capsys.readouterr().out == "Hello there.\n[Context: 120 of 64,000 tokens]\n"
    assert [getattr(item, "type", "user") for item in conversation.history] == [
        "user",
        "message",
    ]
    assert conversation.history[1].content[0].text == "Hello there."


def test_request_body(
    api: Api, turn: Turn, streaming: bool, settings: config.Config
) -> None:
    settings.model = "some-model"
    api.replies = [response_body(message_item("ok"))]
    turn("hi")

    body = api.requests[0]
    assert body["model"] == "some-model"
    assert body["instructions"] == "Be brief."
    assert body["store"] is False
    assert body.get("stream", False) is streaming
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert [tool["name"] for tool in body["tools"]] == list(tools.TOOLS)[:-1]
    assert all(tool["strict"] is True for tool in body["tools"])


def test_tool_call_round_trip(
    api: Api, turn: Turn, streaming: bool, workspace: Path
) -> None:
    call = call_item("write_file", path="out.txt", content="héllo", append=False)
    api.replies = [response_body(call), response_body(message_item("Written."))]
    turn("write it")

    assert (workspace / "out.txt").read_text(encoding="utf-8") == "héllo"
    # The model's own function_call item is sent back, followed by its output.
    second = api.requests[1]
    assert input_kinds(second) == ["user", "function_call", "function_call_output"]
    sent_call, sent_output = second["input"][1], second["input"][2]
    assert sent_call["call_id"] == "call_1"
    assert sent_call["name"] == "write_file"
    assert json.loads(sent_call["arguments"]) == {
        "path": "out.txt",
        "content": "héllo",
        "append": False,
    }
    assert sent_output == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok": true, "message": "written 6 bytes to out.txt"}',
    }


def test_text_and_several_tool_calls_in_one_response(
    api: Api,
    turn: Turn,
    streaming: bool,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api.replies = [
        response_body(
            message_item("Working on it."),
            call_item("write_file", "a", path="a.txt", content="A", append=False),
            call_item("write_file", "b", path="b.txt", content="B", append=False),
        ),
        response_body(message_item("Done.")),
    ]
    turn("go")

    assert capsys.readouterr().out.startswith("Working on it.\nDone.\n")
    assert (workspace / "a.txt").read_text() == "A"
    assert (workspace / "b.txt").read_text() == "B"
    assert input_kinds(api.requests[1]) == [
        "user",
        "message",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert [item["call_id"] for item in api.requests[1]["input"][4:]] == ["a", "b"]


def test_earlier_answers_are_replayed_on_the_next_turn(
    api: Api, turn: Turn, streaming: bool
) -> None:
    api.replies = [
        response_body(message_item("First answer.")),
        response_body(message_item("Second answer.", "msg_2")),
    ]
    conversation = turn("one")
    turn("two", conversation)

    second = api.requests[1]
    assert input_kinds(second) == ["user", "message", "user"]
    assert second["input"][1]["role"] == "assistant"
    assert second["input"][1]["content"][0]["text"] == "First answer."


def test_reasoning_items_are_replayed_with_their_encrypted_content(
    api: Api, turn: Turn, streaming: bool
) -> None:
    """With store=False the API cannot look a reasoning item up by id, so the
    encrypted content it returned has to go back with the item."""
    call = call_item("write_file", path="out.txt", content="x", append=False)
    api.replies = [
        response_body(reasoning_item(), call),
        response_body(message_item("Done.")),
    ]
    turn("think, then write")

    second = api.requests[1]
    assert input_kinds(second) == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert second["input"][1] == reasoning_item()


def test_usage_calibrates_the_token_ratio(
    api: Api, turn: Turn, streaming: bool
) -> None:
    api.replies = [response_body(message_item("ok"))]
    conversation = turn("hi")
    assert conversation.tokens_per_byte != main.DEFAULT_TOKENS_PER_BYTE
    assert 0.1 <= conversation.tokens_per_byte <= 1.0


def test_incomplete_response(
    api: Api, turn: Turn, capsys: pytest.CaptureFixture[str]
) -> None:
    api.replies = [
        response_body(
            message_item("Partial"),
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
    ]
    conversation = turn("hi")
    assert "[Response incomplete: max_output_tokens]" in capsys.readouterr().out
    assert conversation.history == []


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, openai.AuthenticationError),
        (429, openai.RateLimitError),
        (500, openai.InternalServerError),
    ],
)
def test_api_errors_propagate_and_leave_history_clean(
    api: Api, turn: Turn, streaming: bool, status: int, error: type[Exception]
) -> None:
    api.replies = [httpx2.Response(status, json={"error": {"message": "nope"}})]
    conversation = main.Conversation()
    with pytest.raises(error, match="nope"):
        turn("hi", conversation)
    assert conversation.history == []


def test_main_reports_api_errors_and_continues(
    api: Api, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api.replies = [
        httpx2.Response(500, json={"error": {"message": "server down"}}),
        response_body(message_item("Recovered.")),
    ]
    inputs = iter(["first", "second", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(main, "OpenAI", lambda **kwargs: api.client)
    main.main()

    out = capsys.readouterr().out
    assert "Agent error: InternalServerError:" in out
    assert "server down" in out
    assert "Recovered." in out
    # The failed prompt is not replayed with the next one.
    assert api.requests[1]["input"] == [{"role": "user", "content": "second"}]
