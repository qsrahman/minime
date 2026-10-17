"""The conversation loop, run through the real OpenAI client.

`api` (see conftest) answers the client's HTTP requests from a script, so these
tests also notice a change in the SDK itself: request shape, stream event
names, response parsing.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from conftest import (
    Api,
    call_item,
    kinds,
    message_item,
    reasoning_item,
    response_body,
)

import config
import main
import tools


def write(call_id: str, path: str = "out.txt") -> dict[str, Any]:
    return call_item("write_file", call_id, path=path, content=call_id, append=False)


def read(call_id: str, path: str) -> dict[str, Any]:
    return call_item(
        "read_file", call_id, path=path, start_line=None, end_line=None, offset=None
    )


def answer(text: str = "done", **kwargs: Any) -> dict[str, Any]:
    return response_body(message_item(text), **kwargs)


def run(
    api: Api,
    history: list[Any] | None = None,
    prompt: str = "new",
    instructions: str = "",
    conversation: main.Conversation | None = None,
) -> main.Conversation:
    conversation = conversation or main.Conversation(history=history or [])
    main.run_turn(
        api.client, conversation, prompt, tools.build_tool_schemas(), instructions
    )
    return conversation


def old_history() -> list[Any]:
    return [{"role": "user", "content": "old"}]


def overhead() -> int:
    """Bytes of tool schemas sent with every request (instructions are extra)."""
    return main.json_bytes(tools.build_tool_schemas())


def tokens_for(history_bytes: int, instructions: str = "") -> int:
    """The token limit that leaves room for this many bytes of history at the default ratio."""
    return (overhead() + len(instructions) + history_bytes + 3) // 4


@pytest.fixture(params=[False, True], ids=["create", "stream"])
def streaming(request: pytest.FixtureRequest, settings: config.Config) -> bool:
    settings.streaming = request.param
    return request.param


class TestRequests:
    def test_request_body(
        self, api: Api, streaming: bool, settings: config.Config
    ) -> None:
        settings.model = "some-model"
        api.replies = [answer()]
        run(api, prompt="hi", instructions="Be brief.")

        body = api.requests[0]
        assert body["model"] == "some-model"
        assert body["instructions"] == "Be brief."
        assert body["store"] is False
        assert body.get("stream", False) is streaming
        assert body["input"] == [{"role": "user", "content": "hi"}]
        assert [tool["name"] for tool in body["tools"]] == list(tools.TOOLS)[:-1]
        assert all(tool["strict"] is True for tool in body["tools"])
        assert "reasoning" not in body

    def test_instructions_are_sent_with_every_request(self, api: Api) -> None:
        api.replies = [response_body(write("a")), answer()]
        run(api, instructions="Be brief.")
        assert [body["instructions"] for body in api.requests] == ["Be brief."] * 2

    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_reasoning_effort_is_sent_when_configured(
        self, api: Api, streaming: bool, settings: config.Config, effort: str
    ) -> None:
        settings.reasoning_effort = effort  # type: ignore[assignment]
        api.replies = [response_body(write("a")), answer()]
        run(api)
        assert [body["reasoning"] for body in api.requests] == [{"effort": effort}] * 2


class TestAnswers:
    def test_final_answer(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("the answer")]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old", "user:new", "message"]
        assert conversation.history[2].content[0].text == "the answer"
        assert capsys.readouterr().out == "the answer\n"

    def test_text_alongside_tool_calls_is_printed(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(message_item("checking"), write("a")), answer()]
        run(api)
        assert capsys.readouterr().out == "checking\ndone\n"

    def test_empty_response_leaves_no_trace(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body()]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old"]
        assert capsys.readouterr().out == ""

    def test_earlier_answers_are_replayed_on_the_next_turn(
        self, api: Api, streaming: bool
    ) -> None:
        api.replies = [answer("First answer."), answer("Second answer.")]
        conversation = run(api, prompt="one")
        run(api, prompt="two", conversation=conversation)

        second = api.requests[1]
        assert kinds(second["input"]) == ["user:one", "message", "user:two"]
        assert second["input"][1]["role"] == "assistant"
        assert second["input"][1]["content"][0]["text"] == "First answer."

    @pytest.mark.parametrize(
        ("reason", "shown"),
        [("max_output_tokens", "max_output_tokens"), (None, "unknown")],
    )
    def test_incomplete_response_is_reported_and_dropped(
        self,
        api: Api,
        streaming: bool,
        capsys: pytest.CaptureFixture[str],
        reason: str | None,
        shown: str,
    ) -> None:
        api.replies = [answer("partial", status="incomplete", reason=reason)]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old"]
        assert capsys.readouterr().out == f"partial\n[Response incomplete: {shown}]\n"

    def test_incomplete_response_after_a_tool_step_keeps_the_step(
        self, api: Api
    ) -> None:
        api.replies = [
            response_body(write("a")),
            answer("partial", status="incomplete"),
        ]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
        ]


class TestToolCalls:
    def test_tool_step_then_answer(
        self, api: Api, streaming: bool, workspace: Path
    ) -> None:
        api.replies = [response_body(write("a")), answer()]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
            "message",
        ]
        assert (workspace / "out.txt").read_text() == "a"
        # The second request carries the model's call and its output.
        assert kinds(api.requests[1]["input"]) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
        ]

    def test_call_and_output_are_sent_back_as_the_api_expects(
        self, api: Api, streaming: bool, workspace: Path
    ) -> None:
        call = call_item("write_file", path="out.txt", content="héllo", append=False)
        api.replies = [response_body(call), answer()]
        run(api)

        assert (workspace / "out.txt").read_text(encoding="utf-8") == "héllo"
        sent_call, sent_output = api.requests[1]["input"][1:]
        assert sent_call["type"] == "function_call"
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

    def test_several_calls_are_run_and_recorded_in_order(
        self, api: Api, streaming: bool, workspace: Path
    ) -> None:
        api.replies = [
            response_body(
                message_item("working"), write("a", "a.txt"), write("b", "b.txt")
            ),
            answer(),
        ]
        conversation = run(api)
        assert kinds(conversation.history) == [
            "user:new",
            "message",
            "call:a",
            "call:b",
            "output:a:ok",
            "output:b:ok",
            "message",
        ]
        assert kinds(api.requests[1]["input"]) == kinds(conversation.history)[:-1]
        assert (workspace / "a.txt").read_text() == "a"
        assert (workspace / "b.txt").read_text() == "b"

    def test_tool_failures_are_recorded_as_results(self, api: Api) -> None:
        api.replies = [response_body(read("r", "missing.txt")), answer()]
        conversation = run(api)
        result = json.loads(conversation.history[2]["output"])
        assert result["ok"] is False
        assert result["message"].startswith("FileNotFoundError")

    def test_non_ascii_output_is_not_escaped(self, api: Api, workspace: Path) -> None:
        (workspace / "utf.txt").write_text("café 漢字\n", encoding="utf-8")
        api.replies = [response_body(read("r", "utf.txt")), answer()]
        conversation = run(api)
        assert "café 漢字" in conversation.history[2]["output"]
        assert "\\u" not in conversation.history[2]["output"]

    def test_reasoning_items_are_replayed_with_their_encrypted_content(
        self, api: Api, streaming: bool
    ) -> None:
        """With store=False the API cannot look a reasoning item up by id, so the
        encrypted content it returned has to go back with the item."""
        api.replies = [response_body(reasoning_item(), write("a")), answer()]
        run(api)

        second = api.requests[1]
        assert kinds(second["input"]) == [
            "user:new",
            "reasoning",
            "call:a",
            "output:a:ok",
        ]
        assert second["input"][1] == reasoning_item()

    def test_turn_limit(
        self,
        api: Api,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_turns = 2
        api.replies = [
            response_body(write("a")),
            response_body(write("b")),
            answer("never reached"),
        ]
        conversation = run(api)
        assert len(api.requests) == 2
        # Tool calls from the last allowed request still run.
        assert kinds(conversation.history) == [
            "user:new",
            "call:a",
            "output:a:ok",
            "call:b",
            "output:b:ok",
        ]
        assert "Turn limit reached (2)." in capsys.readouterr().out

    def test_answer_on_the_last_allowed_turn_is_not_a_limit(
        self,
        api: Api,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_turns = 2
        api.replies = [response_body(write("a")), answer()]
        run(api)
        assert "Turn limit" not in capsys.readouterr().out


class TestInterruptsAndErrors:
    def test_interrupt_before_any_step_restores_history(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [KeyboardInterrupt()]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old"]
        assert "[Interrupted; returning to prompt]" in capsys.readouterr().out

    def test_interrupt_after_a_step_keeps_the_step(
        self, api: Api, streaming: bool
    ) -> None:
        api.replies = [response_body(write("a")), KeyboardInterrupt()]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
        ]

    def test_interrupt_during_a_tool_keeps_a_complete_step(
        self,
        api: Api,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def interrupted(**kwargs: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        monkeypatch.setitem(
            tools.TOOLS, "run_command", (interrupted, tools.RunCommandArguments)
        )
        sleep = call_item("run_command", "b", cmd="sleep 60", timeout=None, cwd=None)
        api.replies = [response_body(write("a"), sleep, write("c")), answer("never")]
        conversation = run(api, old_history())

        assert kinds(conversation.history) == [
            "user:old",
            "user:new",
            "call:a",
            "call:b",
            "call:c",
            "output:a:ok",
            "output:b:Interrupted by the user.",
            "output:c:Interrupted by the user.",
        ]
        assert len(api.requests) == 1
        assert "[Interrupted; returning to prompt]" in capsys.readouterr().out
        # The call before the interrupt ran; the one after it did not.
        assert (workspace / "out.txt").read_text() == "a"

    def test_next_turn_follows_an_interrupted_one(self, api: Api) -> None:
        api.replies = [response_body(write("a")), KeyboardInterrupt(), answer()]
        conversation = run(api, prompt="first")
        run(api, prompt="continue", conversation=conversation)
        assert kinds(api.requests[2]["input"]) == [
            "user:first",
            "call:a",
            "output:a:ok",
            "user:continue",
        ]

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (401, openai.AuthenticationError),
            (429, openai.RateLimitError),
            (500, openai.InternalServerError),
        ],
    )
    def test_api_error_before_any_step_restores_history(
        self, api: Api, streaming: bool, status: int, error: type[Exception]
    ) -> None:
        api.replies = [httpx2.Response(status, json={"error": {"message": "nope"}})]
        conversation = main.Conversation(history=old_history())
        with pytest.raises(error, match="nope"):
            run(api, conversation=conversation)
        assert kinds(conversation.history) == ["user:old"]

    def test_api_error_after_a_step_keeps_the_step(self, api: Api) -> None:
        api.replies = [
            response_body(write("a")),
            httpx2.Response(500, json={"error": {"message": "boom"}}),
        ]
        conversation = main.Conversation(history=old_history())
        with pytest.raises(openai.InternalServerError, match="boom"):
            run(api, conversation=conversation)
        assert kinds(conversation.history) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
        ]

    def test_main_reports_api_errors_and_continues(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        api.replies = [
            httpx2.Response(500, json={"error": {"message": "server down"}}),
            answer("Recovered."),
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


class TestTokenBudget:
    """History is limited in tokens, using the model's own count per byte."""

    def test_usage_is_reported_after_the_turn(
        self,
        api: Api,
        streaming: bool,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_history_tokens = 64_000
        api.replies = [answer("ok", input_tokens=12_345)]
        run(api)
        assert (
            capsys.readouterr().out
            == "ok\n[Context: 12,345 of 64,000 tokens; output: 7 tokens]\n"
        )

    def test_cached_tokens_are_reported(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("ok", input_tokens=5805, cached_tokens=4608)]
        run(api)
        assert (
            capsys.readouterr().out
            == "ok\n[Context: 5,805 of 200,000 tokens, 4,608 cached; output: 7 tokens]\n"
        )

    def test_cached_tokens_do_not_reduce_the_context_count(self, api: Api) -> None:
        """They are a subset of input_tokens and still fill the context window."""
        api.replies = [answer(input_tokens=2000, cached_tokens=1900)]
        conversation = run(api, prompt="p" * 900)
        request_bytes = overhead() + main.json_bytes(api.requests[0]["input"])
        assert conversation.tokens_per_byte == pytest.approx(2000 / request_bytes)

    def test_the_last_request_of_the_turn_is_the_one_reported(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [
            response_body(write("a"), input_tokens=100),
            answer(input_tokens=250),
        ]
        run(api)
        out = capsys.readouterr().out
        assert "[Context: 250 of" in out
        assert "[Context: 100 of" not in out

    def test_nothing_is_reported_without_usage(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("ok")]
        run(api)
        assert "Context" not in capsys.readouterr().out

    def test_usage_is_reported_even_when_interrupted(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(write("a"), input_tokens=100), KeyboardInterrupt()]
        run(api)
        assert "[Context: 100 of" in capsys.readouterr().out

    def test_default_ratio_is_four_bytes_per_token(self) -> None:
        assert main.Conversation().tokens_per_byte == 0.25

    def test_ratio_is_learned_from_reported_usage(
        self, api: Api, streaming: bool
    ) -> None:
        instructions = "i" * 60
        api.replies = [answer(input_tokens=2000)]
        conversation = run(api, prompt="p" * 900, instructions=instructions)
        request_bytes = (
            len(instructions) + overhead() + main.json_bytes(api.requests[0]["input"])
        )
        assert conversation.tokens_per_byte == pytest.approx(2000 / request_bytes)
        assert 0.1 < conversation.tokens_per_byte < 1.0

    def test_ratio_is_kept_without_usage(self, api: Api) -> None:
        api.replies = [answer()]
        assert run(api).tokens_per_byte == 0.25

    @pytest.mark.parametrize(
        ("input_tokens", "expected"), [(1, 0.1), (10_000_000, 1.0)]
    )
    def test_implausible_usage_is_bounded(
        self, api: Api, input_tokens: int, expected: float
    ) -> None:
        api.replies = [answer(input_tokens=input_tokens)]
        assert run(api).tokens_per_byte == expected

    def test_old_turns_are_dropped_to_fit(
        self, api: Api, settings: config.Config
    ) -> None:
        settings.max_history_tokens = tokens_for(1000)
        history = [{"role": "user", "content": "o" * 2000}]
        api.replies = [answer(""), answer("ok")]  # no summary text, then the answer
        conversation = run(api, history)
        assert kinds(api.requests[1]["input"]) == ["user:new"]
        assert kinds(conversation.history) == ["user:new", "message"]

    def test_output_tokens_are_added_up_over_the_turn(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [
            response_body(write("a"), input_tokens=100, output_tokens=40),
            answer(input_tokens=250, output_tokens=15),
        ]
        run(api)
        assert (
            "[Context: 250 of 200,000 tokens; output: 55 tokens]"
            in capsys.readouterr().out
        )

    def test_many_turns_are_kept_while_they_fit(self, api: Api) -> None:
        conversation = main.Conversation()
        for number in range(20):
            api.replies = [answer(f"answer {number}")]
            run(api, prompt=f"turn {number}", conversation=conversation)
        assert kinds(api.requests[-1]["input"])[0] == "user:turn 0"
        assert len(conversation.history) == 40

    def test_large_tool_outputs_are_omitted_within_a_turn(
        self, api: Api, settings: config.Config, workspace: Path
    ) -> None:
        (workspace / "big.txt").write_text("x" * 6000)
        settings.max_history_tokens = tokens_for(16_000)
        api.replies = [
            *(response_body(read(f"r{i}", "big.txt")) for i in range(3)),
            answer(),
        ]
        conversation = run(api)

        outputs = [
            item["output"]
            for item in conversation.history
            if isinstance(item, dict) and "output" in item
        ]
        assert outputs[0] == main.OMITTED_TOOL_OUTPUT
        assert outputs[-1] != main.OMITTED_TOOL_OUTPUT
        # The request that followed the third read went out with the first omitted.
        sent = [item for item in api.requests[3]["input"] if "output" in item]
        assert sent[0]["output"] == main.OMITTED_TOOL_OUTPUT
        assert len(sent[2]["output"]) > 6000

    def test_oversized_prompt_stops_the_turn(
        self,
        api: Api,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_history_tokens = tokens_for(500)
        api.replies = [answer("never reached")]
        conversation = run(api, prompt="p" * 5000)
        assert api.requests == []
        assert (
            "[Turn stopped because its history exceeded the context budget]"
            in capsys.readouterr().out
        )
        assert [item["content"] for item in conversation.history] == [
            "[Previous request omitted because its history exceeded the context budget.]"
        ]

    def test_token_dense_text_is_trimmed_sooner(
        self, api: Api, settings: config.Config, workspace: Path
    ) -> None:
        """The same history fits at 4 bytes per token but not at 1 byte per token."""
        (workspace / "big.txt").write_text("x" * 6000)
        settings.max_history_tokens = overhead() + 3000

        def second_request_output(tokens_per_byte: float) -> str:
            api.requests.clear()
            api.replies = [response_body(read("r", "big.txt")), answer()]
            run(api, conversation=main.Conversation(tokens_per_byte=tokens_per_byte))
            return api.requests[1]["input"][2]["output"]

        assert second_request_output(0.25) != main.OMITTED_TOOL_OUTPUT
        assert second_request_output(1.0) == main.OMITTED_TOOL_OUTPUT

    def test_instructions_and_tools_count_against_the_limit(
        self, api: Api, settings: config.Config
    ) -> None:
        instructions = "i" * 3000
        new_message = [{"role": "user", "content": "new"}]
        # Room for the new message and a few bytes more, but not for "old" too.
        settings.max_history_tokens = tokens_for(
            main.json_bytes(new_message) + 5, instructions
        )
        api.replies = [answer(""), answer(), answer()]
        run(api, old_history(), instructions=instructions)
        assert kinds(api.requests[1]["input"]) == ["user:new"]

        # Without the instructions the same history fits untouched.
        run(api, old_history())
        assert kinds(api.requests[2]["input"]) == ["user:old", "user:new"]

    def test_learned_ratio_carries_into_the_next_turn(
        self, api: Api, settings: config.Config
    ) -> None:
        # Room for 2000 bytes of history at 1 token per byte, far more at the default.
        settings.max_history_tokens = overhead() + 2000
        api.replies = [answer("a" * 10, input_tokens=10_000_000), answer(""), answer()]
        conversation = run(api, prompt="p" * 1900)
        assert conversation.tokens_per_byte == 1.0
        run(api, prompt="next", conversation=conversation)
        assert kinds(api.requests[2]["input"]) == ["user:next"]


def big(letter: str) -> dict[str, Any]:
    return {"role": "user", "content": letter * 3000}


class TestSummaries:
    """Turns dropped to fit the limit are replaced by a model-written summary."""

    @pytest.fixture(autouse=True)
    def limit(self, settings: config.Config) -> None:
        # Room for about 5000 bytes: two big() messages plus a prompt do not fit,
        # so the first is dropped, leaving space for a short summary.
        settings.max_history_tokens = tokens_for(5000)

    def test_dropped_turns_are_replaced_by_a_summary(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("Earlier: the user sent aaa."), answer("ok")]
        conversation = run(api, [big("a"), big("b")])

        summary = f"{main.SUMMARY_MARKER}\nEarlier: the user sent aaa."
        expected = [f"user:{summary}", f"user:{'b' * 3000}", "user:new"]
        assert kinds(api.requests[1]["input"]) == expected
        assert kinds(conversation.history) == [*expected, "message"]
        out = capsys.readouterr().out
        assert out.startswith("[Summarising earlier conversation to save space…]\nok\n")
        assert "trimmed to fit" not in out
        assert "Earlier: the user sent aaa." not in out  # the summary is not shown

    def test_summary_request(self, api: Api, streaming: bool) -> None:
        api.replies = [answer("summary"), answer("ok")]
        run(api, [big("a"), big("b")], instructions="Be brief.")

        request, following = api.requests
        # Exactly what was dropped, followed by the request to summarise it.
        assert request["input"] == [
            big("a"),
            {"role": "user", "content": main.SUMMARY_REQUEST},
        ]
        assert request["tool_choice"] == "none"
        assert "stream" not in request
        assert request["store"] is False
        # The same prefix as ordinary requests, so the provider's cache applies.
        for key in ("model", "instructions", "tools"):
            assert request[key] == following[key]

    def test_summary_request_uses_the_configured_reasoning_effort(
        self, api: Api, settings: config.Config
    ) -> None:
        settings.reasoning_effort = "low"
        api.replies = [answer("summary"), answer("ok")]
        run(api, [big("a"), big("b")])
        assert api.requests[0]["reasoning"] == {"effort": "low"}

    def test_tool_calls_in_dropped_turns_are_sent_to_the_summariser(
        self, api: Api
    ) -> None:
        api.replies = [
            response_body(write("a")),
            answer("done"),
            answer("summary"),
            answer(),
        ]
        conversation = run(api, prompt="first")
        conversation.history += [big("b"), big("c")]
        run(api, prompt="new", conversation=conversation)
        assert kinds(api.requests[2]["input"])[:4] == [
            "user:first",
            "call:a",
            "output:a:ok",
            "message",
        ]

    def test_an_earlier_summary_is_folded_into_the_next_one(self, api: Api) -> None:
        api.replies = [answer("first summary"), answer("ok")]
        conversation = run(api, [big("a"), big("b")])

        api.replies = [answer("second summary"), answer("ok")]
        run(api, prompt="c" * 3000, conversation=conversation)

        second_request = api.requests[2]["input"]
        assert second_request[0]["content"] == f"{main.SUMMARY_MARKER}\nfirst summary"
        assert second_request[-1]["content"] == main.SUMMARY_REQUEST
        contents = [
            item["content"] for item in conversation.history if isinstance(item, dict)
        ]
        assert contents[0] == f"{main.SUMMARY_MARKER}\nsecond summary"
        assert sum(main.SUMMARY_MARKER in content for content in contents) == 1

    def test_summary_output_tokens_are_counted(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [
            answer("summary", input_tokens=900, output_tokens=120),
            answer("ok", input_tokens=700, output_tokens=5),
        ]
        run(api, [big("a"), big("b")])
        out = capsys.readouterr().out
        assert "[Context: 700 of" in out  # the real request, not the summary one
        assert "output: 125 tokens]" in out

    @pytest.mark.parametrize("text", ["", "   \n"])
    def test_empty_summary_is_not_inserted(self, api: Api, text: str) -> None:
        api.replies = [answer(text), answer("ok")]
        run(api, [big("a"), big("b")])
        assert kinds(api.requests[1]["input"]) == [f"user:{'b' * 3000}", "user:new"]

    def test_summary_that_does_not_fit_is_not_inserted(self, api: Api) -> None:
        api.replies = [answer("s" * 4000), answer("ok")]
        run(api, [big("a"), big("b")])
        assert kinds(api.requests[1]["input"]) == [f"user:{'b' * 3000}", "user:new"]

    def test_failed_summary_falls_back_to_dropping(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [
            httpx2.Response(500, json={"error": {"message": "down"}}),
            answer("ok"),
        ]
        conversation = run(api, [big("a"), big("b")])
        assert kinds(api.requests[1]["input"]) == [f"user:{'b' * 3000}", "user:new"]
        assert kinds(conversation.history)[-1] == "message"
        assert (
            "[No summary (InternalServerError); it was dropped instead]"
            in capsys.readouterr().out
        )

    def test_interrupt_while_summarising_restores_the_history(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [KeyboardInterrupt()]
        conversation = run(api, [big("a"), big("b")])
        assert conversation.history == [big("a"), big("b")]
        assert "[Interrupted; returning to prompt]" in capsys.readouterr().out

    def test_summaries_can_be_turned_off(
        self, api: Api, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.summarize = False
        api.replies = [answer("ok")]
        run(api, [big("a"), big("b")])
        assert len(api.requests) == 1
        assert kinds(api.requests[0]["input"]) == [f"user:{'b' * 3000}", "user:new"]
        out = capsys.readouterr().out
        assert "Summarising" not in out
        assert "[Conversation history trimmed to fit" in out

    def test_omitting_tool_outputs_needs_no_summary(
        self, api: Api, workspace: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (workspace / "big.txt").write_text("x" * 2500)
        api.replies = [
            response_body(read("r1", "big.txt")),
            response_body(read("r2", "big.txt")),
            answer(),
        ]
        run(api)
        assert len(api.requests) == 3  # no extra request
        out = capsys.readouterr().out
        assert "Summarising" not in out
        assert "[Conversation history trimmed to fit" in out
        assert api.requests[2]["input"][2]["output"] == main.OMITTED_TOOL_OUTPUT


class TestFailedResponses:
    def test_failed_response_is_reported_and_dropped(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(status="failed", error="The model is overloaded.")]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old"]
        assert (
            capsys.readouterr().out == "[Response failed: The model is overloaded.]\n"
        )

    def test_failed_response_without_details(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(status="failed")]
        run(api)
        assert "[Response failed: unknown error]" in capsys.readouterr().out

    def test_failed_response_after_a_tool_step_keeps_the_step(self, api: Api) -> None:
        api.replies = [
            response_body(write("a")),
            response_body(status="failed", error="boom"),
        ]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
        ]


class TestThinking:
    def test_reasoning_is_announced_while_streaming(
        self, api: Api, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.streaming = True
        api.replies = [response_body(reasoning_item(), message_item("the answer"))]
        run(api)
        assert capsys.readouterr().out == "[thinking…]\nthe answer\n"

    def test_each_reasoning_request_is_announced(
        self, api: Api, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.streaming = True
        api.replies = [
            response_body(reasoning_item("rs_1"), write("a")),
            response_body(reasoning_item("rs_2"), message_item("done")),
        ]
        run(api)
        assert capsys.readouterr().out == "[thinking…]\n[thinking…]\ndone\n"

    def test_nothing_is_announced_without_reasoning(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("the answer")]
        run(api)
        assert "thinking" not in capsys.readouterr().out

    def test_nothing_can_be_announced_without_streaming(
        self, api: Api, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.streaming = False
        api.replies = [response_body(reasoning_item(), message_item("the answer"))]
        run(api)
        assert capsys.readouterr().out == "the answer\n"


def test_system_prompt_has_no_stray_escape_characters() -> None:
    """A lone backslash in the prompt source (such as "\\frac" written as "\frac")
    turns into a control character instead of the text that was meant."""
    assert [c for c in main.system_prompt if ord(c) < 32 and c != "\n"] == []


class TestMain:
    @pytest.fixture
    def repl(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Run main() with scripted input, a fake OpenAI client and a recorded run_turn."""
        state: dict[str, Any] = {
            "inputs": [],
            "prompts": [],
            "client_kwargs": None,
            "turn": None,
            "instructions": None,
        }

        def fake_input(prompt: str) -> str:
            item = state["inputs"].pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        def fake_openai(**kwargs: Any) -> str:
            state["client_kwargs"] = kwargs
            return "client"

        def fake_run_turn(
            client: Any,
            conversation: main.Conversation,
            prompt: str,
            tool_schemas: Any,
            instructions: str,
        ) -> None:
            state["prompts"].append(prompt)
            state["turn"] = (client, conversation, tool_schemas)
            state["instructions"] = instructions
            if prompt == "explode":
                raise RuntimeError("boom")

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(main, "OpenAI", fake_openai)
        monkeypatch.setattr(main, "run_turn", fake_run_turn)
        return state

    def test_banner_and_client_settings(
        self,
        repl: dict[str, Any],
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.model = "banner-model"
        settings.base_url = "http://localhost:9/v1"
        repl["inputs"] = ["exit"]
        main.main()
        out = capsys.readouterr().out
        assert f"model: banner-model | workspace: {settings.workspace_root}" in out
        assert out.endswith("\nGoodbye!\n")
        assert repl["client_kwargs"] == {
            "api_key": "test-key",
            "base_url": "http://localhost:9/v1",
        }

    def test_instructions_describe_the_environment(
        self, repl: dict[str, Any], settings: config.Config
    ) -> None:
        repl["inputs"] = ["hello", "exit"]
        main.main()
        instructions = repl["instructions"]
        assert instructions.startswith(main.system_prompt)
        assert f"- Workspace root: {settings.workspace_root}" in instructions
        assert f"- Current date: {date.today().isoformat()}" in instructions
        assert "- Operating system: " in instructions
        assert "untrusted data" in instructions

    def test_braces_in_the_system_prompt_are_sent_as_written(
        self, repl: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            main, "system_prompt", 'Reply with JSON like {"ok": true}.\n'
        )
        repl["inputs"] = ["hello", "exit"]
        main.main()
        assert repl["instructions"].startswith(
            'Reply with JSON like {"ok": true}.\n\nEnvironment:\n'
        )

    def test_date_is_refreshed_every_turn(
        self, repl: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        days = iter([date(2026, 10, 13), date(2026, 10, 14)])

        class Calendar:
            @staticmethod
            def today() -> date:
                return next(days)

        seen: list[str] = []
        monkeypatch.setattr(main, "date", Calendar)
        monkeypatch.setattr(
            main,
            "run_turn",
            lambda client, conversation, prompt, schemas, instructions: seen.append(
                instructions
            ),
        )
        repl["inputs"] = ["before midnight", "after midnight", "exit"]
        main.main()
        assert "- Current date: 2026-10-13\n" in seen[0]
        assert "- Current date: 2026-10-14\n" in seen[1]

    @pytest.mark.parametrize("word", ["exit", "quit", "EXIT", "  Quit  "])
    def test_exit_words(self, repl: dict[str, Any], word: str) -> None:
        repl["inputs"] = [word, "never read"]
        main.main()
        assert repl["prompts"] == []
        assert repl["inputs"] == ["never read"]

    @pytest.mark.parametrize("signal", [EOFError(), KeyboardInterrupt()])
    def test_end_of_input_or_ctrl_c_at_the_prompt_quits(
        self,
        repl: dict[str, Any],
        signal: BaseException,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repl["inputs"] = [signal]
        main.main()
        assert capsys.readouterr().out.count("Goodbye!") == 1

    def test_prompts_are_passed_to_run_turn(self, repl: dict[str, Any]) -> None:
        repl["inputs"] = ["first", "", "second", "exit"]
        main.main()
        assert repl["prompts"] == ["first", "second"]
        client, conversation, tool_schemas = repl["turn"]
        assert client == "client"
        assert conversation == main.Conversation()
        assert [schema["name"] for schema in tool_schemas] == list(tools.TOOLS)[:-1]

    def test_errors_are_reported_and_the_loop_continues(
        self, repl: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl["inputs"] = ["explode", "after", "exit"]
        main.main()
        assert repl["prompts"] == ["explode", "after"]
        assert "Agent error: RuntimeError: boom" in capsys.readouterr().out

    def test_one_conversation_is_shared_across_turns(
        self, repl: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        monkeypatch.setattr(
            main,
            "run_turn",
            lambda client, conversation, prompt, schemas, instructions: seen.append(
                id(conversation)
            ),
        )
        repl["inputs"] = ["a", "b", "exit"]
        main.main()
        assert len(seen) == 2 and seen[0] == seen[1]

    def test_banner_mentions_clear(
        self, repl: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl["inputs"] = ["exit"]
        main.main()
        assert "'/clear' to start a new conversation" in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["/clear", "/CLEAR", "  /clear  "])
    def test_clear_empties_the_history_and_keeps_going(
        self,
        repl: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        sizes: list[int] = []

        def fake_run_turn(
            client: Any,
            conversation: main.Conversation,
            prompt: str,
            schemas: Any,
            instructions: str,
        ) -> None:
            sizes.append(len(conversation.history))
            conversation.history.append({"role": "user", "content": prompt})
            conversation.tokens_per_byte = 0.5

        monkeypatch.setattr(main, "run_turn", fake_run_turn)
        repl["inputs"] = ["one", "two", command, "three", "exit"]
        main.main()
        # History grows, is emptied by /clear, and /clear itself is not sent.
        assert sizes == [0, 1, 0]
        assert "[Conversation cleared]" in capsys.readouterr().out

    def test_clear_keeps_the_learned_token_ratio(
        self, repl: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ratios: list[float] = []

        def fake_run_turn(
            client: Any,
            conversation: main.Conversation,
            prompt: str,
            schemas: Any,
            instructions: str,
        ) -> None:
            ratios.append(conversation.tokens_per_byte)
            conversation.tokens_per_byte = 0.5

        monkeypatch.setattr(main, "run_turn", fake_run_turn)
        repl["inputs"] = ["one", "/clear", "two", "exit"]
        main.main()
        assert ratios == [0.25, 0.5]
