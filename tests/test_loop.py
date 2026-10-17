"""The conversation loop, run through the real OpenAI client.

`api` (see conftest) answers the client's HTTP requests from a script, so these
tests also notice a change in the SDK itself: request shape, stream event
names, response parsing.
"""

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest

from conftest import (
    OMITTED_TOOL_OUTPUT,
    SUMMARY_MARKER,
    Api,
    call_item,
    kinds,
    message_item,
    reasoning_item,
    response_body,
)
from minime import config, main, shell
from minime.agent import SUMMARY_REQUEST, SYSTEM_PROMPT, Agent, build_toolbox
from minime.context import Context, json_bytes
from minime.main import Repl


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
    conversation: Context | None = None,
) -> Context:
    """One turn as the terminal runs it: text and notices are printed, problems
    are reported rather than raised, and a blank line follows."""
    agent = api.agent(conversation or Context(history=history or []))
    Repl(agent).ask(prompt)
    return agent.context


def turn(
    api: Api,
    history: list[Any] | None = None,
    prompt: str = "new",
    conversation: Context | None = None,
) -> Context:
    """One turn through Agent.run_turn itself, so problems are raised."""
    agent = api.agent(conversation or Context(history=history or []))
    repl = Repl(agent)
    agent.run_turn(prompt, on_text=repl.write, on_notice=repl.notice)
    return agent.context


def old_history() -> list[Any]:
    return [{"role": "user", "content": "old"}]


def overhead(settings: config.Config) -> int:
    """Bytes sent with every request besides the history: instructions and tools."""
    agent = Api(settings).agent()
    return len(agent.instructions().encode()) + json_bytes(agent.toolbox.schemas())


def tokens_for(settings: config.Config, history_bytes: int) -> int:
    """The token limit that leaves room for this many bytes of history at the default ratio."""
    return (overhead(settings) + history_bytes + 3) // 4


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
        run(api, prompt="hi")

        body = api.requests[0]
        assert body["model"] == "some-model"
        assert body["instructions"] == Api(settings).agent().instructions()
        assert body["store"] is False
        assert body.get("stream", False) is streaming
        assert body["input"] == [{"role": "user", "content": "hi"}]
        assert [tool["name"] for tool in body["tools"]] == list(
            build_toolbox(settings).tools
        )
        assert all(tool["strict"] is True for tool in body["tools"])
        assert "reasoning" not in body

    def test_instructions_are_sent_with_every_request(self, api: Api) -> None:
        api.replies = [response_body(write("a")), answer()]
        run(api)
        first, second = (body["instructions"] for body in api.requests)
        assert first.startswith(SYSTEM_PROMPT)
        assert second == first

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
        assert capsys.readouterr().out == "\nthe answer\n\n"

    def test_text_alongside_tool_calls_is_printed(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(message_item("checking"), write("a")), answer()]
        run(api)
        assert capsys.readouterr().out == "\nchecking\ndone\n\n"

    def test_empty_response_leaves_no_trace(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body()]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old"]
        assert capsys.readouterr().out == "\n\n"

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
        assert (
            capsys.readouterr().out == f"\npartial\n[Response incomplete: {shown}]\n\n"
        )

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

    def test_step_limit(
        self,
        api: Api,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_steps = 2
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
        assert "[Step limit reached (2)]" in capsys.readouterr().out

    def test_answer_on_the_last_allowed_step_is_not_a_limit(
        self,
        api: Api,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_steps = 2
        api.replies = [response_body(write("a")), answer()]
        run(api)
        assert "Step limit" not in capsys.readouterr().out


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
        def interrupted(self: shell.RunCommandTool, args: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        monkeypatch.setattr(shell.RunCommandTool, "run", interrupted)
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
        conversation = Context(history=old_history())
        with pytest.raises(error, match="nope"):
            turn(api, conversation=conversation)
        assert kinds(conversation.history) == ["user:old"]

    def test_api_error_after_a_step_keeps_the_step(self, api: Api) -> None:
        api.replies = [
            response_body(write("a")),
            httpx2.Response(500, json={"error": {"message": "boom"}}),
        ]
        conversation = Context(history=old_history())
        with pytest.raises(openai.InternalServerError, match="boom"):
            turn(api, conversation=conversation)
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
        inputs = iter(["first", "second", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(main, "Config", lambda: api.settings)
        monkeypatch.setattr("minime.agent.OpenAI", lambda **kwargs: api.client)
        main.main()

        out = capsys.readouterr().out
        assert "Agent error: InternalServerError:" in out
        assert "server down" in out
        assert "Recovered." in out
        # The failed prompt is not replayed with the next one.
        assert api.requests[1]["input"] == [{"role": "user", "content": "second"}]


class TestTokenBudget:
    """History is limited in tokens, using the model's own count per byte."""

    def test_cached_tokens_do_not_reduce_the_context_count(
        self, api: Api, settings: config.Config
    ) -> None:
        """They are a subset of input_tokens and still fill the context window."""
        api.replies = [answer(input_tokens=2000, cached_tokens=1900)]
        conversation = run(api, prompt="p" * 900)
        request_bytes = overhead(settings) + json_bytes(api.requests[0]["input"])
        assert conversation.tokens_per_byte == pytest.approx(2000 / request_bytes)

    def test_token_counts_are_not_printed(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("ok", input_tokens=5805, cached_tokens=4608)]
        run(api)
        assert capsys.readouterr().out == "\nok\n\n"

    def test_default_ratio_is_four_bytes_per_token(self) -> None:
        assert Context().tokens_per_byte == 0.25

    def test_ratio_is_learned_from_reported_usage(
        self, api: Api, streaming: bool, settings: config.Config
    ) -> None:
        api.replies = [answer(input_tokens=2000)]
        conversation = run(api, prompt="p" * 900)
        request_bytes = overhead(settings) + json_bytes(api.requests[0]["input"])
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
        settings.max_history_tokens = tokens_for(settings, 1000)
        history = [{"role": "user", "content": "o" * 2000}]
        api.replies = [answer(""), answer("ok")]  # no summary text, then the answer
        conversation = run(api, history)
        assert kinds(api.requests[1]["input"]) == ["user:new"]
        assert kinds(conversation.history) == ["user:new", "message"]

    def test_many_turns_are_kept_while_they_fit(self, api: Api) -> None:
        conversation = Context()
        for number in range(20):
            api.replies = [answer(f"answer {number}")]
            run(api, prompt=f"turn {number}", conversation=conversation)
        assert kinds(api.requests[-1]["input"])[0] == "user:turn 0"
        assert len(conversation.history) == 40

    def test_large_tool_outputs_are_omitted_within_a_turn(
        self, api: Api, settings: config.Config, workspace: Path
    ) -> None:
        (workspace / "big.txt").write_text("x" * 6000)
        settings.max_history_tokens = tokens_for(settings, 16_000)
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
        assert outputs[0] == OMITTED_TOOL_OUTPUT
        assert outputs[-1] != OMITTED_TOOL_OUTPUT
        # The request that followed the third read went out with the first omitted.
        sent = [item for item in api.requests[3]["input"] if "output" in item]
        assert sent[0]["output"] == OMITTED_TOOL_OUTPUT
        assert len(sent[2]["output"]) > 6000

    def test_oversized_prompt_stops_the_turn(
        self,
        api: Api,
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.max_history_tokens = tokens_for(settings, 500)
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
        settings.max_history_tokens = overhead(settings) + 3000

        def second_request_output(tokens_per_byte: float) -> str:
            api.requests.clear()
            api.replies = [response_body(read("r", "big.txt")), answer()]
            run(api, conversation=Context(tokens_per_byte=tokens_per_byte))
            return api.requests[1]["input"][2]["output"]

        assert second_request_output(0.25) != OMITTED_TOOL_OUTPUT
        assert second_request_output(1.0) == OMITTED_TOOL_OUTPUT

    def test_instructions_and_tools_count_against_the_limit(
        self, api: Api, settings: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        new_message = [{"role": "user", "content": "new"}]
        # Room for the new message and a few bytes more, but not for "old" too.
        settings.max_history_tokens = tokens_for(settings, json_bytes(new_message) + 5)
        # "old" is dropped (and an empty summary of it discarded).
        api.replies = [answer(""), answer()]
        run(api, old_history())
        assert kinds(api.requests[1]["input"]) == ["user:new"]

        # With 3000 bytes more room the same history fits untouched...
        settings.max_history_tokens += 3000 // 4
        api.replies = [answer()]
        run(api, old_history())
        assert kinds(api.requests[2]["input"]) == ["user:old", "user:new"]

        # ...until a 3000-byte longer system prompt takes that room back.
        monkeypatch.setattr("minime.agent.SYSTEM_PROMPT", SYSTEM_PROMPT + "i" * 3000)
        api.replies = [answer(""), answer()]
        run(api, old_history())
        assert kinds(api.requests[4]["input"]) == ["user:new"]

    def test_learned_ratio_carries_into_the_next_turn(
        self, api: Api, settings: config.Config
    ) -> None:
        # Room for 2000 bytes of history at 1 token per byte, far more at the default.
        settings.max_history_tokens = overhead(settings) + 2000
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
        settings.max_history_tokens = tokens_for(settings, 5000)

    def test_dropped_turns_are_replaced_by_a_summary(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer("Earlier: the user sent aaa."), answer("ok")]
        conversation = run(api, [big("a"), big("b")])

        summary = f"{SUMMARY_MARKER}\nEarlier: the user sent aaa."
        expected = [f"user:{summary}", f"user:{'b' * 3000}", "user:new"]
        assert kinds(api.requests[1]["input"]) == expected
        assert kinds(conversation.history) == [*expected, "message"]
        out = capsys.readouterr().out
        assert out.startswith(
            "\n[Summarising earlier conversation to save space…]\nok\n"
        )
        assert "trimmed to fit" not in out
        assert "Earlier: the user sent aaa." not in out  # the summary is not shown

    def test_summary_request(self, api: Api, streaming: bool) -> None:
        api.replies = [answer("summary"), answer("ok")]
        run(api, [big("a"), big("b")])

        request, following = api.requests
        # Exactly what was dropped, followed by the request to summarise it.
        assert request["input"] == [
            big("a"),
            {"role": "user", "content": SUMMARY_REQUEST},
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
        assert second_request[0]["content"] == f"{SUMMARY_MARKER}\nfirst summary"
        assert second_request[-1]["content"] == SUMMARY_REQUEST
        contents = [
            item["content"] for item in conversation.history if isinstance(item, dict)
        ]
        assert contents[0] == f"{SUMMARY_MARKER}\nsecond summary"
        assert sum(SUMMARY_MARKER in content for content in contents) == 1

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
        limit = settings.max_history_tokens
        assert limit >= 1000  # so the thousands separator shows
        assert f"[Conversation history trimmed to fit {limit:,} tokens]" in out

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
        assert api.requests[2]["input"][2]["output"] == OMITTED_TOOL_OUTPUT


class TestFailedResponses:
    def test_failed_response_is_reported_and_dropped(
        self, api: Api, streaming: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(status="failed", error="The model is overloaded.")]
        conversation = run(api, old_history())
        assert kinds(conversation.history) == ["user:old"]
        assert (
            capsys.readouterr().out
            == "\n[Response failed: The model is overloaded.]\n\n"
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
        assert capsys.readouterr().out == "\n[thinking…]\nthe answer\n\n"

    def test_each_reasoning_request_is_announced(
        self, api: Api, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.streaming = True
        api.replies = [
            response_body(reasoning_item("rs_1"), write("a")),
            response_body(reasoning_item("rs_2"), message_item("done")),
        ]
        run(api)
        assert capsys.readouterr().out == "\n[thinking…]\n[thinking…]\ndone\n\n"

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
        assert capsys.readouterr().out == "\nthe answer\n\n"


def test_system_prompt_has_no_stray_escape_characters() -> None:
    """A lone backslash in the prompt source (such as "\\frac" written as "\frac")
    turns into a control character instead of the text that was meant."""
    assert [c for c in SYSTEM_PROMPT if ord(c) < 32 and c != "\n"] == []


class TestInstructions:
    def test_environment_is_described(self, settings: config.Config) -> None:
        instructions = Api(settings).agent().instructions()
        assert instructions.startswith(SYSTEM_PROMPT)
        assert f"- Workspace root: {settings.workspace_root}" in instructions
        assert f"- Current date: {date.today().isoformat()}" in instructions
        assert "- Operating system: " in instructions
        assert "untrusted data" in instructions

    def test_braces_in_the_system_prompt_are_sent_as_written(
        self, settings: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "minime.agent.SYSTEM_PROMPT", 'Reply with JSON like {"ok": true}.\n'
        )
        assert (
            Api(settings)
            .agent()
            .instructions()
            .startswith('Reply with JSON like {"ok": true}.\n\nEnvironment:\n')
        )

    def test_date_is_refreshed_every_turn(
        self, api: Api, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        days = iter([date(2026, 10, 13), date(2026, 10, 14)])

        class Calendar:
            @staticmethod
            def today() -> date:
                return next(days)

        monkeypatch.setattr("minime.agent.date", Calendar)
        api.replies = [answer(), answer()]
        agent = api.agent()
        agent.run_turn("before midnight")
        agent.run_turn("after midnight")
        assert "- Current date: 2026-10-13\n" in api.requests[0]["instructions"]
        assert "- Current date: 2026-10-14\n" in api.requests[1]["instructions"]


class TestRepl:
    @pytest.fixture
    def repl(
        self, settings: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, Any]:
        """Run main() with scripted input, a fake OpenAI client and a recorded run_turn."""
        state: dict[str, Any] = {
            "inputs": [],
            "prompts": [],
            "agents": [],
            "client_kwargs": None,
        }

        def fake_input(prompt: str) -> str:
            item = state["inputs"].pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        def fake_openai(**kwargs: Any) -> str:
            state["client_kwargs"] = kwargs
            return "client"

        def fake_run_turn(agent: Agent, prompt: str, **callbacks: Any) -> None:
            state["prompts"].append(prompt)
            state["agents"].append(agent)
            agent.context.history.append({"role": "user", "content": prompt})
            if prompt == "explode":
                raise RuntimeError("boom")
            if prompt == "learn":
                agent.context.tokens_per_byte = 0.5

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(main, "Config", lambda: settings)
        monkeypatch.setattr("minime.agent.OpenAI", fake_openai)
        monkeypatch.setattr(Agent, "run_turn", fake_run_turn)
        return state

    def test_banner_and_client_settings(
        self,
        repl: dict[str, Any],
        settings: config.Config,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.model = "banner-model"
        settings.base_url = "http://localhost:9/v1"
        repl["inputs"] = ["/exit"]
        main.main()
        out = capsys.readouterr().out
        assert f"model: banner-model | workspace: {settings.workspace_root}" in out
        assert "Type /help to list commands." in out
        assert out.endswith("\nGoodbye!\n")
        assert repl["client_kwargs"] == {
            "api_key": "test-key",
            "base_url": "http://localhost:9/v1",
        }

    @pytest.mark.parametrize("word", ["/exit", "/quit", "/EXIT", "  /Quit  "])
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

    def test_prompts_are_passed_on_and_blank_lines_skipped(
        self, repl: dict[str, Any]
    ) -> None:
        repl["inputs"] = ["first", "", "second", "/exit"]
        main.main()
        assert repl["prompts"] == ["first", "second"]
        agent = repl["agents"][0]
        assert agent.client == "client"
        assert list(agent.toolbox.tools) == list(build_toolbox(agent.config).tools)

    @pytest.mark.parametrize("blank", ["   ", "\t", " \t "])
    def test_blank_prompts_are_not_sent(self, repl: dict[str, Any], blank: str) -> None:
        repl["inputs"] = [blank, "real", "/exit"]
        main.main()
        assert repl["prompts"] == ["real"]

    def test_one_agent_serves_every_turn(self, repl: dict[str, Any]) -> None:
        repl["inputs"] = ["a", "b", "/exit"]
        main.main()
        first, second = repl["agents"]
        assert first is second
        assert kinds(first.context.history) == ["user:a", "user:b"]

    def test_errors_are_reported_and_the_loop_continues(
        self, repl: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl["inputs"] = ["explode", "after", "/exit"]
        main.main()
        assert repl["prompts"] == ["explode", "after"]
        assert "[Agent error: RuntimeError: boom]" in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["/reset", "/RESET", "  /reset  "])
    def test_reset_empties_the_history_and_keeps_going(
        self,
        repl: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        repl["inputs"] = ["one", "two", command, "three", "/exit"]
        main.main()
        # /reset itself is not sent to the model.
        assert repl["prompts"] == ["one", "two", "three"]
        assert kinds(repl["agents"][0].context.history) == ["user:three"]
        assert "[History reset]" in capsys.readouterr().out

    def test_reset_keeps_the_learned_token_ratio(self, repl: dict[str, Any]) -> None:
        repl["inputs"] = ["learn", "/reset", "/exit"]
        main.main()
        assert repl["agents"][0].context.tokens_per_byte == 0.5

    def test_help_lists_every_command(
        self, repl: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl["inputs"] = ["/help", "/exit"]
        main.main()
        out = capsys.readouterr().out
        for command in ("/clear", "/exit", "/help", "/quit", "/reset"):
            assert f"  {command} " in out
        assert repl["prompts"] == []

    @pytest.mark.parametrize("tty", [True, False])
    def test_clear_wipes_the_screen_and_keeps_the_history(
        self,
        repl: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        tty: bool,
    ) -> None:
        monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
        repl["inputs"] = ["one", "/clear", "two", "/exit"]
        main.main()
        assert repl["prompts"] == ["one", "two"]
        assert kinds(repl["agents"][0].context.history) == ["user:one", "user:two"]
        # Escape codes only reach a terminal, never redirected output.
        assert ("\033[2J" in capsys.readouterr().out) is tty

    @pytest.mark.parametrize("word", ["/rest", "/tmp/notes.txt"])
    def test_unknown_command_is_reported_and_not_sent(
        self, repl: dict[str, Any], capsys: pytest.CaptureFixture[str], word: str
    ) -> None:
        repl["inputs"] = [word, "/exit"]
        main.main()
        assert repl["prompts"] == []
        assert f"[Unknown command: {word}. Type /help" in capsys.readouterr().out

    def test_slash_prompt_with_spaces_goes_to_the_model(
        self, repl: dict[str, Any]
    ) -> None:
        repl["inputs"] = ["/usr/bin is missing", "/exit"]
        main.main()
        assert repl["prompts"] == ["/usr/bin is missing"]


class TestNotices:
    """How Repl lays out the model's text and its own bracketed notices."""

    def test_notice_after_text_starts_a_new_line(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(Api(settings).agent())
        repl.write("partial answer")
        repl.notice("Interrupted; returning to prompt")
        assert (
            capsys.readouterr().out
            == "partial answer\n[Interrupted; returning to prompt]\n"
        )

    def test_notice_at_a_line_start_adds_no_blank_line(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(Api(settings).agent())
        repl.write("full line\n")
        repl.notice("thinking…")
        repl.notice("thinking…")
        assert capsys.readouterr().out == "full line\n[thinking…]\n[thinking…]\n"
