import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeClient, kinds, response, tool_call

import config
import main
import tools

SCHEMAS: list[Any] = [{"type": "function", "name": "stub"}]


def write(call_id: str, path: str = "out.txt") -> Any:
    return tool_call("write_file", call_id, path=path, content=call_id, append=False)


def run(
    client: FakeClient,
    history: list[Any],
    prompt: str = "new",
    instructions: str = "",
) -> main.Conversation:
    conversation = main.Conversation(history=history)
    main.run_turn(client, conversation, prompt, SCHEMAS, instructions)  # type: ignore[arg-type]
    return conversation


def old_history() -> list[Any]:
    return [{"role": "user", "content": "old"}]


class TestRunTurn:
    def test_final_answer(self, capsys: pytest.CaptureFixture[str]) -> None:
        history = old_history()
        run(FakeClient(response("the answer")), history)
        assert kinds(history) == ["user:old", "user:new", "message"]
        assert capsys.readouterr().out == "the answer\n"

    def test_tool_step_then_answer(self, workspace: Path) -> None:
        history = old_history()
        client = FakeClient(response(calls=[write("a")]), response("done"))
        run(client, history)
        assert kinds(history) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
            "message",
        ]
        assert (workspace / "out.txt").read_text() == "a"
        # The second request carries the tool call and its output.
        assert kinds(client.requests[1]["input"]) == [
            "user:old",
            "user:new",
            "call:a",
            "output:a:ok",
        ]

    def test_request_parameters(self, settings: config.Config) -> None:
        settings.model = "test-model"
        client = FakeClient(response("ok"))
        run(client, [], "hello", instructions="Be brief.")
        request = client.requests[0]
        assert request["model"] == "test-model"
        assert request["instructions"] == "Be brief."
        assert request["tools"] is SCHEMAS
        assert request["store"] is False
        assert kinds(request["input"]) == ["user:hello"]

    def test_instructions_are_sent_with_every_request(self) -> None:
        client = FakeClient(response(calls=[write("a")]), response("done"))
        run(client, [], instructions="Be brief.")
        assert [request["instructions"] for request in client.requests] == [
            "Be brief.",
            "Be brief.",
        ]

    def test_tool_calls_are_run_and_recorded_in_order(self, workspace: Path) -> None:
        history: list[Any] = []
        calls = [write("a", "a.txt"), write("b", "b.txt")]
        run(FakeClient(response("working", calls), response("done")), history)
        assert kinds(history) == [
            "user:new",
            "call:a",
            "call:b",
            "message",
            "output:a:ok",
            "output:b:ok",
            "message",
        ]
        assert (workspace / "a.txt").read_text() == "a"
        assert (workspace / "b.txt").read_text() == "b"

    def test_tool_failures_are_recorded_as_results(self) -> None:
        history: list[Any] = []
        call = tool_call(
            "read_file",
            "r",
            path="missing.txt",
            start_line=None,
            end_line=None,
            offset=None,
        )
        run(FakeClient(response(calls=[call]), response("done")), history)
        result = json.loads(history[2]["output"])
        assert result["ok"] is False
        assert result["message"].startswith("FileNotFoundError")

    def test_output_is_the_tool_result_as_json(self) -> None:
        history: list[Any] = []
        run(FakeClient(response(calls=[write("a")]), response("done")), history)
        assert history[2] == {
            "type": "function_call_output",
            "call_id": "a",
            "output": '{"ok": true, "message": "written 1 bytes to out.txt"}',
        }

    def test_non_ascii_output_is_not_escaped(self, workspace: Path) -> None:
        (workspace / "utf.txt").write_text("café 漢字\n", encoding="utf-8")
        call = tool_call(
            "read_file",
            "r",
            path="utf.txt",
            start_line=None,
            end_line=None,
            offset=None,
        )
        history: list[Any] = []
        run(FakeClient(response(calls=[call]), response("done")), history)
        assert "café 漢字" in history[2]["output"]
        assert "\\u" not in history[2]["output"]

    def test_streaming_prints_deltas_as_they_arrive(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.streaming = True
        history: list[Any] = []
        run(FakeClient(response("streamed answer")), history)
        assert capsys.readouterr().out == "streamed answer\n"
        assert kinds(history) == ["user:new", "message"]

    def test_streaming_with_tool_calls(
        self, settings: config.Config, workspace: Path
    ) -> None:
        settings.streaming = True
        history: list[Any] = []
        run(FakeClient(response(calls=[write("a")]), response("done")), history)
        assert kinds(history) == ["user:new", "call:a", "output:a:ok", "message"]

    def test_text_alongside_tool_calls_is_printed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run(FakeClient(response("checking", [write("a")]), response("done")), [])
        assert capsys.readouterr().out == "checking\ndone\n"

    def test_empty_response_leaves_no_trace(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        history = old_history()
        run(FakeClient(response()), history)
        assert kinds(history) == ["user:old"]
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        ("reason", "shown"),
        [("max_output_tokens", "max_output_tokens"), (None, "unknown")],
    )
    def test_incomplete_response_is_reported_and_dropped(
        self, capsys: pytest.CaptureFixture[str], reason: str | None, shown: str
    ) -> None:
        history = old_history()
        run(
            FakeClient(response("partial", status="incomplete", reason=reason)), history
        )
        assert kinds(history) == ["user:old"]
        assert capsys.readouterr().out == f"partial\n[Response incomplete: {shown}]\n"

    def test_incomplete_response_after_a_tool_step_keeps_the_step(self) -> None:
        history = old_history()
        run(
            FakeClient(
                response(calls=[write("a")]), response("partial", status="incomplete")
            ),
            history,
        )
        assert kinds(history) == ["user:old", "user:new", "call:a", "output:a:ok"]

    def test_turn_limit(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.max_turns = 2
        history: list[Any] = []
        client = FakeClient(
            response(calls=[write("a")]),
            response(calls=[write("b")]),
            response("never reached"),
        )
        run(client, history)
        assert len(client.requests) == 2
        # Tool calls from the last allowed request still run.
        assert kinds(history) == [
            "user:new",
            "call:a",
            "output:a:ok",
            "call:b",
            "output:b:ok",
        ]
        assert "Turn limit reached (2)." in capsys.readouterr().out

    def test_answer_on_the_last_allowed_turn_is_not_a_limit(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.max_turns = 2
        run(FakeClient(response(calls=[write("a")]), response("done")), [])
        assert "Turn limit" not in capsys.readouterr().out

    def test_interrupt_before_any_step_restores_history(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        history = old_history()
        run(FakeClient(KeyboardInterrupt()), history)
        assert kinds(history) == ["user:old"]
        assert "[Interrupted; returning to prompt]" in capsys.readouterr().out

    def test_interrupt_after_a_step_keeps_the_step(self) -> None:
        history = old_history()
        run(FakeClient(response(calls=[write("a")]), KeyboardInterrupt()), history)
        assert kinds(history) == ["user:old", "user:new", "call:a", "output:a:ok"]

    def test_interrupt_during_a_tool_keeps_a_complete_step(
        self,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def interrupted(**kwargs: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        monkeypatch.setitem(
            tools.TOOLS, "run_command", (interrupted, tools.RunCommandArguments)
        )
        history = old_history()
        calls = [
            write("a"),
            tool_call("run_command", "b", cmd="sleep 60", timeout=None, cwd=None),
            write("c"),
        ]
        client = FakeClient(response(calls=calls), response("never reached"))
        run(client, history)

        assert kinds(history) == [
            "user:old",
            "user:new",
            "call:a",
            "call:b",
            "call:c",
            "output:a:ok",
            "output:b:Interrupted by the user.",
            "output:c:Interrupted by the user.",
        ]
        assert len(client.requests) == 1
        assert "[Interrupted; returning to prompt]" in capsys.readouterr().out
        # The call before the interrupt ran; the one after it did not.
        assert (workspace / "out.txt").read_text() == "a"

    def test_api_error_before_any_step_restores_history(self) -> None:
        history = old_history()
        with pytest.raises(RuntimeError, match="boom"):
            run(FakeClient(RuntimeError("boom")), history)
        assert kinds(history) == ["user:old"]

    def test_api_error_after_a_step_keeps_the_step(self) -> None:
        history = old_history()
        with pytest.raises(RuntimeError, match="boom"):
            run(FakeClient(response(calls=[write("a")]), RuntimeError("boom")), history)
        assert kinds(history) == ["user:old", "user:new", "call:a", "output:a:ok"]

    def test_next_turn_follows_an_interrupted_one(self) -> None:
        history: list[Any] = []
        run(
            FakeClient(response(calls=[write("a")]), KeyboardInterrupt()),
            history,
            "first",
        )
        client = FakeClient(response("continuing"))
        run(client, history, "continue")
        assert kinds(client.requests[0]["input"]) == [
            "user:first",
            "call:a",
            "output:a:ok",
            "user:continue",
        ]

    def test_old_turns_are_trimmed_before_the_request(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.max_history_turns = 1
        history = old_history()
        client = FakeClient(response("ok"))
        run(client, history)
        assert kinds(client.requests[0]["input"]) == ["user:new"]
        assert kinds(history) == ["user:new", "message"]
        assert (
            f"[Conversation history trimmed to fit {settings.max_history_tokens} tokens]"
            in capsys.readouterr().out
        )

    def test_large_tool_outputs_are_omitted_within_a_turn(
        self, settings: config.Config, workspace: Path
    ) -> None:
        (workspace / "big.txt").write_text("x" * 6000)
        settings.max_history_tokens = 4000
        reads = [
            tool_call(
                "read_file",
                f"r{i}",
                path="big.txt",
                start_line=None,
                end_line=None,
                offset=None,
            )
            for i in range(3)
        ]
        client = FakeClient(
            *(response(calls=[read]) for read in reads), response("done")
        )
        history: list[Any] = []
        run(client, history)

        outputs = [
            item["output"]
            for item in history
            if isinstance(item, dict) and "output" in item
        ]
        assert outputs[0] == main.OMITTED_TOOL_OUTPUT
        assert outputs[-1] != main.OMITTED_TOOL_OUTPUT
        for request in client.requests:
            assert len(json.dumps(request["input"], default=str).encode()) <= 16000

    def test_oversized_prompt_stops_the_turn(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.max_history_tokens = 125
        history: list[Any] = []
        client = FakeClient(response("never reached"))
        run(client, history, "p" * 5000)
        assert client.requests == []
        assert (
            "[Turn stopped because its history exceeded the context budget]"
            in capsys.readouterr().out
        )
        assert [item["content"] for item in history] == [
            "[Previous request omitted because its history exceeded the context budget.]"
        ]


class TestTokenBudget:
    """History is limited in tokens, using the model's own count per byte."""

    def test_usage_is_reported_after_the_turn(
        self, settings: config.Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings.max_history_tokens = 64_000
        run(FakeClient(response("ok", input_tokens=12_345)), [])
        assert capsys.readouterr().out == "ok\n[Context: 12,345 of 64,000 tokens]\n"

    def test_the_last_request_of_the_turn_is_the_one_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = FakeClient(
            response(calls=[write("a")], input_tokens=100),
            response("done", input_tokens=250),
        )
        run(client, [])
        out = capsys.readouterr().out
        assert "[Context: 250 of" in out
        assert "[Context: 100 of" not in out

    def test_nothing_is_reported_without_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run(FakeClient(response("ok")), [])
        assert "Context" not in capsys.readouterr().out

    def test_usage_is_reported_even_when_interrupted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = FakeClient(
            response(calls=[write("a")], input_tokens=100), KeyboardInterrupt()
        )
        run(client, [])
        assert "[Context: 100 of" in capsys.readouterr().out

    def test_default_ratio_is_four_bytes_per_token(self) -> None:
        assert main.Conversation().tokens_per_byte == 0.25

    def test_ratio_is_learned_from_reported_usage(self) -> None:
        client = FakeClient(response("ok", input_tokens=500))
        conversation = run(client, [], "p" * 900, instructions="i" * 60)
        request = client.requests[0]
        request_bytes = (
            len(request["instructions"])
            + main.json_bytes(SCHEMAS)
            + main.json_bytes(request["input"])
        )
        assert conversation.tokens_per_byte == pytest.approx(500 / request_bytes)
        assert 0.4 < conversation.tokens_per_byte < 0.6

    def test_ratio_is_kept_without_usage(self) -> None:
        conversation = run(FakeClient(response("ok")), [])
        assert conversation.tokens_per_byte == 0.25

    @pytest.mark.parametrize(
        ("input_tokens", "expected"), [(1, 0.1), (10_000_000, 1.0)]
    )
    def test_implausible_usage_is_bounded(
        self, input_tokens: int, expected: float
    ) -> None:
        conversation = run(FakeClient(response("ok", input_tokens=input_tokens)), [])
        assert conversation.tokens_per_byte == expected

    def test_token_dense_text_is_trimmed_sooner(
        self, settings: config.Config, workspace: Path
    ) -> None:
        """The same history fits at 4 bytes per token but not at 1 byte per token."""
        (workspace / "big.txt").write_text("x" * 6000)
        settings.max_history_tokens = 4000

        def turn(tokens_per_byte: float) -> list[Any]:
            read = tool_call(
                "read_file",
                "r",
                path="big.txt",
                start_line=None,
                end_line=None,
                offset=None,
            )
            conversation = main.Conversation(tokens_per_byte=tokens_per_byte)
            client = FakeClient(response(calls=[read]), response("done"))
            main.run_turn(client, conversation, "go", SCHEMAS)  # type: ignore[arg-type]
            return client.requests[1]["input"]

        assert turn(0.25)[2]["output"] != main.OMITTED_TOOL_OUTPUT
        assert turn(1.0)[2]["output"] == main.OMITTED_TOOL_OUTPUT

    def test_instructions_and_tools_count_against_the_limit(
        self, settings: config.Config
    ) -> None:
        settings.max_history_tokens = 1000  # 4000 bytes at the default ratio
        new_message = [{"role": "user", "content": "new"}]
        # Leave room for the new message and a few bytes more, but not for "old".
        room = main.json_bytes(new_message) + 5
        instructions = "i" * (4000 - main.json_bytes(SCHEMAS) - room)
        client = FakeClient(response("ok"))
        run(client, old_history(), instructions=instructions)
        assert kinds(client.requests[0]["input"]) == ["user:new"]

        # Without the instructions the same history fits untouched.
        client = FakeClient(response("ok"))
        run(client, old_history())
        assert kinds(client.requests[0]["input"]) == ["user:old", "user:new"]

    def test_learned_ratio_carries_into_the_next_turn(
        self, settings: config.Config
    ) -> None:
        settings.max_history_tokens = 2000
        conversation = main.Conversation()
        first = FakeClient(response("a" * 10, input_tokens=10_000_000))  # 1 token/byte
        main.run_turn(first, conversation, "p" * 1900, SCHEMAS)  # type: ignore[arg-type]
        second = FakeClient(response("ok"))
        main.run_turn(second, conversation, "next", SCHEMAS)  # type: ignore[arg-type]
        # At the default ratio both turns would fit in 8000 bytes; at 1 token per
        # byte only 2000 bytes fit, so the first turn is dropped.
        assert kinds(second.requests[0]["input"]) == ["user:next"]


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
