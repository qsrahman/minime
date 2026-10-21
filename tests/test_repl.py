"""The terminal: Repl over a FakeAgent or a real agent on the fake API.

What the user sees (banner, prompt, notices, tool lines, spinner, Markdown,
colours, titles) and how input is read and dispatched to the commands.
"""

import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import openai
import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from conftest import (
    Api,
    answer,
    call_item,
    kinds,
    message_item,
    response_body,
    run,
    write,
)
from minime import config, main
from minime import repl as repl_module
from minime.agent import SUMMARY_REQUEST, Agent, TurnError
from minime.context import SUMMARY_MARKER
from minime.main import build_toolbox
from minime.repl import Repl, home_relative


class FakeAgent:
    """All that Repl uses of an agent: run_turn() and reset()."""

    def __init__(self) -> None:
        self.model = "fake-model"
        self.prompts: list[str] = []
        self.resets = 0

    def run_turn(
        self,
        prompt: str,
        on_text: Callable[[str], None],
        on_notice: Callable[[str], None],
        on_tool_start: Callable[[str], None],
        on_tool_end: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.prompts.append(prompt)
        if prompt == "explode":
            raise RuntimeError("boom")
        if prompt == "give up":
            raise TurnError("Step limit reached (16)")
        on_text(f"answer to {prompt}\n")

    def reset(self) -> None:
        self.resets += 1


def repl_session[AgentT: (FakeAgent, Agent)](
    monkeypatch: pytest.MonkeyPatch,
    *inputs: str | BaseException,
    agent: AgentT | None = None,
) -> AgentT:
    """Run a Repl, by default over a FakeAgent, typing these inputs at its
    prompt; "/exit" is typed last unless the inputs end the session."""
    queue = [*inputs, "/exit"]

    def fake_input(prompt: str) -> str:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    used = agent if agent is not None else FakeAgent()
    Repl(cast(Agent, used)).run()
    return cast(AgentT, used)


class TestRepl:
    def test_main_wires_up_the_agent(
        self, settings: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings.model = "wired-model"
        settings.base_url = "http://localhost:9/v1"
        built: dict[str, Any] = {}
        monkeypatch.setattr(main, "Config", lambda **kwargs: settings)
        monkeypatch.setattr(
            main, "OpenAI", lambda **kwargs: built.setdefault("client", kwargs)
        )
        monkeypatch.setattr(
            main,
            "Repl",
            lambda agent: SimpleNamespace(run=lambda: built.update(agent=agent)),
        )
        main.main()
        assert built["client"] == {
            "api_key": "test-key",
            "base_url": "http://localhost:9/v1",
        }
        agent = built["agent"]
        assert agent.client is built["client"]
        assert list(agent.toolbox.tools) == list(build_toolbox(settings).tools)
        assert agent.model == "wired-model"

    def test_banner_and_farewell(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl_session(monkeypatch, "/exit")
        out = capsys.readouterr().out
        assert out.startswith(f"minime · fake-model · {home_relative(Path.cwd())}\n")
        assert "Type /help to list commands." in out
        assert out.endswith("\nBye.\n")

    @pytest.mark.parametrize("word", ["/exit", "/quit", "/EXIT", "  /Quit  "])
    def test_exit_words(self, monkeypatch: pytest.MonkeyPatch, word: str) -> None:
        # Had the loop gone on, "never read" would have been sent as a prompt.
        assert repl_session(monkeypatch, word, "never read").prompts == []

    def test_end_of_input_at_the_prompt_quits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl_session(monkeypatch, EOFError(), "never read")
        assert capsys.readouterr().out.count("Bye.") == 1

    def test_one_ctrl_c_at_the_prompt_does_not_quit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(monkeypatch, KeyboardInterrupt(), "still here")
        assert agent.prompts == ["still here"]
        assert "Press Ctrl-C again to quit, or Ctrl-D." in capsys.readouterr().out

    def test_two_quick_ctrl_cs_at_the_prompt_quit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(
            monkeypatch, KeyboardInterrupt(), KeyboardInterrupt(), "never read"
        )
        assert agent.prompts == []
        assert capsys.readouterr().out.endswith("\nBye.\n")

    def test_two_slow_ctrl_cs_do_not_quit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(repl_module, "QUIT_SECONDS", 0)
        agent = repl_session(
            monkeypatch, KeyboardInterrupt(), KeyboardInterrupt(), "still here"
        )
        assert agent.prompts == ["still here"]

    def test_ctrl_c_outside_a_turn_is_reported_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupted(self: Repl, prompt: str) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(Repl, "ask", interrupted)
        repl_session(monkeypatch, "hello")
        out = capsys.readouterr().out
        assert "[Interrupted]" in out and out.endswith("Bye.\n")

    def test_draft_starts_the_next_prompt_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        defaults: list[str] = []
        inputs = ["/exit", "", "hi"]

        class Session:
            def prompt(self, default: str = "") -> str:
                defaults.append(default)
                return inputs.pop()

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(Repl, "prompt_session", lambda self: Session())
        repl = Repl(cast(Agent, FakeAgent()))
        repl.draft = "edit me"
        repl.run()
        assert defaults == ["edit me", "", ""]

    def test_prompts_go_to_the_agent_and_answers_are_printed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(monkeypatch, "first", "second", "/exit")
        assert agent.prompts == ["first", "second"]
        out = capsys.readouterr().out
        assert "\nanswer to first\n\n" in out
        assert "\nanswer to second\n\n" in out

    @pytest.mark.parametrize("blank", ["", "   ", "\t", " \t "])
    def test_blank_prompts_are_not_sent(
        self, monkeypatch: pytest.MonkeyPatch, blank: str
    ) -> None:
        assert repl_session(monkeypatch, blank, "real", "/exit").prompts == ["real"]

    @pytest.mark.parametrize(
        ("prompt", "notice"),
        [
            ("explode", "[Agent error: RuntimeError: boom]"),
            ("give up", "[Step limit reached (16)]"),
        ],
    )
    def test_errors_are_reported_and_the_loop_continues(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        prompt: str,
        notice: str,
    ) -> None:
        agent = repl_session(monkeypatch, prompt, "after", "/exit")
        assert agent.prompts == [prompt, "after"]
        assert notice in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["/reset", "/RESET", "  /reset  "])
    def test_reset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        agent = repl_session(monkeypatch, "one", command, "two", "/exit")
        assert agent.resets == 1
        assert agent.prompts == ["one", "two"]  # /reset is not sent to the model
        assert "[History reset]" in capsys.readouterr().out

    def test_help_lists_every_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(monkeypatch, "/help", "/exit")
        out = capsys.readouterr().out
        for command in ("/clear", "/exit", "/help", "/quit", "/reset"):
            assert f"  {command} " in out
        assert agent.prompts == []

    def test_help_lists_commands_alphabetically(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl_session(monkeypatch, "/help")
        names = re.findall(r"^  (/\w+)", capsys.readouterr().out, re.MULTILINE)
        assert len(names) == len(Repl(cast(Agent, FakeAgent())).commands)
        assert names == sorted(names)
        assert names.index("/model") < names.index("/quit") < names.index("/reset")

    @pytest.mark.parametrize("tty", [True, False])
    def test_clear_wipes_the_screen_and_keeps_the_history(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tty: bool,
    ) -> None:
        monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
        agent = repl_session(monkeypatch, "one", "/clear", "two", "/exit")
        assert agent.prompts == ["one", "two"]
        assert agent.resets == 0
        # Escape codes only reach a terminal, never redirected output.
        assert ("\033[2J" in capsys.readouterr().out) is tty

    @pytest.mark.parametrize("word", ["/rest", "/tmp/notes.txt"])
    def test_unknown_command_is_reported_and_not_sent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        word: str,
    ) -> None:
        assert repl_session(monkeypatch, word, "/exit").prompts == []
        assert f"[Unknown command: {word}. Type /help" in capsys.readouterr().out

    def test_slash_prompt_with_spaces_goes_to_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = repl_session(monkeypatch, "/usr/bin is missing", "/exit")
        assert agent.prompts == ["/usr/bin is missing"]


class TestCommands:
    """The slash commands that act on a real agent (see commands.py)."""

    def test_help_shows_arguments(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl_session(monkeypatch, "/help")
        out = capsys.readouterr().out
        for usage in (
            "/model [name]",
            "/effort [level]",
            "/cd <dir>",
            "/undo",
            "/export [file]",
        ):
            assert f"  {usage} " in out

    def test_command_without_its_argument(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(monkeypatch, "/cd")
        assert "[Usage: /cd <dir>]" in capsys.readouterr().out
        assert agent.prompts == []

    def test_argument_to_a_command_that_takes_none(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(monkeypatch, "/reset now")
        assert "[/reset takes no argument]" in capsys.readouterr().out
        assert (agent.resets, agent.prompts) == (0, [])

    def test_model_is_shown_and_switched(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        api.replies = [answer()]
        agent = repl_session(
            monkeypatch, "/model", "/model other-model", "hi", agent=api.agent()
        )
        out = capsys.readouterr().out
        assert "[Model: gpt-5.4-mini]" in out
        assert "[Model set to other-model]" in out
        assert agent.model == "other-model"
        assert api.requests[0]["model"] == "other-model"

    def test_effort_is_shown_set_and_cleared(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        api.replies = [answer(), answer()]
        repl_session(
            monkeypatch,
            "/effort",
            "/effort HIGH",
            "first",
            "/effort default",
            "second",
            agent=api.agent(),
        )
        out = capsys.readouterr().out
        assert "[Reasoning effort: model default]" in out
        assert "[Reasoning effort set to high]" in out
        assert api.requests[0]["reasoning"] == {"effort": "high"}
        assert "reasoning" not in api.requests[1]

    def test_unknown_effort_is_refused(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        agent = repl_session(monkeypatch, "/effort extreme", agent=api.agent())
        assert agent.reasoning_effort is None
        assert "Unknown effort 'extreme'; use one of: none, minimal" in (
            capsys.readouterr().out
        )

    def test_cd(
        self,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (workspace / "sub").mkdir()
        repl_session(monkeypatch, "/cd sub")
        assert Path.cwd() == (workspace / "sub").resolve()
        assert f"[Working directory: {Path.cwd()}]" in capsys.readouterr().out

    def test_cd_to_a_missing_directory_is_reported(
        self,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repl_session(monkeypatch, "/cd missing")
        assert Path.cwd() == workspace.resolve()
        assert "[/cd failed: FileNotFoundError:" in capsys.readouterr().out

    def test_retry_sends_the_last_prompt_again(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = repl_session(monkeypatch, "/retry", "give up", "/retry")
        assert agent.prompts == ["give up", "give up"]
        assert "[Nothing to retry]" in capsys.readouterr().out

    def test_undo_removes_only_the_last_turn(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        api.replies = [response_body(write("a")), answer(), answer()]
        agent = repl_session(
            monkeypatch, "one", "two", "/undo", "/undo", "/undo", agent=api.agent()
        )
        assert agent.context.history == []
        out = capsys.readouterr().out
        assert out.count("[Last turn removed]") == 2
        assert "[Nothing to undo]" in out

    def test_edit_removes_the_last_turn_and_drafts_its_prompt(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer(), answer()]
        repl = Repl(api.agent())
        repl.handle("/edit")
        assert "[Nothing to edit]" in capsys.readouterr().out
        repl.ask("one")
        repl.ask("two\nlines")
        repl.handle("/edit")
        assert repl.draft == "two\nlines"
        assert kinds(repl.agent.context.history) == ["user:one", "message"]

    def test_export_writes_the_conversation_as_markdown(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [
            response_body(message_item("Writing it."), write("a")),
            response_body(
                call_item(
                    "read_file",
                    "b",
                    path="nope.txt",
                    start_line=None,
                    end_line=None,
                    offset=None,
                )
            ),
            answer("All **done**."),
        ]
        repl = Repl(api.agent())
        repl.handle("/export")
        assert "[Nothing to export]" in capsys.readouterr().out
        repl.ask("make\nthe file")
        repl.handle("/export notes/chat.md")
        assert "[Conversation exported to notes/chat.md]" in capsys.readouterr().out
        text = Path("notes/chat.md").read_text()
        assert text.startswith("# MiniMe conversation, ")
        assert "\n---\n\n> make\n> the file\n\nWriting it.\n\n" in text
        assert "- ✓ `write_file: out.txt`\n" in text
        assert "- ✗ `read_file: nope.txt`\n" in text
        assert text.endswith("All **done**.\n")

    def test_export_has_a_default_file_name(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer()]
        repl = Repl(api.agent())
        repl.ask("hi")
        repl.handle("/export")
        [exported] = Path.cwd().glob("minime-*.md")
        assert re.fullmatch(r"minime-\d{8}-\d{6}\.md", exported.name)

    @pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")
    def test_shell_command_output_is_shown_and_kept_for_the_model(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [answer()]
        repl = Repl(api.agent())
        repl.handle("!echo hi")
        out = capsys.readouterr().out
        assert re.fullmatch(r"\nhi\n✓ \$ echo hi  \d+\.\ds\n\n", out)
        assert api.requests == []  # the model is not asked
        repl.ask("what did it print?")
        assert kinds(api.requests[0]["input"]) == [
            "user:I ran this shell command myself (exit code 0):\n$ echo hi\nstdout:\nhi",
            "user:what did it print?",
        ]

    @pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")
    def test_failed_shell_command_shows_stderr_once(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(api.agent())
        repl.handle("!echo oops >&2; exit 2")
        out = capsys.readouterr().out
        assert out.count("\noops\n") == 1  # the output line, not the command
        assert re.search(r"✗ \$ echo oops >&2; exit 2  exit 2  \d+\.\ds", out)
        [note] = repl.agent.context.history
        assert note["content"].startswith(
            "I ran this shell command myself (exit code 2)"
        )
        assert note["content"].endswith("stderr:\noops")

    def test_bare_bang_shows_usage(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(api.agent())
        repl.handle("!")
        assert "[Usage: !<command>]" in capsys.readouterr().out
        assert repl.agent.context.history == []

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
    def test_diff_shows_changes_and_untracked_files(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                check=True,
                capture_output=True,
            )

        repl = Repl(api.agent())
        repl.handle("/diff")
        assert "[Not a git repository]" in capsys.readouterr().out
        git("init", "-q")
        repl.handle("/diff")
        assert "[No changes]" in capsys.readouterr().out
        Path("a.txt").write_text("old\n")
        git("add", "a.txt")
        repl.handle("/diff")  # staged, before the first commit
        assert "+old" in capsys.readouterr().out
        git("commit", "-q", "-m", "first")
        Path("a.txt").write_text("new\n")
        Path("new file.txt").write_text("x")
        repl.handle("/diff")
        out = capsys.readouterr().out
        assert "-old" in out and "+new" in out
        assert "[Untracked: new file.txt]" in out

    def test_undo_keeps_earlier_turns(
        self, api: Api, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api.replies = [answer(), answer()]
        agent = repl_session(monkeypatch, "one", "two", "/undo", agent=api.agent())
        assert kinds(agent.context.history) == ["user:one", "message"]

    def test_compact_replaces_the_history_with_a_summary(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        api.replies = [answer("one done"), answer("They asked for one.")]
        agent = repl_session(monkeypatch, "one", "/compact", agent=api.agent())
        assert kinds(agent.context.history) == [
            f"user:{SUMMARY_MARKER}\nThey asked for one."
        ]
        summary_request = api.requests[1]
        assert kinds(summary_request["input"]) == [
            "user:one",
            "message",
            f"user:{SUMMARY_REQUEST}",
        ]
        assert summary_request["tool_choice"] == "none"
        assert "[History compacted]" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("reply", "notice"),
        [
            (
                httpx2.Response(500, json={"error": {"message": "down"}}),
                "[No summary (InternalServerError); the history is unchanged]",
            ),
            (
                answer("  "),
                "[The summary was empty or too large; the history is unchanged]",
            ),
            (KeyboardInterrupt(), "[Interrupted; returning to prompt]"),
        ],
        ids=["api-error", "empty", "interrupt"],
    )
    def test_failed_compact_keeps_the_history(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        reply: Any,
        notice: str,
    ) -> None:
        api.replies = [answer("one done"), reply]
        agent = repl_session(monkeypatch, "one", "/compact", agent=api.agent())
        assert kinds(agent.context.history) == ["user:one", "message"]
        assert notice in capsys.readouterr().out

    def test_compact_with_no_history(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repl_session(monkeypatch, "/compact", agent=api.agent())
        assert api.requests == []
        assert "[Nothing to compact]" in capsys.readouterr().out

    def test_context(
        self,
        api: Api,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        api.replies = [answer(input_tokens=1234, cached_tokens=1000)]
        agent = repl_session(
            monkeypatch, "/context", "one", "/context", agent=api.agent()
        )
        out = capsys.readouterr().out
        assert (
            "[Context: limit 65,536 tokens, no request since the last change; "
            "0 turns; 4.0 bytes per token]"
        ) in out
        per_token = 1 / agent.context.tokens_per_byte
        assert (
            "[Context: 1,234 of 65,536 tokens (2%), 1,000 cached; 1 turns; "
            f"{per_token:.1f} bytes per token]"
        ) in out


class TestNotices:
    """How Repl lays out the model's text and its own bracketed notices."""

    def test_notice_after_text_starts_a_new_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(cast(Agent, FakeAgent()))
        repl.write("partial answer")
        repl.notice("Interrupted; returning to prompt")
        assert (
            capsys.readouterr().out
            == "partial answer\n[Interrupted; returning to prompt]\n"
        )

    def test_notice_at_a_line_start_adds_no_blank_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(cast(Agent, FakeAgent()))
        repl.write("full line\n")
        repl.notice("one")
        repl.notice("two")
        assert capsys.readouterr().out == "full line\n[one]\n[two]\n"

    def test_answer_is_rerendered_as_markdown_in_a_terminal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("TERM", "xterm")
        repl = Repl(cast(Agent, FakeAgent()))
        repl.console = Console(
            force_terminal=True, color_system=None, width=40, height=10
        )
        repl.write("# Title\nsome **bold** text\n")
        repl.notice("next")
        raw, rendered = capsys.readouterr().out.split("\033[2F\033[J")
        assert raw == "# Title\nsome **bold** text\n"
        assert "Title" in rendered and "#" not in rendered
        assert "some bold text" in rendered
        assert rendered.endswith("[next]\n")

    def test_wrapped_lines_are_counted_as_screen_rows(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("TERM", "xterm")
        repl = Repl(cast(Agent, FakeAgent()))
        repl.console = Console(
            force_terminal=True, color_system=None, width=10, height=10
        )
        repl.write("x" * 25)  # three rows, and no newline yet
        repl.end_answer()
        assert "x" * 25 + "\n\033[3F\033[J" in capsys.readouterr().out

    def test_answer_taller_than_the_screen_stays_raw(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("TERM", "xterm")
        repl = Repl(cast(Agent, FakeAgent()))
        repl.console = Console(
            force_terminal=True, color_system=None, width=40, height=3
        )
        repl.write("# one\ntwo\nthree\n")
        repl.end_answer()
        assert capsys.readouterr().out == "# one\ntwo\nthree\n"

    def test_dumb_terminal_gets_raw_text(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("TERM", "dumb")
        repl = Repl(cast(Agent, FakeAgent()))
        repl.console = Console(
            force_terminal=True, color_system=None, width=40, height=10
        )
        repl.write("# Title\n")
        repl.end_answer()
        assert capsys.readouterr().out == "# Title\n"


def interactive_repl(monkeypatch: pytest.MonkeyPatch, agent: Any = None) -> Repl:
    """A Repl whose console is a 40x10 colour terminal."""
    monkeypatch.setenv("TERM", "xterm")
    repl = Repl(cast(Agent, agent or FakeAgent()))
    repl.console = Console(
        force_terminal=True,
        color_system="standard",
        width=40,
        height=10,
        theme=repl_module.THEME,
    )
    return repl


class TestToolCallLines:
    def test_each_call_is_shown_with_its_outcome(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api.replies = [response_body(write("a"), write("b", "other.txt")), answer()]
        run(api)
        assert re.search(
            r"\n✓ write_file: out\.txt  \d+\.\ds\n"
            r"✓ write_file: other\.txt  \d+\.\ds\ndone\n",
            capsys.readouterr().out,
        )

    @pytest.mark.parametrize(
        ("cmd", "failure"),
        [("exit 3", "exit 3"), ("sleep 5", "timed out")],
    )
    @pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")
    def test_a_failed_command_shows_why(
        self, api: Api, capsys: pytest.CaptureFixture[str], cmd: str, failure: str
    ) -> None:
        api.replies = [
            response_body(call_item("run_command", "c", cmd=cmd, timeout=1, cwd=None)),
            answer(),
        ]
        run(api)
        assert re.search(
            rf"\n✗ run_command: {cmd}  {failure}  \d+\.\ds\n", capsys.readouterr().out
        )

    @pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")
    def test_a_failed_command_shows_the_end_of_its_output(
        self, api: Api, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd = "for i in 1 2 3 4 5 6; do echo line $i >&2; done; exit 1"
        api.replies = [
            response_body(
                call_item("run_command", "c", cmd=cmd, timeout=None, cwd=None)
            ),
            answer(),
        ]
        run(api)
        out = capsys.readouterr().out
        assert re.search(
            r"exit 1  \d+\.\ds\n    line 3\n    line 4\n    line 5\n    line 6\ndone",
            out,
        )

    def test_a_successful_call_shows_no_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(cast(Agent, FakeAgent()))
        repl.tool_start("run_command: ls")
        repl.tool_end("run_command: ls", {"ok": True, "stdout": "a\nb", "stderr": ""})
        assert "    a" not in capsys.readouterr().out

    def test_a_tool_error_shows_its_message_shortened(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = Repl(cast(Agent, FakeAgent()))
        repl.tool_start("read_file: x")
        repl.tool_end("read_file: x", {"ok": False, "message": "No such file " * 10})
        out = capsys.readouterr().out
        assert out.startswith("✗ read_file: x  No such file No such file")
        assert re.search(r" …  \d+\.\ds\n$", out)

    def test_ctrl_c_marks_the_running_call_interrupted(
        self,
        api: Api,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def interrupt(call: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        api.replies = [response_body(write("a"))]
        agent = api.agent()
        monkeypatch.setattr(agent.toolbox, "execute", interrupt)
        Repl(agent).ask("go")
        assert re.search(
            r"\n✗ write_file: out\.txt  interrupted  \d+\.\ds\n"
            r"\[Interrupted; returning to prompt\]\n",
            capsys.readouterr().out,
        )


class TestLook:
    """Spinner, colours and code blocks in an interactive terminal."""

    def test_spinner_runs_until_text_arrives_and_leaves_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.start_spinner("thinking")
        assert repl.spinner is not None and repl.spinner.live.is_started
        repl.write("hi\n")
        assert repl.spinner is None
        out = capsys.readouterr().out
        assert out.endswith("hi\n")
        assert "thinking" not in out.split("\x1b[2K")[-1]  # erased before the text

    def test_no_spinner_when_output_is_not_a_terminal(self) -> None:
        repl = Repl(cast(Agent, FakeAgent()))
        repl.start_spinner("thinking")
        assert repl.spinner is None

    def test_spinner_shows_the_running_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.console.width = 80
        repl.tool_start("run_command: make")
        assert repl.spinner is not None
        assert repl.spinner.tasks[0].description == "run_command: make"
        repl.tool_end("run_command: make", {"ok": True})
        assert repl.spinner.tasks[0].description == "thinking"
        repl.stop_spinner()

    def test_spinner_says_how_to_stop_and_fits_one_row(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.start_spinner("run_command: " + "x" * 80)
        assert repl.spinner is not None
        repl.spinner.refresh()
        repl.stop_spinner()
        drawn = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", capsys.readouterr().out)
        row = next(line for line in drawn.split("\r") if "ctrl-c to stop" in line)
        assert "…" in row
        assert len(row.rstrip()) <= 40

    def test_long_tool_line_is_shortened_to_one_row(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.tool_start("run_command: " + "x" * 80)
        repl.stop_spinner()
        repl.tool_end("run_command: " + "x" * 80, {"ok": False, "returncode": 2})
        repl.stop_spinner()
        plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", capsys.readouterr().out)
        [line] = re.findall(r"✗ run_command: x+…  exit 2  \d+\.\ds", plain)
        assert len(line) == 40

    @pytest.mark.parametrize(
        ("style", "code"),
        [("dim", "\x1b[2m"), ("yellow", "\x1b[33m"), ("red", "\x1b[31m")],
    )
    def test_notices_are_coloured_by_kind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        style: str,
        code: str,
    ) -> None:
        interactive_repl(monkeypatch).notice("note", style)
        assert capsys.readouterr().out == f"{code}[note]\x1b[0m\n"

    @pytest.mark.parametrize(
        ("prompt", "code"), [("explode", "\x1b[31m"), ("give up", "\x1b[33m")]
    )
    def test_errors_are_red_and_warnings_yellow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        prompt: str,
        code: str,
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.ask(prompt)
        repl.stop_spinner()
        assert f"{code}[" in capsys.readouterr().out

    def test_answers_are_at_most_100_columns_wide(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.console.width = 150
        repl.write("word " * 60 + "\n")
        repl.end_answer()
        rendered = capsys.readouterr().out.split("\x1b[J", 1)[1]
        lines = rendered.rstrip("\n").split("\n")
        assert len(lines) == 3
        assert max(len(line.rstrip()) for line in lines) <= 100

    def test_headings_are_left_aligned(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.write("# Title\n\n## Sub\n")
        repl.end_answer()
        rendered = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
        rendered = rendered.split("\x1b[J", 1)[1]
        assert rendered.startswith("Title ")
        assert "\nSub " in rendered

    def test_code_has_no_background(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.write("Use `x`:\n```python\nprint(1)\n```\n")
        repl.end_answer()
        rendered = capsys.readouterr().out.split("\x1b[J", 1)[1]
        assert "print" in rendered
        assert not re.search(r"\x1b\[[0-9;]*(4\d|48;)", rendered)  # no background


def status_error(kind: type[openai.APIStatusError], status: int) -> Exception:
    request = httpx2.Request("POST", "http://localhost/v1/responses")
    return kind("nope", response=httpx2.Response(status, request=request), body=None)


class TestFinishing:
    def test_title_says_working_during_a_turn(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl = interactive_repl(monkeypatch)
        repl.ask("hi")
        out = capsys.readouterr().out
        working = out.index("\x1b]0;minime · working…\x07")
        assert working < out.index("answer to hi") < out.index("\x1b]0;minime\x07")
        assert "\a" not in out.replace("\x07", "")  # a short turn rings no bell

    def test_long_turn_rings_the_bell(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(repl_module, "LONG_TURN_SECONDS", -1)
        repl = interactive_repl(monkeypatch)
        repl.ask("hi")
        assert capsys.readouterr().out.endswith("\x1b]0;minime\x07\x07")

    def test_session_saves_and_restores_the_title(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr(
            repl_module,
            "Console",
            lambda **kwargs: Console(force_terminal=True, **kwargs),
        )
        repl_session(monkeypatch)
        out = capsys.readouterr().out
        assert "\x1b[22;0t\x1b]0;minime\x07" in out
        assert out.endswith("Bye.\n\x1b[23;0t")

    def test_no_title_outside_a_terminal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repl_session(monkeypatch, "hi")
        assert "\x1b" not in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("error", "hint"),
        [
            (
                openai.APIConnectionError(
                    request=httpx2.Request("POST", "http://localhost/v1/responses")
                ),
                "OPENAI_BASE_URL",
            ),
            (
                openai.APITimeoutError(
                    request=httpx2.Request("POST", "http://localhost/v1/responses")
                ),
                "OPENAI_BASE_URL",
            ),
            (status_error(openai.AuthenticationError, 401), "OPENAI_API_KEY"),
            (status_error(openai.PermissionDeniedError, 403), "access to the model"),
            (status_error(openai.NotFoundError, 404), "/model"),
            (status_error(openai.RateLimitError, 429), "quota"),
            (status_error(openai.InternalServerError, 500), "/retry"),
        ],
    )
    def test_api_errors_come_with_a_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        error: Exception,
        hint: str,
    ) -> None:
        def fail(*args: Any, **kwargs: Any) -> None:
            raise error

        agent = FakeAgent()
        monkeypatch.setattr(agent, "run_turn", fail)
        Repl(cast(Agent, agent)).ask("hi")
        lines = capsys.readouterr().out.split("\n")
        error_line = next(i for i, line in enumerate(lines) if "Agent error" in line)
        assert lines[error_line + 1].startswith("  ")
        assert hint in lines[error_line + 1]

    def test_other_errors_have_no_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        Repl(cast(Agent, FakeAgent())).ask("explode")
        assert capsys.readouterr().out == "\n[Agent error: RuntimeError: boom]\n\n"


class TestBanner:
    @pytest.mark.parametrize(("relative", "shown"), [(".", "~"), ("a/b", "~/a/b")])
    def test_home_is_shortened(self, relative: str, shown: str) -> None:
        assert home_relative(Path.home() / relative) == shown

    def test_other_paths_are_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert home_relative(tmp_path / "work") == str(tmp_path / "work")


class TestPromptSession:
    @pytest.fixture
    def session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> PromptSession[str]:
        monkeypatch.setattr(repl_module, "HISTORY_FILE", tmp_path / "history")
        return Repl(cast(Agent, FakeAgent())).prompt_session()

    def test_commands_are_completed_with_their_help_text(
        self, session: PromptSession[str]
    ) -> None:
        assert session.completer is not None
        [completion] = session.completer.get_completions(
            Document("/und"), CompleteEvent()
        )
        assert completion.text == "/undo"
        assert completion.display_meta_text == "remove the last turn from the history"

    def test_completion_menu_is_alphabetical(self, session: PromptSession[str]) -> None:
        assert session.completer is not None
        names = [
            c.text
            for c in session.completer.get_completions(Document("/"), CompleteEvent())
        ]
        assert names == sorted(names) and "/edit" in names

    def test_earlier_prompts_are_suggested(self, session: PromptSession[str]) -> None:
        session.history.append_string("explain the tests")
        assert session.auto_suggest is not None
        buffer = Buffer(history=session.history)
        buffer.insert_text("expl")
        suggestion = session.auto_suggest.get_suggestion(buffer, buffer.document)
        assert suggestion is not None and suggestion.text == "ain the tests"

    def test_toolbar_is_quiet_with_key_hints_at_the_right(
        self, session: PromptSession[str], api: Api
    ) -> None:
        assert session.style is not None
        attrs = session.style.get_attrs_for_style_str("class:bottom-toolbar")
        assert not attrs.reverse and attrs.dim
        repl = Repl(api.agent())
        status = repl.status()
        wide = repl.toolbar(120)
        assert wide.startswith(status) and wide.endswith(repl_module.KEY_HINTS)
        assert len(wide) == 119
        assert repl.toolbar(40) == status  # no room for the hints

    def test_prompt_is_a_coloured_arrow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(repl_module, "HISTORY_FILE", tmp_path / "history")
        session = Repl(cast(Agent, FakeAgent())).prompt_session()
        assert session.message == [("class:prompt", "› ")]
        assert session.enable_open_in_editor
        assert session.style is not None
        assert session.style.get_attrs_for_style_str("class:prompt").color == "ansicyan"

    def test_status_shows_model_effort_and_context_use(self, api: Api) -> None:
        agent = api.agent()
        repl = Repl(agent)
        assert repl.status() == f"{agent.model} · effort default"
        agent.reasoning_effort = "low"
        api.replies = [
            response_body(message_item("hi"), input_tokens=1000, cached_tokens=250)
        ]
        agent.run_turn("hello")
        limit = agent.config.max_context_tokens
        assert repl.status() == (
            f"{agent.model} · effort low · 1,000/{limit:,} tokens · 25% cached"
        )

    def test_completes_commands_and_directories_after_cd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(repl_module, "HISTORY_FILE", tmp_path / "history")
        (Path.cwd() / "subdir").mkdir()
        (Path.cwd() / "file.txt").write_text("")
        completer = Repl(cast(Agent, FakeAgent())).prompt_session().completer
        assert completer is not None

        def complete(text: str) -> list[str]:
            return [
                c.text
                for c in completer.get_completions(Document(text), CompleteEvent())
            ]

        assert complete("/mo") == ["/model"]
        assert complete("/cd su") == ["bdir"]
        assert complete("why /mo") == []
