"""End-to-end tests against the real model configured in .env.

Deselected by default because they make real API calls (and cost money on a
paid endpoint). Run them with:

    uv run pytest -m live

They use the .env settings with a temporary workspace, and assert on effects
(files written, tool calls made, facts recalled) rather than on wording.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import ENV_VARS
from pydantic import ValidationError

from agent import Agent, TurnError
from config import Config

pytestmark = pytest.mark.live
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Captured at import, before the autouse settings fixture clears these for
# offline tests, so `OPENAI_MODEL=... uv run pytest -m live` still applies.
SHELL_SETTINGS = {name: os.environ[name] for name in ENV_VARS if name in os.environ}


@pytest.fixture
def make_agent(tmp_path: Path) -> Callable[..., Agent]:
    """An Agent for the .env settings, working in a temporary directory.

    Keyword arguments override settings by their environment variable names,
    for example make_agent(STREAMING=False).
    """

    def build(**overrides: Any) -> Agent:
        try:
            config = Config(  # pyright: ignore[reportCallIssue]
                **{**SHELL_SETTINGS, "WORKSPACE_ROOT": tmp_path, **overrides}
            )
        except ValidationError as exc:
            pytest.skip(f"no usable .env for live tests: {exc.error_count()} error(s)")
        return Agent(config)

    return build


class Turn:
    """What one turn printed and which tools it used."""

    def __init__(self, agent: Agent, prompt: str) -> None:
        self.text = ""
        self.notices: list[str] = []
        start = len(agent.context.history)
        agent.run_turn(prompt, on_text=self._text, on_notice=self.notices.append)
        self.items = agent.context.history[start:]
        self.usage = agent.usage

    def _text(self, text: str) -> None:
        self.text += text

    @property
    def tools_called(self) -> list[str]:
        return [
            item.name
            for item in self.items
            if getattr(item, "type", None) == "function_call"
        ]

    def says(self, *words: str) -> bool:
        return all(word.lower() in self.text.lower() for word in words)


@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "create"])
def test_plain_answer(make_agent: Callable[..., Agent], streaming: bool) -> None:
    agent = make_agent(STREAMING=streaming)
    turn = Turn(agent, "Reply with exactly the word: pineapple")
    assert turn.says("pineapple")
    assert turn.tools_called == []
    assert turn.usage.context_tokens > 0
    assert turn.usage.output_tokens > 0


def test_read_file(make_agent: Callable[..., Agent], tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The launch code word is HERON-77.\n")
    turn = Turn(
        make_agent(),
        "Use the read_file tool on notes.txt and tell me the launch code word.",
    )
    assert "read_file" in turn.tools_called
    assert turn.says("HERON-77")


def test_write_then_edit_file(make_agent: Callable[..., Agent], tmp_path: Path) -> None:
    agent = make_agent()
    Turn(
        agent,
        "Use the write_file tool to create greeting.py containing exactly one "
        "line: print('hello')",
    )
    path = tmp_path / "greeting.py"
    assert path.exists()
    assert "print('hello')" in path.read_text() or 'print("hello")' in path.read_text()

    turn = Turn(
        agent,
        "Use the edit_file tool to change hello to goodbye in greeting.py.",
    )
    assert "edit_file" in turn.tools_called
    assert "goodbye" in path.read_text()
    assert "hello" not in path.read_text()


def test_edit_a_crlf_file(make_agent: Callable[..., Agent], tmp_path: Path) -> None:
    path = tmp_path / "settings.ini"
    path.write_bytes(b"[main]\r\nretries = 2\r\ncolour = blue\r\n")
    Turn(
        make_agent(),
        "In settings.ini, change retries from 2 to 5 using read_file and then "
        "edit_file. Change nothing else.",
    )
    assert path.read_bytes() == b"[main]\r\nretries = 5\r\ncolour = blue\r\n"


def test_run_command(make_agent: Callable[..., Agent]) -> None:
    turn = Turn(
        make_agent(),
        f"Use run_command to run: {sys.executable} -c 'print(6 * 7)' "
        "and tell me the number it printed.",
    )
    assert "run_command" in turn.tools_called
    assert turn.says("42")


def test_several_tools_in_one_turn(
    make_agent: Callable[..., Agent], tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("alpha=3\n")
    (tmp_path / "b.txt").write_text("beta=4\n")
    turn = Turn(
        make_agent(),
        "Read a.txt and b.txt with read_file, then use write_file to create "
        "sum.txt containing only the sum of alpha and beta.",
    )
    assert turn.tools_called.count("read_file") >= 2
    assert (tmp_path / "sum.txt").read_text().strip() == "7"


def test_conversation_is_remembered(make_agent: Callable[..., Agent]) -> None:
    agent = make_agent()
    Turn(agent, "Remember this: my cat is called Biscuit. Just acknowledge it.")
    turn = Turn(agent, "What is my cat called? Answer with the name only.")
    assert turn.says("Biscuit")


def test_reset_forgets(make_agent: Callable[..., Agent]) -> None:
    agent = make_agent()
    Turn(agent, "Remember this: my cat is called Biscuit. Just acknowledge it.")
    agent.context.reset()
    turn = Turn(
        agent,
        "Without using any tools: what is my cat called? If you do not know, "
        "say exactly: unknown",
    )
    assert not turn.says("Biscuit")


@pytest.mark.parametrize("effort", ["low", None], ids=["reasoning", "default"])
def test_tool_round_trip_with_and_without_reasoning(
    make_agent: Callable[..., Agent], tmp_path: Path, effort: str | None
) -> None:
    """With store=False, any reasoning items have to be replayed intact for
    the request after a tool call to be accepted."""
    (tmp_path / "n.txt").write_text("17\n")
    agent = make_agent(REASONING_EFFORT=effort)
    turn = Turn(
        agent,
        "Use read_file on n.txt, then tell me that number multiplied by 3.",
    )
    assert "read_file" in turn.tools_called
    assert turn.says("51")


def test_old_turns_are_summarised_and_remembered(
    make_agent: Callable[..., Agent], tmp_path: Path
) -> None:
    """A small limit forces earlier turns out; the summary keeps their facts."""
    filler = "This line is only here to take up space in the file.\n" * 120
    (tmp_path / "one.txt").write_text("The first code is AMBER.\n" + filler)
    (tmp_path / "two.txt").write_text("The second code is COBALT.\n" + filler)
    (tmp_path / "three.txt").write_text("The third code is DAHLIA.\n" + filler)
    agent = make_agent()
    # Room for the instructions and tools plus roughly one file read.
    agent.config.max_history_tokens = _limit_for(agent, 2600)

    notices: list[str] = []
    for name in ("one", "two", "three"):
        turn = Turn(agent, f"Use read_file on {name}.txt and tell me the code in it.")
        notices += turn.notices
    assert any("Summarising" in notice for notice in notices)

    turn = Turn(
        agent,
        "Without using any tools: what were the first and second codes?",
    )
    assert turn.tools_called == []
    assert turn.says("AMBER", "COBALT")


def _limit_for(agent: Agent, history_tokens: int) -> int:
    """A token limit leaving about history_tokens for history after the fixed
    part of every request (instructions and tool schemas)."""
    fixed = len(agent.instructions()) + len(str(agent.tool_schemas))
    return fixed // 4 + history_tokens


def test_web_fetch(make_agent: Callable[..., Agent]) -> None:
    turn = Turn(
        make_agent(),
        "Use web_fetch on https://example.com and tell me the page's main heading.",
    )
    assert "web_fetch" in turn.tools_called
    assert turn.says("Example Domain")


def test_web_search(make_agent: Callable[..., Agent]) -> None:
    agent = make_agent()
    if "web_search" not in agent.toolbox.tools:
        pytest.skip("BRAVE_API_KEY is not set")
    turn = Turn(
        agent,
        "Use web_search to find the official website of the Python programming "
        "language and tell me its domain.",
    )
    assert "web_search" in turn.tools_called
    assert turn.says("python.org")


def test_turn_limit_is_reported(
    make_agent: Callable[..., Agent], tmp_path: Path
) -> None:
    for n in range(5):
        (tmp_path / f"f{n}.txt").write_text(f"{n}\n")
    agent = make_agent(MAX_TURNS_PER_RUN=1)
    with pytest.raises(TurnError, match="Turn limit reached"):
        Turn(agent, "Use read_file on f0.txt, then on f1.txt, then tell me both.")
    # The step that ran is kept, with its outputs.
    kinds = [
        getattr(item, "type", None) or item.get("type")
        for item in agent.context.history
    ]
    assert "function_call" in kinds and "function_call_output" in kinds


def test_the_program_end_to_end(tmp_path: Path) -> None:
    """main.py as a user runs it, with input piped in."""
    try:
        Config(**{**SHELL_SETTINGS, "WORKSPACE_ROOT": tmp_path})  # pyright: ignore[reportCallIssue]
    except ValidationError:
        pytest.skip("no usable .env for live tests")
    (tmp_path / "note.txt").write_text("marker-7431\n")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py")],
        input="Read note.txt and reply with its contents only.\n/clear\nexit\n",
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, **SHELL_SETTINGS, "WORKSPACE_ROOT": str(tmp_path)},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "marker-7431" in result.stdout
    assert "[Context: " in result.stdout
    assert "[Conversation cleared]" in result.stdout
    assert result.stdout.rstrip().endswith("Goodbye!")
