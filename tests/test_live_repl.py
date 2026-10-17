"""The real agent on a pseudo-terminal, talking to the real model in .env.

Run with `uv run pytest -m live`. Keys are typed as a user would, so Ctrl-C
goes through the terminal as a real SIGINT.
"""

import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import CTRL_C, CTRL_D, PROMPT_AGAIN, Terminal, process_is_gone
from minime.config import Config
from test_live import SHELL_SETTINGS

pytestmark = [
    pytest.mark.live,
    pytest.mark.pty,
    pytest.mark.skipif(os.name != "posix", reason="needs a pseudo-terminal"),
]
PROMPT = r"> $"


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Callable[..., Terminal]]:
    """Start `minime` for the .env settings in a temporary workspace.

    Keyword arguments override settings by their environment variable names.
    """
    sessions: list[Terminal] = []

    def run(**overrides: str) -> Terminal:
        settings = {**SHELL_SETTINGS, "WORKSPACE_ROOT": str(tmp_path), **overrides}
        try:
            Config(**settings)  # pyright: ignore[reportCallIssue]
        except ValidationError:
            pytest.skip("no usable .env for live tests")
        env = {**os.environ, **settings}
        session = Terminal([sys.executable, "-m", "minime.main"], env, tmp_path)
        sessions.append(session)
        session.expect(PROMPT, timeout=30)
        return session

    yield run
    for session in sessions:
        session.close()


def answer(terminal: Terminal, prompt: str, timeout: float = 120) -> str:
    """Type a prompt and return everything printed until the next prompt."""
    terminal.type(prompt)
    terminal.expect(r"\[Context: [^\]]*\][\s\S]*" + PROMPT, timeout=timeout)
    return terminal.screen


def test_banner_turn_commands_and_exit(start: Callable[..., Terminal]) -> None:
    terminal = start()
    first_screen = terminal.screen

    screen = answer(terminal, "Reply with exactly the word: pineapple")
    assert "pineapple" in screen.lower()
    assert re.search(r"\[Context: [\d,]+ of [\d,]+ tokens", screen)

    terminal.type("   ")  # a blank line is ignored: no request, just a new prompt
    terminal.expect(PROMPT)
    assert "Context" not in terminal.screen

    terminal.type("/reset")
    terminal.expect(r"\[Conversation cleared\][\s\S]*" + PROMPT)

    terminal.type("/exit")
    terminal.expect(r"Goodbye!")
    assert terminal.wait_for_exit() == 0
    assert "AI coding agent started" in first_screen


def test_memory_and_reset(start: Callable[..., Terminal]) -> None:
    terminal = start()
    answer(terminal, "Remember this: my cat is called Biscuit. Just acknowledge it.")
    assert (
        "biscuit"
        in answer(terminal, "What is my cat called? Answer with the name only.").lower()
    )

    terminal.type("/reset")
    terminal.expect(r"\[Conversation cleared\][\s\S]*" + PROMPT)
    screen = answer(
        terminal,
        "Without using any tools: what is my cat called? If you do not know, "
        "say exactly: unknown",
    )
    assert "biscuit" not in screen.lower()


def test_ctrl_c_while_an_answer_streams(start: Callable[..., Terminal]) -> None:
    terminal = start()
    terminal.type("Count from 1 to 500, one number per line, with no other text.")
    terminal.expect(r"\n12\r?\n", timeout=120)  # the answer is arriving
    started = time.monotonic()
    terminal.type(CTRL_C)
    terminal.expect(PROMPT_AGAIN, timeout=15)
    assert time.monotonic() - started < 10
    assert "\n500" not in terminal.screen
    # Even when library code swallows the interrupt, no traceback is shown.
    assert "Traceback" not in terminal.screen
    assert "Exception ignored" not in terminal.screen

    # The agent carries on normally afterwards.
    assert "still here" in answer(terminal, "Reply with exactly: still here").lower()


def test_ctrl_c_while_a_command_runs(
    start: Callable[..., Terminal], tmp_path: Path
) -> None:
    terminal = start()
    terminal.type(
        "Use run_command to run exactly this, with a timeout of 120 seconds: "
        "echo $$ > pid; exec sleep 100"
    )
    pid_file = tmp_path / "pid"
    terminal.wait_for(
        lambda: pid_file.exists() and pid_file.read_text().strip(),
        "the command never started",
        timeout=120,
    )
    command_pid = int(pid_file.read_text())

    started = time.monotonic()
    terminal.type(CTRL_C)
    terminal.expect(PROMPT_AGAIN, timeout=15)
    assert time.monotonic() - started < 10  # not the 100 seconds it wanted
    terminal.wait_for(
        lambda: process_is_gone(command_pid), "the command survived Ctrl-C", timeout=10
    )

    # The model is told its command was interrupted.
    screen = answer(
        terminal,
        "Without using any tools: did the command you just ran finish, or was it "
        "interrupted? Answer in one short sentence.",
    )
    assert "interrupt" in screen.lower()


def test_ctrl_c_while_the_model_thinks(start: Callable[..., Terminal]) -> None:
    terminal = start(REASONING_EFFORT="high", STREAMING="true")
    terminal.type(
        "Think carefully, then answer: how many prime numbers are there between "
        "1000 and 1100? Show your reasoning."
    )
    terminal.expect(r"\[thinking…\]", timeout=120)
    terminal.type(CTRL_C)
    terminal.expect(PROMPT_AGAIN, timeout=15)
    assert "still here" in answer(terminal, "Reply with exactly: still here").lower()


def test_two_quick_ctrl_cs_do_not_crash(start: Callable[..., Terminal]) -> None:
    terminal = start()
    terminal.type("Count from 1 to 500, one number per line, with no other text.")
    terminal.expect(r"\n12\r?\n", timeout=120)
    terminal.type(CTRL_C)
    time.sleep(0.02)
    terminal.type(CTRL_C)
    # Either both land in the turn, or the second one quits at the prompt;
    # what must not happen is a traceback.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not re.search(
        r"> $|Goodbye!", terminal.screen
    ):
        terminal.read(0.1)
    assert "Traceback" not in terminal.screen, terminal.screen
    if "Goodbye!" in terminal.screen:
        assert terminal.wait_for_exit() == 0


@pytest.mark.parametrize("key", [CTRL_C, CTRL_D], ids=["ctrl-c", "ctrl-d"])
def test_quitting_from_the_prompt(start: Callable[..., Terminal], key: bytes) -> None:
    terminal = start()
    answer(terminal, "Reply with exactly: ok")
    terminal.type(key)
    terminal.expect(r"Goodbye!")
    assert terminal.wait_for_exit() == 0
