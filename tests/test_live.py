"""End-to-end tests against the real model configured in .env.

Deselected by default because they make real API calls (and cost money on a
paid endpoint). Run them with:

    uv run pytest -m live

They use the .env settings with a temporary workspace, and assert on effects
(files written, tool calls made, facts recalled) rather than on wording.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI
from pydantic import ValidationError

from conftest import ENV_VARS
from minime.agent import Agent, TurnError
from minime.config import Config
from minime.main import build_toolbox
from minime.tools import Toolbox

pytestmark = pytest.mark.live
# Captured at import, before the autouse settings fixture clears these for
# offline tests, so `OPENAI_MODEL=... uv run pytest -m live` still applies.
SHELL_SETTINGS = {name: os.environ[name] for name in ENV_VARS if name in os.environ}


@pytest.fixture
def make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Agent]:
    """An Agent for the .env settings, working in a temporary directory.

    Keyword arguments override settings by their environment variable names,
    for example make_agent(STREAMING=False).
    """
    monkeypatch.chdir(tmp_path)

    def build(**overrides: Any) -> Agent:
        try:
            config = Config(  # pyright: ignore[reportCallIssue]
                **{**SHELL_SETTINGS, **overrides}
            )
        except ValidationError as exc:
            pytest.skip(f"no usable .env for live tests: {exc.error_count()} error(s)")
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        return Agent(config, client, build_toolbox(config))

    return build


class Turn:
    """What one turn printed and which tools it used."""

    def __init__(self, agent: Agent, prompt: str) -> None:
        self.text = ""
        self.notices: list[str] = []
        start = len(agent.context.history)
        agent.run_turn(prompt, on_text=self._text, on_notice=self.notices.append)
        self.items = agent.context.history[start:]

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
    assert agent.context.tokens_per_byte != 0.25  # learned from the reported usage


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
    agent.reset()
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
    agent.config.max_context_tokens = _limit_for(agent, 2600)

    notices: list[str] = []
    for name in ("one", "two", "three"):
        turn = Turn(agent, f"Use read_file on {name}.txt and tell me the code in it.")
        notices += turn.notices
    assert any("Summarising" in notice for notice in notices), notices

    turn = Turn(
        agent,
        "Without using any tools: what were the first and second codes?",
    )
    summaries = [
        item["content"]
        for item in agent.context.history
        if isinstance(item, dict)
        and str(item.get("content", "")).startswith("[Summary")
    ]
    assert turn.tools_called == [], turn.tools_called
    assert turn.says("AMBER", "COBALT"), (turn.text, summaries)


def _limit_for(agent: Agent, history_tokens: int) -> int:
    """A token limit leaving about history_tokens for history after the fixed
    part of every request (instructions and tool schemas)."""
    fixed = len(agent._instructions()) + len(str(agent.tool_schemas))
    return fixed // 4 + history_tokens


def test_step_limit_is_reported(
    make_agent: Callable[..., Agent], tmp_path: Path
) -> None:
    for n in range(5):
        (tmp_path / f"f{n}.txt").write_text(f"{n}\n")
    agent = make_agent(MAX_STEPS_PER_TURN=1)
    with pytest.raises(TurnError, match="Step limit reached"):
        Turn(agent, "Use read_file on f0.txt, then on f1.txt, then tell me both.")
    # The step that ran is kept, with its outputs.
    kinds = [
        getattr(item, "type", None) or item.get("type")
        for item in agent.context.history
    ]
    assert "function_call" in kinds and "function_call_output" in kinds


def test_the_program_end_to_end(tmp_path: Path) -> None:
    """The agent as a user runs it, with input piped in."""
    try:
        Config(**SHELL_SETTINGS)  # pyright: ignore[reportCallIssue]
    except ValidationError:
        pytest.skip("no usable .env for live tests")
    (tmp_path / "note.txt").write_text("marker-7431\n")
    result = subprocess.run(
        [sys.executable, "-m", "minime.main"],
        input="Read note.txt and reply with its contents only.\n/reset\n/exit\n",
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, **SHELL_SETTINGS},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "marker-7431" in result.stdout
    assert "[Context: " not in result.stdout
    assert "[History reset]" in result.stdout
    assert result.stdout.rstrip().endswith("Bye.")


def only_tools(agent: Agent, *names: str) -> Agent:
    """Offer the model only these tools, so it cannot route around the ones
    under test with grep, sed or a script."""
    agent.toolbox = Toolbox([agent.toolbox.tools[name] for name in names])
    agent.tool_schemas = agent.toolbox.schemas()
    return agent


def read_file_only(agent: Agent) -> Agent:
    return only_tools(agent, "read_file")


class TestReadFile:
    """read_file as the model actually uses it: files too long for one result,
    single lines longer than the limit, and partial reads. Only read_file is
    offered."""

    def test_long_file_is_read_to_the_end(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        lines = [
            f"{n:05d} routine log entry, nothing to see here" for n in range(1, 1201)
        ]
        lines.append("FINAL ENTRY: the password is TANGERINE-58")
        (tmp_path / "big.log").write_text("\n".join(lines) + "\n")  # about 50 KB
        turn = Turn(
            read_file_only(make_agent()),
            "big.log is long. Use read_file to read it all, part by part, and "
            "tell me the password on its final line.",
        )
        assert turn.tools_called.count("read_file") >= 3
        assert turn.says("TANGERINE-58")

    def test_single_line_longer_than_the_limit(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        fields = ",".join(f'"key{n}":{n}' for n in range(5000))
        (tmp_path / "min.json").write_text(
            "{" + fields + ',"secret_marker":"zebra-4417"}'
        )  # one line of about 50 KB
        turn = Turn(
            read_file_only(make_agent()),
            "min.json is a single very long line. Use read_file to read all of "
            "it and tell me the value of secret_marker.",
        )
        assert turn.tools_called.count("read_file") >= 3
        assert turn.says("zebra-4417")

    def test_partial_read_says_the_file_continues(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        (tmp_path / "list.txt").write_text("".join(f"item {n}\n" for n in range(1, 41)))
        turn = Turn(
            read_file_only(make_agent()),
            "Call read_file on list.txt with start_line 1 and end_line 10, once. "
            "Then, judging only from that tool result and without reading again, "
            "answer: does the file continue past line 10? Reply with yes or no.",
        )
        assert turn.tools_called == ["read_file"]
        assert turn.says("yes")

    def test_continue_from_a_partial_read(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        lines = [f"item {n}" for n in range(1, 61)]
        lines[47] = "item 48: the hidden word is MARZIPAN"
        (tmp_path / "list.txt").write_text("\n".join(lines) + "\n")
        turn = Turn(
            read_file_only(make_agent()),
            "Read list.txt with read_file 10 lines at a time (use end_line), "
            "continuing until you find the hidden word, then tell me the word.",
        )
        assert turn.tools_called.count("read_file") >= 2
        assert turn.says("MARZIPAN")


def file_tools(agent: Agent) -> Agent:
    return only_tools(agent, "read_file", "write_file", "edit_file")


class TestWriteFile:
    """write_file as the model uses it. Only the file tools are offered."""

    def test_creates_nested_directories(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        turn = Turn(
            file_tools(make_agent()),
            "Use write_file to create src/pkg/version.py containing exactly one "
            "line: VERSION = '1.4.2'",
        )
        assert "write_file" in turn.tools_called
        path = tmp_path / "src" / "pkg" / "version.py"
        assert path.read_text().strip() in {"VERSION = '1.4.2'", 'VERSION = "1.4.2"'}

    def test_replaces_an_existing_file(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "todo.txt"
        path.write_text("old task one\nold task two\n")
        Turn(
            file_tools(make_agent()),
            "Use write_file to replace the whole of todo.txt so it contains exactly "
            "two lines: buy milk, then walk dog.",
        )
        assert path.read_text().strip().splitlines() == ["buy milk", "walk dog"]

    def test_appends(self, make_agent: Callable[..., Agent], tmp_path: Path) -> None:
        # Kept strict on purpose: gemma4:cloud fails this by sending a single
        # line's trailing newline escaped ("second entry\\n"), so a literal
        # backslash-n is written. That is the model's mistake, not the tool's.
        path = tmp_path / "log.txt"
        path.write_text("first entry\n")
        turn = Turn(
            file_tools(make_agent()),
            "Use write_file with append set to true to add a line saying "
            "second entry to log.txt. Do not change the existing line.",
        )
        assert "write_file" in turn.tools_called
        assert path.read_text().splitlines() == ["first entry", "second entry"]

    def test_non_ascii_text(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        Turn(
            file_tools(make_agent()),
            "Use write_file to create menu.txt containing exactly one line: "
            "Café crème — 3,50 € — 咖啡",
        )
        assert (tmp_path / "menu.txt").read_text(encoding="utf-8").strip() == (
            "Café crème — 3,50 € — 咖啡"
        )

    def test_writing_outside_the_workspace(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
        try:
            Turn(
                file_tools(make_agent()),
                f"Use write_file to write the word hello to the path {outside}",
            )
            assert outside.read_text().strip().lower() == "hello"
        finally:
            outside.unlink(missing_ok=True)

    def test_a_longer_generated_file(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        Turn(
            file_tools(make_agent()),
            "Use write_file to create squares.csv with a header line n,square "
            "followed by one line for each n from 1 to 150, like 1,1 then 2,4.",
        )
        lines = (tmp_path / "squares.csv").read_text().strip().splitlines()
        expected = ["n,square", *(f"{n},{n * n}" for n in range(1, 151))]
        wrong = next(
            (
                (number, got, want)
                for number, (got, want) in enumerate(
                    zip(lines, expected, strict=False), 1
                )
                if got != want
            ),
            None,
        )
        # A failure here is usually the model mistyping a row, not the tool.
        assert wrong is None and len(lines) == len(expected), (wrong, len(lines))


PROGRAM = """import math


def area(radius):
    return math.pi * radius ** 2


def describe(radius):
    size = area(radius)
    return f"radius {radius} has area {size:.2f}"


if __name__ == "__main__":
    print(describe(2))
"""


class TestEditFile:
    """edit_file as the model uses it, checking the whole file afterwards so
    an edit that also changed something else fails. Only the file tools are
    offered."""

    def test_unique_replacement_leaves_the_rest_alone(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "shapes.py"
        path.write_text(PROGRAM)
        turn = Turn(
            file_tools(make_agent()),
            "In shapes.py, use edit_file to change the format in describe from "
            ":.2f to :.3f. Change nothing else.",
        )
        assert "edit_file" in turn.tools_called
        assert path.read_text() == PROGRAM.replace(":.2f", ":.3f")

    def test_multi_line_edit(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "shapes.py"
        path.write_text(PROGRAM)
        Turn(
            file_tools(make_agent()),
            "In shapes.py, use edit_file to make area() raise ValueError('negative "
            "radius') when radius is negative, before it returns. Change nothing else.",
        )
        text = path.read_text()
        assert "raise ValueError('negative radius')" in text or (
            'raise ValueError("negative radius")' in text
        )
        namespace: dict[str, object] = {}
        exec(compile(text, "shapes.py", "exec"), namespace)
        area = namespace["area"]
        assert callable(area)
        assert round(area(1), 5) == 3.14159
        try:
            area(-1)
        except ValueError:
            pass
        else:
            raise AssertionError("area(-1) did not raise")
        # Only area() changed.
        assert text.startswith("import math\n\n\ndef area(radius):\n")
        assert (
            text[text.index("def describe") :]
            == PROGRAM[PROGRAM.index("def describe") :]
        )

    def test_one_of_several_identical_lines(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "config.ini"
        original = (
            "[database]\nenabled = false\nhost = db.local\n\n"
            "[cache]\nenabled = false\nhost = cache.local\n"
        )
        path.write_text(original)
        turn = Turn(
            file_tools(make_agent()),
            "In config.ini, use edit_file to set enabled = true in the [cache] "
            "section only. The [database] section must stay as it is.",
        )
        assert "edit_file" in turn.tools_called
        assert path.read_text() == original.replace(
            "[cache]\nenabled = false", "[cache]\nenabled = true"
        )

    def test_replace_all(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "shapes.py"
        path.write_text(PROGRAM)
        Turn(
            file_tools(make_agent()),
            "In shapes.py, rename the parameter radius to r everywhere, using "
            "edit_file with replace_all. Change nothing else.",
        )
        assert path.read_text() == PROGRAM.replace("radius", "r")

    def test_deleting_lines(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "shapes.py"
        path.write_text(PROGRAM)
        Turn(
            file_tools(make_agent()),
            "In shapes.py, use edit_file to delete the whole if __name__ == "
            "'__main__' block at the end, including its print line. Change "
            "nothing else.",
        )
        assert (
            path.read_text().rstrip()
            == PROGRAM[: PROGRAM.index("if __name__")].rstrip()
        )

    def test_tab_indented_file(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "Makefile"
        original = "build:\n\tgo build ./...\n\ntest:\n\tgo test ./...\n"
        path.write_text(original)
        Turn(
            file_tools(make_agent()),
            "In Makefile, use edit_file to change the test recipe to run "
            "go test -race ./... instead. Keep the tab indentation.",
        )
        assert path.read_text() == original.replace(
            "\tgo test ./...", "\tgo test -race ./..."
        )

    def test_non_ascii_edit(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        path = tmp_path / "greeting.txt"
        path.write_text("Grüße aus München!\nSee you soon.\n", encoding="utf-8")
        Turn(
            file_tools(make_agent()),
            "In greeting.txt, use edit_file to change München to Köln. Change "
            "nothing else.",
        )
        assert path.read_text(encoding="utf-8") == "Grüße aus Köln!\nSee you soon.\n"

    def test_crlf_file(self, make_agent: Callable[..., Agent], tmp_path: Path) -> None:
        path = tmp_path / "settings.ini"
        path.write_bytes(b"[main]\r\nretries = 2\r\ncolour = blue\r\n")
        Turn(
            file_tools(make_agent()),
            "In settings.ini, change retries from 2 to 5 with edit_file. Change "
            "nothing else.",
        )
        assert path.read_bytes() == b"[main]\r\nretries = 5\r\ncolour = blue\r\n"


def shell_only(agent: Agent) -> Agent:
    return only_tools(agent, "run_command")


def calls(turn: Turn, name: str) -> list[dict[str, Any]]:
    """The arguments of each call the model made to one tool."""
    return [
        json.loads(item.arguments)
        for item in turn.items
        if getattr(item, "type", None) == "function_call" and item.name == name
    ]


def results(turn: Turn) -> list[dict[str, Any]]:
    """Every tool result of the turn, decoded."""
    return [
        json.loads(item["output"])
        for item in turn.items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


class TestRunCommand:
    """run_command as the model uses it. Only run_command is offered."""

    def test_output_is_reported(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            shell_only(make_agent()),
            f"Run this with run_command and tell me what it printed: "
            f'{sys.executable} -c "print(sum(range(101)))"',
        )
        assert "run_command" in turn.tools_called
        assert any(
            result.get("stdout", "").strip() == "5050" for result in results(turn)
        )
        assert turn.says("5050")

    def test_exit_code(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            shell_only(make_agent()),
            f"Run this with run_command and tell me its exit code as a number: "
            f'{sys.executable} -c "import sys; sys.exit(3)"',
        )
        assert any(result.get("returncode") == 3 for result in results(turn))
        assert turn.says("3")

    def test_stderr_is_reported(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            shell_only(make_agent()),
            "Run with run_command: ls definitely-missing-file-8812 . Then tell me, "
            "word for word, the name of the file the error message complains about.",
        )
        assert any(
            "definitely-missing-file-8812" in result.get("stderr", "")
            for result in results(turn)
        )
        assert turn.says("definitely-missing-file-8812")

    def test_timeout(self, make_agent: Callable[..., Agent]) -> None:
        started = time.monotonic()
        turn = Turn(
            shell_only(make_agent()),
            "Run `sleep 60` with run_command and a timeout of 2 seconds, then tell "
            "me whether it finished or timed out.",
        )
        took = time.monotonic() - started
        assert any(call.get("timeout") == 2 for call in calls(turn, "run_command"))
        assert any(result.get("timed_out") is True for result in results(turn))
        assert took < 45  # far less than the 60 seconds the command wanted
        assert turn.says("timed out") or turn.says("time out") or turn.says("timeout")

    def test_working_directory(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        (tmp_path / "sub" / "deeper").mkdir(parents=True)
        (tmp_path / "sub" / "deeper" / "here.txt").write_text("x")
        turn = Turn(
            shell_only(make_agent()),
            "Use run_command with cwd set to sub/deeper to run ls, and tell me the "
            "file name it lists.",
        )
        assert any(call.get("cwd") for call in calls(turn, "run_command"))
        assert turn.says("here.txt")

    def test_working_directory_outside_the_workspace(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        turn = Turn(
            shell_only(make_agent()),
            f"Use run_command with cwd set to {tmp_path.parent} to run pwd.",
        )
        assert any(
            result["stdout"].strip() == str(tmp_path.parent.resolve())
            for result in results(turn)
        )

    def test_long_output_keeps_its_end(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            shell_only(make_agent()),
            "Run `seq 1 200000` with run_command (do not filter or pipe it) and "
            "tell me the last number it printed.",
        )
        outputs = [result.get("stdout", "") for result in results(turn)]
        assert any("bytes omitted" in output for output in outputs)
        assert any(output.rstrip().endswith("200000") for output in outputs)
        assert turn.says("200000")

    def test_pipes_and_redirects(
        self, make_agent: Callable[..., Agent], tmp_path: Path
    ) -> None:
        Turn(
            shell_only(make_agent()),
            "With one run_command call, run: printf 'pear\\napple\\nfig\\n' | sort > "
            "sorted.txt",
        )
        assert (tmp_path / "sorted.txt").read_text() == "apple\nfig\npear\n"

    def test_commands_get_no_input(self, make_agent: Callable[..., Agent]) -> None:
        started = time.monotonic()
        turn = Turn(
            shell_only(make_agent()),
            "Run `cat` with run_command, exactly as written with no arguments and "
            "a timeout of 20 seconds, and tell me what it printed.",
        )
        # With no standard input, cat ends at once instead of waiting 20 seconds.
        assert any(
            result.get("timed_out") is False and result.get("returncode") == 0
            for result in results(turn)
        )
        assert time.monotonic() - started < 40


def web_only(agent: Agent) -> Agent:
    return only_tools(
        agent, *(n for n in ("web_fetch", "web_search") if n in agent.toolbox.tools)
    )


FUNCTIONS_PAGE = "https://docs.python.org/3/library/functions.html"


class TestWebFetch:
    """web_fetch against real pages. Only the web tools are offered."""

    def test_title_and_text(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            web_only(make_agent()),
            "Use web_fetch on https://example.com and tell me the page title and, "
            "word for word, the first sentence of its text.",
        )
        fetched = [r for r in results(turn) if r.get("ok")]
        assert fetched and fetched[0]["title"] == "Example Domain"
        assert turn.says("Example Domain", "documentation examples")

    def test_links_are_kept_and_resolved(
        self, make_agent: Callable[..., Agent]
    ) -> None:
        turn = Turn(
            web_only(make_agent()),
            "Use web_fetch on https://docs.python.org/3/library/html.parser.html and "
            "tell me the full URL that its 'Source code' link points to.",
        )
        assert turn.says("github.com/python/cpython")

    def test_redirect_and_paging_into_a_long_page(
        self, make_agent: Callable[..., Agent]
    ) -> None:
        """The page redirects, and the zip() section is near the end of about
        100,000 characters, so the model has to continue with offset."""
        turn = Turn(
            web_only(make_agent()),
            f"Use web_fetch on {FUNCTIONS_PAGE}. It is long: continue with offset "
            "until you reach the section on zip(). Then tell me which exception "
            "zip(..., strict=True) raises when the iterables have different lengths.",
        )
        offsets = [call.get("offset") or 0 for call in calls(turn, "web_fetch")]
        assert max(offsets) > 80_000, offsets
        final_urls = {r["url"] for r in results(turn) if r.get("ok")}
        assert all(url.startswith("https://docs.python.org/") for url in final_urls)
        assert final_urls != {FUNCTIONS_PAGE}  # the redirect was followed
        assert turn.says("ValueError")

    def test_json(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            web_only(make_agent()),
            "Use web_fetch on https://pypi.org/pypi/requests/json and tell me, word "
            "for word, the package's summary field.",
        )
        assert turn.says("Python HTTP for Humans")

    def test_missing_page(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            web_only(make_agent()),
            "Use web_fetch on https://example.com/definitely-missing-page-8812 and "
            "tell me the HTTP status code it gave, as a number.",
        )
        assert any("404" in r.get("message", "") for r in results(turn))
        assert turn.says("404")

    def test_binary_content_is_refused(self, make_agent: Callable[..., Agent]) -> None:
        turn = Turn(
            web_only(make_agent()),
            "Use web_fetch on https://www.python.org/static/img/python-logo.png and "
            "tell me what happened.",
        )
        assert any(
            "unsupported content type: image/png" in r.get("message", "")
            for r in results(turn)
        )


class TestWebSearch:
    """web_search against the real Brave API. Only the web tools are offered."""

    @pytest.fixture
    def agent(self, make_agent: Callable[..., Agent]) -> Agent:
        agent = make_agent()
        if "web_search" not in agent.toolbox.tools:
            pytest.skip("BRAVE_API_KEY is not set")
        return web_only(agent)

    def test_finds_an_official_site(self, agent: Agent) -> None:
        turn = Turn(
            agent,
            "Use web_search to find the official website of the uv Python package "
            "manager by Astral, and tell me its documentation domain.",
        )
        assert "web_search" in turn.tools_called
        found = [item["url"] for r in results(turn) for item in r.get("results", [])]
        assert found and all(url.startswith("http") for url in found)
        assert turn.says("astral.sh")

    def test_site_operator(self, agent: Agent) -> None:
        turn = Turn(
            agent,
            "Use web_search with the query: site:docs.python.org asyncio TaskGroup "
            "and tell me the URL of the first result.",
        )
        found = [item["url"] for r in results(turn) for item in r.get("results", [])]
        assert found
        assert all("docs.python.org" in url for url in found), found

    def test_count(self, agent: Agent) -> None:
        turn = Turn(
            agent,
            "Use web_search with count set to 3 to search for: python pathlib "
            "tutorial. Then just say how many results you got.",
        )
        assert any(call.get("count") == 3 for call in calls(turn, "web_search"))
        sizes = [len(r.get("results", [])) for r in results(turn)]
        assert sizes and all(1 <= size <= 3 for size in sizes), sizes

    def test_descriptions_are_plain_text(self, agent: Agent) -> None:
        turn = Turn(agent, "Use web_search for: python requests library")
        items = [item for r in results(turn) for item in r.get("results", [])]
        assert items
        for item in items:
            assert "<strong>" not in item["title"] + item["description"]
            assert "&amp;" not in item["title"] + item["description"]

    def test_search_then_fetch(self, agent: Agent) -> None:
        turn = Turn(
            agent,
            "Find the uv documentation page about the uv sync command with "
            "web_search, read it with web_fetch, and tell me in one sentence what "
            "its --frozen option does.",
        )
        assert "web_search" in turn.tools_called
        assert "web_fetch" in turn.tools_called
        assert turn.says("lock")
