import io
import os
import re
import time
from pathlib import Path

import pytest

from conftest import Tools, process_is_gone
from minime import config, shell, tools

pytestmark = pytest.mark.skipif(os.name != "posix", reason="uses POSIX shell commands")

LIMIT = tools.MAX_TOOL_OUTPUT_BYTES
OMITTED = re.compile(r"\n\[\.\.\. (\d+) bytes omitted \.\.\.\]\n")


def capture(data: bytes) -> tuple[bytes, bytes, int]:
    return shell.read_bounded_output(io.BytesIO(data))


def render(data: bytes, budget: int = LIMIT) -> str:
    return shell.render_output(*capture(data), budget)


def omitted_count(output: str) -> int:
    match = OMITTED.search(output)
    assert match, "no omission marker"
    return int(match.group(1))


class TestReadBoundedOutput:
    def test_short_output_is_kept_whole(self) -> None:
        assert capture(b"hello\n") == (b"hello\n", b"", 0)

    def test_empty_output(self) -> None:
        assert capture(b"") == (b"", b"", 0)

    def test_output_at_the_limit_is_kept_whole(self) -> None:
        head, tail, omitted = capture(b"a" * LIMIT)
        assert (head + tail, omitted) == (b"a" * LIMIT, 0)

    def test_long_output_keeps_half_the_limit_from_each_end(self) -> None:
        data = b"H" * (LIMIT // 2) + b"m" * 100_000 + b"T" * (LIMIT // 2)
        assert capture(data) == (b"H" * (LIMIT // 2), b"T" * (LIMIT // 2), 100_000)

    def test_memory_is_bounded_for_endless_streams(self) -> None:
        head, tail, omitted = capture(b"x" * 5_000_000)
        assert len(head) + len(tail) == LIMIT
        assert omitted == 5_000_000 - LIMIT


class TestRenderOutput:
    def test_short_output_is_unchanged(self) -> None:
        assert render(b"hello\n") == "hello\n"

    def test_empty_output(self) -> None:
        assert render(b"") == ""

    def test_output_at_the_budget_is_unchanged(self) -> None:
        assert render(b"a" * LIMIT) == "a" * LIMIT

    def test_long_output_keeps_both_ends(self) -> None:
        output = render(b"H" * 5000 + b"m" * (LIMIT * 3) + b"T" * 5000)
        assert len(output.encode()) <= LIMIT
        assert output.startswith("H" * 5000)
        assert output.endswith("T" * 5000)

    @pytest.mark.parametrize("extra", [1, 1000, 100_000])
    def test_omitted_count_is_exact(self, extra: int) -> None:
        data = b"a" * (LIMIT + extra)
        output = render(data)
        assert len(output.encode()) <= LIMIT
        assert omitted_count(output) == len(data) - output.count("a")

    def test_smaller_budget_clips_output_that_was_captured_whole(self) -> None:
        data = b"H" * 1000 + b"X" * 8000 + b"T" * 1000
        output = render(data, budget=4096)
        assert len(output.encode()) <= 4096
        assert output.startswith("H" * 1000)
        assert output.endswith("T" * 1000)
        assert omitted_count(output) == len(data) - sum(output.count(c) for c in "HXT")

    def test_invalid_bytes_are_replaced(self) -> None:
        assert render(b"ok \xff\xfe end") == "ok \ufffd\ufffd end"

    def test_multibyte_output_is_cut_between_characters(self) -> None:
        output = render("é".encode() * LIMIT)
        assert len(output.encode()) <= LIMIT
        assert "\ufffd" not in output


class TestRunCommand:
    def test_success(self, tool: Tools) -> None:
        assert tool.run_command("echo hello") == {
            "ok": True,
            "timed_out": False,
            "stdout": "hello\n",
            "stderr": "",
            "returncode": 0,
        }

    def test_nonzero_exit(self, tool: Tools) -> None:
        result = tool.run_command("exit 3")
        assert (result["ok"], result["returncode"], result["timed_out"]) == (
            False,
            3,
            False,
        )

    def test_stdout_and_stderr_are_separate(self, tool: Tools) -> None:
        result = tool.run_command("echo out; echo err >&2")
        assert (result["stdout"], result["stderr"]) == ("out\n", "err\n")

    def test_pipes_and_redirects(self, tool: Tools, workspace: Path) -> None:
        result = tool.run_command(
            "printf 'b\\na\\n' | sort > sorted.txt && cat sorted.txt"
        )
        assert result["stdout"] == "a\nb\n"
        assert (workspace / "sorted.txt").read_text() == "a\nb\n"

    def test_unknown_command(self, tool: Tools) -> None:
        result = tool.run_command("definitely-not-a-command-xyz")
        assert result["ok"] is False
        assert result["returncode"] == 127
        assert result["stderr"]

    def test_runs_in_the_workspace_by_default(
        self, tool: Tools, workspace: Path
    ) -> None:
        assert tool.run_command("pwd")["stdout"].strip() == str(workspace.resolve())

    def test_relative_cwd(self, tool: Tools, workspace: Path) -> None:
        (workspace / "sub").mkdir()
        assert tool.run_command("pwd", cwd="sub")["stdout"].strip() == str(
            workspace.resolve() / "sub"
        )

    def test_cwd_outside_the_working_directory(
        self, tool: Tools, workspace: Path
    ) -> None:
        assert tool.run_command("pwd", cwd="..")["stdout"].strip() == str(
            workspace.resolve().parent
        )

    def test_cwd_in_the_home_directory(
        self, tool: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert tool.run_command("pwd", cwd="~")["stdout"].strip() == str(
            tmp_path.resolve()
        )

    def test_cwd_must_be_a_directory(self, tool: Tools, workspace: Path) -> None:
        (workspace / "file.txt").write_text("x")
        with pytest.raises(NotADirectoryError):
            tool.run_command("pwd", cwd="file.txt")
        with pytest.raises(FileNotFoundError):
            tool.run_command("pwd", cwd="missing")

    @pytest.mark.parametrize("timeout", [0, -5])
    def test_timeout_must_be_positive(self, tool: Tools, timeout: int) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            tool.run_command("echo hi", timeout=timeout)

    def test_stdin_is_not_inherited(self, tool: Tools) -> None:
        # Give this process a stdin that never ends; a command that inherited
        # it would block on `cat` until the timeout.
        saved = os.dup(0)
        read_end, write_end = os.pipe()
        os.dup2(read_end, 0)
        try:
            result = tool.run_command("cat; echo done", timeout=2)
        finally:
            os.dup2(saved, 0)
            for descriptor in (saved, read_end, write_end):
                os.close(descriptor)
        assert result["timed_out"] is False
        assert result["stdout"] == "done\n"

    def test_long_output_keeps_the_end(self, tool: Tools) -> None:
        result = tool.run_command("seq 1 20000; echo FINAL_ERROR_LINE")
        stdout = result["stdout"]
        assert result["ok"] is True
        assert len(stdout.encode()) <= LIMIT
        assert stdout.startswith("1\n2\n3\n")
        assert stdout.endswith("20000\nFINAL_ERROR_LINE\n")
        assert OMITTED.search(stdout)

    def test_streams_share_one_limit(self, tool: Tools) -> None:
        result = tool.run_command("seq 1 20000; seq 1 20000 >&2")
        stdout, stderr = result["stdout"], result["stderr"]
        assert len(stdout.encode()) + len(stderr.encode()) <= LIMIT
        for stream in (stdout, stderr):
            assert stream.startswith("1\n2\n")
            assert stream.endswith("19999\n20000\n")
            assert OMITTED.search(stream)

    def test_quiet_stream_leaves_its_share_to_the_other(self, tool: Tools) -> None:
        result = tool.run_command("seq 1 20000; echo warning >&2")
        assert result["stderr"] == "warning\n"
        assert len(result["stdout"].encode()) > LIMIT * 0.9
        assert len(result["stdout"].encode()) + len(result["stderr"].encode()) <= LIMIT

    def test_neither_stream_is_squeezed_below_half(self, tool: Tools) -> None:
        # 10 KB on each stream: both must be clipped, to half the limit each.
        result = tool.run_command(
            "head -c 10000 /dev/zero | tr '\\0' a; head -c 10000 /dev/zero | tr '\\0' b >&2"
        )
        for stream, letter in ((result["stdout"], "a"), (result["stderr"], "b")):
            assert LIMIT // 2 - 100 < len(stream.encode()) <= LIMIT // 2
            assert stream.startswith(letter) and stream.endswith(letter)

    def test_timeout_kills_the_command(self, tool: Tools) -> None:
        started = time.monotonic()
        result = tool.run_command("echo before; sleep 30", timeout=1)
        assert time.monotonic() - started < 5
        assert result == {
            "ok": False,
            "timed_out": True,
            "stdout": "before\n",
            "stderr": "",
            "returncode": None,
        }

    def test_default_timeout_comes_from_settings(
        self, tool: Tools, settings: config.Config
    ) -> None:
        settings.command_timeout = 1
        started = time.monotonic()
        assert tool.run_command("sleep 30")["timed_out"] is True
        assert time.monotonic() - started < 5

    def test_timeout_with_endless_output(self, tool: Tools) -> None:
        result = tool.run_command("yes", timeout=1)
        assert result["timed_out"] is True
        assert len(result["stdout"].encode()) <= LIMIT

    def test_timeout_when_output_is_redirected(self, tool: Tools) -> None:
        started = time.monotonic()
        result = tool.run_command("sleep 30 > /dev/null 2>&1", timeout=1)
        assert result["timed_out"] is True
        assert time.monotonic() - started < 5

    def test_background_child_holding_the_pipes_times_out(self, tool: Tools) -> None:
        started = time.monotonic()
        result = tool.run_command("sleep 30 & echo started", timeout=1)
        assert time.monotonic() - started < 5
        assert result["timed_out"] is True
        assert result["stdout"] == "started\n"

    def test_timeout_kills_child_processes(self, tool: Tools, workspace: Path) -> None:
        result = tool.run_command("sleep 30 & echo $! > child.pid; wait", timeout=1)
        assert result["timed_out"] is True
        assert process_is_gone(int((workspace / "child.pid").read_text()), 3)

    def test_interrupt_kills_the_command_and_propagates(
        self, tool: Tools, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt

        # The shell exits at once; Ctrl-C then arrives while output is still being read.
        monkeypatch.setattr(shell, "wait", interrupted)
        with pytest.raises(KeyboardInterrupt):
            tool.run_command("sleep 30 & echo $! > child.pid", timeout=10)
        assert process_is_gone(int((workspace / "child.pid").read_text()), 3)
