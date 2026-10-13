import io
import os
import re
import time
from pathlib import Path

import pytest

import config
import tools

pytestmark = pytest.mark.skipif(os.name != "posix", reason="uses POSIX shell commands")

LIMIT = tools.MAX_TOOL_OUTPUT_BYTES
OMITTED = re.compile(r"\n\[\.\.\. (\d+) bytes omitted \.\.\.\]\n")


def process_is_gone(pid: int, wait_seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class TestReadBoundedOutput:
    def test_short_output_is_unchanged(self) -> None:
        assert tools.read_bounded_output(io.BytesIO(b"hello\n")) == "hello\n"

    def test_empty_output(self) -> None:
        assert tools.read_bounded_output(io.BytesIO(b"")) == ""

    def test_output_at_the_limit_is_unchanged(self) -> None:
        assert tools.read_bounded_output(io.BytesIO(b"a" * LIMIT)) == "a" * LIMIT

    def test_long_output_keeps_both_ends(self) -> None:
        data = b"H" * 5000 + b"m" * (LIMIT * 3) + b"T" * 5000
        output = tools.read_bounded_output(io.BytesIO(data))
        assert len(output.encode()) <= LIMIT
        assert output.startswith("H" * 5000)
        assert output.endswith("T" * 5000)
        assert OMITTED.search(output)

    def test_one_byte_over_the_limit_is_marked(self) -> None:
        output = tools.read_bounded_output(io.BytesIO(b"a" * (LIMIT + 1)))
        assert len(output.encode()) <= LIMIT
        assert OMITTED.search(output)

    def test_invalid_bytes_are_replaced(self) -> None:
        assert tools.read_bounded_output(io.BytesIO(b"ok \xff\xfe end")) == "ok �� end"

    def test_multibyte_output_stays_within_the_limit(self) -> None:
        output = tools.read_bounded_output(io.BytesIO("é".encode() * LIMIT))
        assert len(output.encode()) <= LIMIT
        assert "�" not in output

class TestRunCommand:
    def test_success(self) -> None:
        assert tools.run_command("echo hello") == {
            "ok": True,
            "timed_out": False,
            "stdout": "hello\n",
            "stderr": "",
            "returncode": 0,
        }

    def test_nonzero_exit(self) -> None:
        result = tools.run_command("exit 3")
        assert (result["ok"], result["returncode"], result["timed_out"]) == (False, 3, False)

    def test_stdout_and_stderr_are_separate(self) -> None:
        result = tools.run_command("echo out; echo err >&2")
        assert (result["stdout"], result["stderr"]) == ("out\n", "err\n")

    def test_pipes_and_redirects(self, workspace: Path) -> None:
        result = tools.run_command("printf 'b\\na\\n' | sort > sorted.txt && cat sorted.txt")
        assert result["stdout"] == "a\nb\n"
        assert (workspace / "sorted.txt").read_text() == "a\nb\n"

    def test_unknown_command(self) -> None:
        result = tools.run_command("definitely-not-a-command-xyz")
        assert result["ok"] is False
        assert result["returncode"] == 127
        assert result["stderr"]

    def test_runs_in_the_workspace_by_default(self, workspace: Path) -> None:
        assert tools.run_command("pwd")["stdout"].strip() == str(workspace.resolve())

    def test_relative_cwd(self, workspace: Path) -> None:
        (workspace / "sub").mkdir()
        assert tools.run_command("pwd", cwd="sub")["stdout"].strip() == str(workspace.resolve() / "sub")

    def test_cwd_outside_the_workspace(self) -> None:
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.run_command("pwd", cwd="..")

    def test_cwd_must_be_a_directory(self, workspace: Path) -> None:
        (workspace / "file.txt").write_text("x")
        with pytest.raises(ValueError, match="not a directory"):
            tools.run_command("pwd", cwd="file.txt")
        with pytest.raises(ValueError, match="not a directory"):
            tools.run_command("pwd", cwd="missing")

    @pytest.mark.parametrize("timeout", [0, -5])
    def test_timeout_must_be_positive(self, timeout: int) -> None:
        with pytest.raises(ValueError, match="timeout must be greater than zero"):
            tools.run_command("echo hi", timeout=timeout)

    def test_stdin_is_not_inherited(self) -> None:
        # Give this process a stdin that never ends; a command that inherited
        # it would block on `cat` until the timeout.
        saved = os.dup(0)
        read_end, write_end = os.pipe()
        os.dup2(read_end, 0)
        try:
            result = tools.run_command("cat; echo done", timeout=2)
        finally:
            os.dup2(saved, 0)
            for descriptor in (saved, read_end, write_end):
                os.close(descriptor)
        assert result["timed_out"] is False
        assert result["stdout"] == "done\n"

    def test_long_output_keeps_the_end(self) -> None:
        result = tools.run_command("seq 1 20000; echo FINAL_ERROR_LINE")
        stdout = result["stdout"]
        assert result["ok"] is True
        assert len(stdout.encode()) <= LIMIT
        assert stdout.startswith("1\n2\n3\n")
        assert stdout.endswith("20000\nFINAL_ERROR_LINE\n")
        assert OMITTED.search(stdout)

    def test_timeout_kills_the_command(self) -> None:
        started = time.monotonic()
        result = tools.run_command("echo before; sleep 30", timeout=1)
        assert time.monotonic() - started < 5
        assert result == {
            "ok": False,
            "timed_out": True,
            "stdout": "before\n",
            "stderr": "",
            "returncode": None,
        }

    def test_default_timeout_comes_from_settings(self, settings: config.Config) -> None:
        settings.command_timeout = 1
        started = time.monotonic()
        assert tools.run_command("sleep 30")["timed_out"] is True
        assert time.monotonic() - started < 5

    def test_timeout_with_endless_output(self) -> None:
        result = tools.run_command("yes", timeout=1)
        assert result["timed_out"] is True
        assert len(result["stdout"].encode()) <= LIMIT

    def test_timeout_when_output_is_redirected(self) -> None:
        started = time.monotonic()
        result = tools.run_command("sleep 30 > /dev/null 2>&1", timeout=1)
        assert result["timed_out"] is True
        assert time.monotonic() - started < 5

    def test_background_child_holding_the_pipes_times_out(self) -> None:
        started = time.monotonic()
        result = tools.run_command("sleep 30 & echo started", timeout=1)
        assert time.monotonic() - started < 5
        assert result["timed_out"] is True
        assert result["stdout"] == "started\n"

    def test_timeout_kills_child_processes(self, workspace: Path) -> None:
        result = tools.run_command("sleep 30 & echo $! > child.pid; wait", timeout=1)
        assert result["timed_out"] is True
        assert process_is_gone(int((workspace / "child.pid").read_text()))

    def test_interrupt_kills_the_command_and_propagates(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt

        # The shell exits at once; Ctrl-C then arrives while output is still being read.
        monkeypatch.setattr(tools, "wait", interrupted)
        with pytest.raises(KeyboardInterrupt):
            tools.run_command("sleep 30 & echo $! > child.pid", timeout=10)
        assert process_is_gone(int((workspace / "child.pid").read_text()))
