import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import IO, Any

from pydantic import Field

from minime.tools import MAX_TOOL_OUTPUT_BYTES, Tool, ToolArguments

# Bytes set aside for the "[... N bytes omitted ...]" line in clipped output.
OMISSION_MARKER_ROOM = 48


def read_bounded_output(stream: IO[Any]) -> tuple[bytes, bytes, int]:
    """Read a stream to its end, keeping its first and last halves of the limit.

    Returns the head, the tail, and how many bytes between them were dropped.
    """
    half = MAX_TOOL_OUTPUT_BYTES // 2
    head = bytearray()
    tail = bytearray()
    omitted = 0
    while chunk := stream.read(8192):
        room = half - len(head)
        head.extend(chunk[:room])
        tail.extend(chunk[room:])
        if len(tail) > half:
            omitted += len(tail) - half
            del tail[:-half]
    return bytes(head), bytes(tail), omitted


def render_output(head: bytes, tail: bytes, omitted: int, budget: int) -> str:
    """Decode captured output, keeping its start and end if it exceeds budget bytes."""
    if omitted:
        head_text = head.decode("utf-8", errors="replace")
        tail_text = tail.decode("utf-8", errors="replace")
    else:
        head_text = tail_text = (head + tail).decode("utf-8", errors="replace")
        if len(head_text.encode("utf-8")) <= budget:
            return head_text

    # Cut on bytes; errors="ignore" drops a character split at either cut.
    keep = (budget - OMISSION_MARKER_ROOM) // 2
    kept_head = head_text.encode("utf-8")[:keep].decode("utf-8", errors="ignore")
    kept_tail = tail_text.encode("utf-8")[-keep:].decode("utf-8", errors="ignore")
    dropped = (
        omitted
        + len(head)
        + len(tail)
        - len(kept_head.encode("utf-8"))
        - len(kept_tail.encode("utf-8"))
    )
    return f"{kept_head}\n[... {dropped} bytes omitted ...]\n{kept_tail}"


def kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


class RunCommandArguments(ToolArguments):
    cmd: str = Field(description="The shell command to run.")
    timeout: int | None = Field(ge=1, description="Maximum execution time in seconds.")
    cwd: str | None = Field(description="Optional working directory for the command.")


class RunCommandTool(Tool[RunCommandArguments]):
    """Run a shell command, including pipes and redirects, with an optional timeout and working directory."""

    name = "run_command"
    arguments_model = RunCommandArguments

    def __init__(self, default_timeout: int) -> None:
        self.default_timeout = default_timeout

    def run(self, args: RunCommandArguments) -> dict[str, Any]:
        timeout = self.default_timeout if args.timeout is None else args.timeout

        process = subprocess.Popen(
            args.cmd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=args.cwd,
            start_new_session=os.name == "posix",
        )
        assert process.stdout is not None
        assert process.stderr is not None
        with ThreadPoolExecutor(max_workers=2) as executor:
            stdout_future = executor.submit(read_bounded_output, process.stdout)
            stderr_future = executor.submit(read_bounded_output, process.stderr)
            deadline = time.monotonic() + timeout
            try:
                returncode: int | None = process.wait(timeout=timeout)
                _, pending = wait(
                    (stdout_future, stderr_future),
                    timeout=max(0, deadline - time.monotonic()),
                )
                if pending:
                    # A background child outlived the shell and still holds the pipes.
                    raise subprocess.TimeoutExpired(args.cmd, timeout)
            except subprocess.TimeoutExpired:
                kill_process_tree(process)
                returncode = None
            except KeyboardInterrupt:
                kill_process_tree(process)
                raise
            stdout_capture = stdout_future.result()
            stderr_capture = stderr_future.result()

        # One limit covers both streams: each may use what the other leaves, and
        # neither gets less than half.
        half = MAX_TOOL_OUTPUT_BYTES // 2
        stdout_size, stderr_size = (
            MAX_TOOL_OUTPUT_BYTES if omitted else len(head) + len(tail)
            for head, tail, omitted in (stdout_capture, stderr_capture)
        )
        return {
            "ok": returncode == 0,
            "timed_out": returncode is None,
            "stdout": render_output(
                *stdout_capture, max(half, MAX_TOOL_OUTPUT_BYTES - stderr_size)
            ),
            "stderr": render_output(
                *stderr_capture, max(half, MAX_TOOL_OUTPUT_BYTES - stdout_size)
            ),
            "returncode": returncode,
        }
