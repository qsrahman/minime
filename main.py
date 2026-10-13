import inspect
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cache
from itertools import islice
from pathlib import Path
from typing import IO, Any, cast

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseInputParam
from openai.types.responses.tool_param import ToolParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_HISTORY_TURNS = 12
MAX_HISTORY_BYTES = 64 * 1024
MAX_TOOL_OUTPUT_BYTES = 16 * 1024


class Config(BaseSettings):
    """Configuration for the AI coding agent."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        extra="ignore",
    )
    api_key: str = Field(
        min_length=1,
        validation_alias="OPENAI_API_KEY",
    )
    model: str = Field(
        default="gpt-5.4-mini",
        min_length=1,
        validation_alias="OPENAI_MODEL",
    )
    base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    max_turns: int = Field(
        default=16,
        gt=0,
        validation_alias="MAX_TURNS_PER_RUN",
    )
    command_timeout: int = Field(
        default=30,
        gt=0,
        validation_alias="COMMAND_TIMEOUT_SECONDS",
    )
    streaming: bool = Field(
        default=True,
        validation_alias="STREAMING",
    )
    workspace_root: Path = Field(
        default_factory=Path.cwd,
        validation_alias="WORKSPACE_ROOT",
    )


@cache
def get_config() -> Config:
    """Load and validate the agent configuration on first use."""
    try:
        return Config()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None


system_prompt = """
You are an expert AI coding agent.

Operating rules:
- Use paths relative to the workspace root. File tools reject paths outside it.
- Prefer the smallest correct change.
- Validate with the smallest relevant command or check.
- Do not make unnecessary edits or broad refactors.
- If a tool call fails, explain the failure clearly and continue with the next best action.
- Keep the user informed with concise progress updates when a task involves multiple steps.
- Use the available tools to inspect, edit, and verify the code accurately.
- Be concise and direct.
"""


def resolve_workspace_path(path: str) -> Path:
    """Resolve a path and reject locations outside the configured workspace."""
    workspace_root = get_config().workspace_root.resolve()
    file_path = Path(path)
    resolved_path = (
        file_path if file_path.is_absolute() else workspace_root / file_path
    ).resolve()
    if not resolved_path.is_relative_to(workspace_root):
        raise ValueError(f"Path is outside the workspace: {path}")
    return resolved_path


def read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> dict[str, Any]:
    """Read a file, optionally limiting the result to an inclusive line range."""
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be greater than zero")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be greater than zero")
    if start_line is not None and end_line is not None and end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line")

    file_path = resolve_workspace_path(path)
    first_line = 1 if start_line is None else start_line

    with file_path.open(encoding="utf-8") as file:
        content = "".join(islice(file, first_line - 1, end_line))

    encoded_content = content.encode("utf-8")
    if len(encoded_content) > MAX_TOOL_OUTPUT_BYTES:
        content = encoded_content[:MAX_TOOL_OUTPUT_BYTES].decode(
            "utf-8", errors="ignore"
        )
        content += (
            f"\n[content truncated after {MAX_TOOL_OUTPUT_BYTES} bytes; "
            "use start_line and end_line to read the rest]"
        )

    return {"ok": True, "content": content}


def write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """Write content to a file, replacing it by default or appending when append is true."""
    file_path = resolve_workspace_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding="utf-8") as file:
        file.write(content)

    operation = "appended" if append else "written"
    return {
        "ok": True,
        "message": f"{operation} {len(content.encode('utf-8'))} bytes to {file_path}",
    }


def edit_file(
    path: str, old_content: str, new_content: str, replace_all: bool = False
) -> dict[str, Any]:
    """Replace a unique text fragment, or replace all matches when replace_all is true."""
    if not old_content:
        raise ValueError("old_content must not be empty")

    file_path = resolve_workspace_path(path)
    existing_content = file_path.read_text(encoding="utf-8", newline="")

    match_count = existing_content.count(old_content)
    if match_count == 0:
        raise ValueError(f"old_content not found in {path}")
    if match_count > 1 and not replace_all:
        raise ValueError(
            f"old_content must match exactly once in {path}; found {match_count} matches"
        )

    file_path.write_text(
        existing_content.replace(old_content, new_content),
        encoding="utf-8",
        newline="",
    )

    return {
        "ok": True,
        "message": f"replaced {match_count} match(es) in {path}.",
    }


def read_bounded_output(stream: IO[Any]) -> str:
    output = bytearray()
    truncated = False
    while chunk := stream.read(8192):
        remaining = MAX_TOOL_OUTPUT_BYTES - len(output)
        output.extend(chunk[:remaining])
        truncated = truncated or len(chunk) > remaining

    result = output.decode("utf-8", errors="replace")
    if truncated:
        result += f"\n[output truncated after {MAX_TOOL_OUTPUT_BYTES} bytes]"
    return result


def kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


def run_command(
    cmd: str, timeout: int | None = None, cwd: str | None = None
) -> dict[str, Any]:
    """Run a shell command, including pipes and redirects, with an optional timeout and working directory."""
    settings = get_config()
    timeout = settings.command_timeout if timeout is None else timeout
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    working_directory = resolve_workspace_path(cwd or str(settings.workspace_root))
    if not working_directory.is_dir():
        raise ValueError(f"Command working directory is not a directory: {cwd}")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=working_directory,
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
                raise subprocess.TimeoutExpired(cmd, timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(process)
            returncode = None
        except KeyboardInterrupt:
            kill_process_tree(process)
            raise
        stdout = stdout_future.result()
        stderr = stderr_future.result()

    return {
        "ok": returncode == 0,
        "timed_out": returncode is None,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
    }


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadFileArguments(ToolArguments):
    path: str = Field(description="Path to the file to read.")
    start_line: int | None = Field(description="Optional 1-based first line to read.")
    end_line: int | None = Field(description="Optional inclusive last line to read.")


class WriteFileArguments(ToolArguments):
    path: str = Field(description="The path to the file you want to write to.")
    content: str = Field(description="The content you want to write.")
    append: bool = Field(description="Append to the file instead of replacing it.")


class EditFileArguments(ToolArguments):
    path: str = Field(description="The path to the file to edit.")
    old_content: str = Field(description="The text to be replaced.")
    new_content: str = Field(description="The text to replace old_content with.")
    replace_all: bool = Field(
        description="Replace every match instead of requiring exactly one."
    )


class RunCommandArguments(ToolArguments):
    cmd: str = Field(description="The shell command to run.")
    timeout: int | None = Field(description="Maximum execution time in seconds.")
    cwd: str | None = Field(description="Optional working directory for the command.")


# The handler's docstring is the tool description sent to the model.
TOOLS: dict[str, tuple[Callable[..., dict[str, Any]], type[ToolArguments]]] = {
    "read_file": (read_file, ReadFileArguments),
    "write_file": (write_file, WriteFileArguments),
    "edit_file": (edit_file, EditFileArguments),
    "run_command": (run_command, RunCommandArguments),
}


def build_tool_schemas() -> list[ToolParam]:
    return [
        {
            "type": "function",
            "name": name,
            "description": inspect.getdoc(handler),
            "parameters": arguments_model.model_json_schema(),
            "strict": True,
        }
        for name, (handler, arguments_model) in TOOLS.items()
    ]


def execute_tool_call(call: ResponseFunctionToolCall) -> dict[str, Any]:
    """Execute a single function call and normalize failures into a result payload."""
    tool = TOOLS.get(call.name)
    if tool is None:
        return {"ok": False, "message": f"Unknown tool: {call.name}"}

    handler, arguments_model = tool
    try:
        arguments = arguments_model.model_validate_json(call.arguments)
        return handler(**arguments.model_dump())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}


def trim_history(history: ResponseInputParam) -> tuple[bool, bool]:
    trimmed = False
    while True:
        user_turn_starts = [
            index
            for index, item in enumerate(history)
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        history_bytes = len(
            json.dumps(history, ensure_ascii=False, default=str).encode("utf-8")
        )
        if (
            len(user_turn_starts) <= MAX_HISTORY_TURNS
            and history_bytes <= MAX_HISTORY_BYTES
        ):
            return trimmed, False

        if len(user_turn_starts) > 1:
            del history[: user_turn_starts[1]]
            trimmed = True
            continue

        if user_turn_starts:
            history[:] = [
                {
                    "role": "user",
                    "content": "[Previous request omitted because its history exceeded the context budget.]",
                }
            ]
            return True, True

        return trimmed, False


def run_turn(
    client: OpenAI,
    history: ResponseInputParam,
    prompt: str,
    tool_schemas: list[ToolParam],
) -> None:
    """Run one user turn, handling function calls until the model responds with a final answer."""
    settings = get_config()
    previous_history = history.copy()
    history.append({"role": "user", "content": prompt})

    try:
        for _ in range(settings.max_turns):
            trimmed, current_turn_dropped = trim_history(history)
            if trimmed:
                print(
                    f"[Conversation history trimmed to fit {MAX_HISTORY_BYTES} bytes]"
                )
            if current_turn_dropped:
                print("[Turn stopped because its history exceeded the context budget]")
                return

            if settings.streaming:
                with client.responses.stream(
                    model=settings.model,
                    instructions=system_prompt,
                    input=history,
                    tools=tool_schemas,
                    store=False,
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            print(event.delta, end="", flush=True)
                    response = stream.get_final_response()
            else:
                response = client.responses.create(
                    model=settings.model,
                    instructions=system_prompt,
                    input=history,
                    tools=tool_schemas,
                    store=False,
                )

            if response.output_text:
                if settings.streaming:
                    print()
                else:
                    print(response.output_text)

            if response.status == "incomplete":
                reason = (
                    response.incomplete_details.reason
                    if response.incomplete_details
                    else "unknown"
                )
                print(f"[Response incomplete: {reason}]")
                return

            history.extend(cast(ResponseInputParam, response.output))

            tool_calls = [
                item
                for item in response.output
                if isinstance(item, ResponseFunctionToolCall)
            ]
            if not tool_calls:
                return

            for call in tool_calls:
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps({"tool_result": execute_tool_call(call)}),
                    }
                )
        print(f"\nTurn limit reached ({settings.max_turns}).")
    except KeyboardInterrupt:
        history[:] = previous_history
        print("\n[Interrupted; returning to prompt]")


def main() -> None:
    """Run the interactive REPL for the coding agent."""
    config = get_config()
    history: ResponseInputParam = []
    tool_schemas = build_tool_schemas()

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    print(
        f"AI coding agent started | model: {config.model} | "
        f"workspace: {config.workspace_root}\nType 'exit' or 'quit' to stop.\n"
    )

    while True:
        try:
            prompt = input("> ")
        except KeyboardInterrupt, EOFError:
            break

        if not prompt:
            continue

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        try:
            run_turn(client, history, prompt, tool_schemas)
        except Exception as exc:  # noqa: BLE001
            print(f"Agent error: {type(exc).__name__}: {exc}")
        print()

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
