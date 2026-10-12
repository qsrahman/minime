import json
import subprocess
import sys
from itertools import islice
from pathlib import Path
from typing import Any, TypeGuard, cast

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseInputParam
from openai.types.responses.tool_param import ToolParam
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_TOOL_OUTPUT_CHARS = 40_000
MAX_COMMAND_TIMEOUT = 600


class Config(BaseSettings):
    """Configuration for the AI coding agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
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
    workspace_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent,
    )


try:
    # BaseSettings resolves required fields from environment sources at runtime.
    config = Config()  # pyright: ignore[reportCallIssue]
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
    workspace_root = config.workspace_root.resolve()
    file_path = Path(path)
    resolved_path = (
        file_path if file_path.is_absolute() else workspace_root / file_path
    ).resolve()
    if not resolved_path.is_relative_to(workspace_root):
        raise ValueError(f"Path is outside the workspace: {path}")
    return resolved_path


def truncate_output(text: str) -> str:
    """Cap tool output so large results cannot exhaust the model context."""
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_TOOL_OUTPUT_CHARS
    return f"{text[:MAX_TOOL_OUTPUT_CHARS]}\n... [truncated {omitted} characters]"


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

    return {"ok": True, "content": truncate_output(content)}


def write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """Write content to a file, optionally appending instead of replacing it."""
    file_path = resolve_workspace_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding="utf-8", newline="") as file:
        file.write(content)

    operation = "appended" if append else "written"
    return {
        "ok": True,
        "message": f"{operation} {len(content.encode('utf-8'))} bytes to {file_path}",
    }


def edit_file(
    path: str, old_content: str, new_content: str, replace_all: bool = False
) -> dict[str, Any]:
    """Replace a unique text fragment, or all matches when explicitly requested."""
    if not old_content:
        raise ValueError("old_content must not be empty")

    file_path = resolve_workspace_path(path)
    with file_path.open(encoding="utf-8", newline="") as file:
        existing_content = file.read()

    match_count = existing_content.count(old_content)
    if match_count == 0:
        raise ValueError(f"old_content not found in {path}")
    if match_count > 1 and not replace_all:
        raise ValueError(
            f"old_content must match exactly once in {path}; found {match_count} matches"
        )

    updated_content = existing_content.replace(old_content, new_content)
    with file_path.open("w", encoding="utf-8", newline="") as file:
        file.write(updated_content)

    return {
        "ok": True,
        "message": f"replaced {match_count} match(es) in {path}.",
    }


def run_command(
    cmd: str, timeout: int | None = None, cwd: str | None = None
) -> dict[str, Any]:
    """Run a shell command with a timeout and an optional working directory."""
    timeout = config.command_timeout if timeout is None else timeout
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    timeout = min(timeout, MAX_COMMAND_TIMEOUT)

    working_directory = resolve_workspace_path(cwd or str(config.workspace_root))
    if not working_directory.is_dir():
        raise ValueError(
            f"Command working directory is not a directory: {working_directory}"
        )

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            cwd=working_directory,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return {
            "ok": False,
            "timed_out": True,
            "stdout": truncate_output(
                stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            ),
            "stderr": truncate_output(
                stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            ),
            "returncode": None,
        }

    return {
        "ok": result.returncode == 0,
        "timed_out": False,
        "stdout": truncate_output(result.stdout),
        "stderr": truncate_output(result.stderr),
        "returncode": result.returncode,
    }


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
}

tool_schemas: list[ToolParam] = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a file, optionally limiting the result to an inclusive line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
                "start_line": {
                    "type": ["integer", "null"],
                    "description": "Optional 1-based first line to read.",
                },
                "end_line": {
                    "type": ["integer", "null"],
                    "description": "Optional inclusive last line to read.",
                },
            },
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file, replacing it by default or appending when append is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file you want to write to.",
                },
                "content": {
                    "type": "string",
                    "description": "The content you want to write.",
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to the file instead of replacing its contents.",
                },
            },
            "required": ["path", "content", "append"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": "Replace a unique text fragment, or replace all matches when replace_all is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit.",
                },
                "old_content": {
                    "type": "string",
                    "description": "The text to be replaced.",
                },
                "new_content": {
                    "type": "string",
                    "description": "The text to replace old_content with.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every match instead of requiring exactly one.",
                },
            },
            "required": ["path", "old_content", "new_content", "replace_all"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a shell command, including pipes and redirects, with an optional timeout and working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The shell command to run."},
                "timeout": {
                    "type": ["integer", "null"],
                    "description": "Maximum execution time in seconds.",
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": "Optional working directory for the command.",
                },
            },
            "required": ["cmd", "timeout", "cwd"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def execute_tool_call(call: ResponseFunctionToolCall) -> dict[str, Any]:
    """Execute a single function call and normalize failures into a result payload."""
    tool_name = call.name
    tool_handler = TOOLS.get(tool_name)
    if tool_handler is None:
        return {
            "ok": False,
            "message": f"Unknown tool: {tool_name}",
        }

    raw_arguments = call.arguments
    try:
        # Parse the raw arguments from JSON if they are provided as a string.
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
        return {
            "ok": False,
            "message": f"JSONDecodeError: {exc}",
        }

    # Ensure that the arguments are a dictionary before proceeding.
    if not isinstance(arguments, dict):
        return {
            "ok": False,
            "message": f"Tool arguments for {tool_name} must be an object, got {type(arguments).__name__}",
        }

    try:
        # Execute the tool handler with the parsed arguments.
        return tool_handler(**arguments)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "message": f"{type(exc).__name__}: {exc}",
        }


def is_function_tool_call(item: object) -> TypeGuard[ResponseFunctionToolCall]:
    return isinstance(item, ResponseFunctionToolCall)


def run_turn(client: OpenAI, history: ResponseInputParam, prompt: str) -> None:
    """Run one user turn, handling function calls until the model responds with a final answer."""
    history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    for turn in range(1, config.max_turns + 1):
        with client.responses.stream(
            model=config.model,
            instructions=system_prompt,
            input=history,
            tools=tool_schemas,
            store=False,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
            response = stream.get_final_response()

        if response.output_text:
            print()

        if response.status == "incomplete":
            reason = (
                response.incomplete_details.reason
                if response.incomplete_details
                else "unknown"
            )
            print(f"[Response incomplete: {reason}]")
            return

        history.extend(cast(ResponseInputParam, response.output))

        tool_calls = [item for item in response.output if is_function_tool_call(item)]

        if not tool_calls:
            return

        at_turn_limit = turn == config.max_turns

        for call in tool_calls:
            if at_turn_limit:
                tool_result = {
                    "ok": False,
                    "message": (
                        "Tool call skipped: maximum turns per run "
                        f"({config.max_turns}) reached."
                    ),
                }
            else:
                tool_result = execute_tool_call(call)

            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps({"tool_result": tool_result}),
                }
            )

        if at_turn_limit:
            print(
                f"\nTurn limit reached ({config.max_turns}); tool calls were skipped."
            )
            return


def main() -> None:
    """Run the interactive REPL for the coding agent."""
    history: ResponseInputParam = []

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    print(
        f"AI coding agent started | model: {config.model} | "
        f"workspace: {config.workspace_root}\nType 'exit' or 'quit' to stop."
    )

    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break

        if not prompt.strip():
            continue

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        try:
            print()
            run_turn(client, history, prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"Agent error: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
    print("\nGoodbye!")
