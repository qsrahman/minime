import json
import os
import pty
import re
import select
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from openai.types.responses import ResponseFunctionToolCall

from minime import config, tools
from minime.agent import Agent
from minime.context import Context
from minime.main import build_toolbox
from minime.repl import Repl

ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "MAX_STEPS_PER_TURN",
    "COMMAND_TIMEOUT_SECONDS",
    "REASONING_EFFORT",
    "SUMMARIZE",
    "MAX_CONTEXT_TOKENS",
    "STREAMING",
    "BRAVE_API_KEY",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run live tests last, and pseudo-terminal tests first within each group.

    Real API clients start system threads that outlive their tests, which
    makes the fork a pseudo-terminal test starts with unsafe.
    """

    def order(item: pytest.Item) -> tuple[bool, bool]:
        return (
            item.get_closest_marker("live") is not None,
            item.get_closest_marker("pty") is None,
        )

    items.sort(key=order)


@pytest.fixture(autouse=True)
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> config.Config:
    """Give every test its own settings, isolated from .env and the shell, and
    run it in its own empty workspace directory.

    Code under test reads these settings when it is built (a Toolbox, an
    Agent), so change them before building, not after.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    # main() reads command-line flags; pytest's own arguments are not for it.
    monkeypatch.setattr("sys.argv", ["minime"])
    test_settings = config.Config(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        OPENAI_API_KEY="test-key",
        STREAMING=False,
    )
    return test_settings


@pytest.fixture
def workspace(settings: config.Config, tmp_path: Path) -> Path:
    """The test's working directory, set up by the settings fixture."""
    return tmp_path / "workspace"


def call(tool: tools.Tool[Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a tool directly, the way Toolbox would after validating arguments.

    Positional arguments follow the argument model's field order; fields left
    out are None, or False for flags.
    """
    fields = tool.arguments_model.model_fields
    values = {
        name: False if info.annotation is bool else None
        for name, info in fields.items()
    }
    values |= dict(zip(fields, args, strict=False)) | kwargs
    return tool.run(tool.arguments_model.model_validate(values))


class Tools:
    """The standard tools for the test's settings, callable like functions:
    tool.read_file("f.txt", 3, 5). Built on first use, so a test can change
    its settings first."""

    def __init__(self, settings: config.Config) -> None:
        self.settings = settings
        self._toolbox: tools.Toolbox | None = None

    @property
    def toolbox(self) -> tools.Toolbox:
        if self._toolbox is None:
            self._toolbox = build_toolbox(self.settings)
        return self._toolbox

    def __getattr__(self, name: str) -> Callable[..., dict[str, Any]]:
        tool = self.toolbox.tools[name]
        return lambda *args, **kwargs: call(tool, *args, **kwargs)


@pytest.fixture
def tool(settings: config.Config) -> Tools:
    return Tools(settings)


def tool_call(
    name: str, call_id: str = "call_1", **arguments: Any
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def raw_tool_call(name: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", call_id="call_1", name=name, arguments=arguments
    )


class Site:
    """A local HTTP server whose routes are set by the test."""

    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.base = f"http://127.0.0.1:{server.server_address[1]}"
        self.routes: dict[
            str, tuple[int, dict[str, str], bytes | Callable[[Any], None]]
        ] = {}
        self.hits: list[str] = []

    def add(
        self,
        path: str,
        body: str | bytes | Callable[[Any], None] = b"",
        content_type: str | None = "text/html; charset=utf-8",
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> str:
        if isinstance(body, str):
            body = body.encode("utf-8")
        all_headers = dict(headers or {})
        if content_type is not None:
            all_headers["Content-Type"] = content_type
        self.routes[path] = (status, all_headers, body)
        return self.url(path)

    def url(self, path: str) -> str:
        return self.base + path


@pytest.fixture
def site() -> Iterator[Site]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            current.hits.append(self.path)
            status, headers, body = current.routes.get(
                self.path, (404, {"Content-Type": "text/plain"}, b"not found")
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            if callable(body):
                self.end_headers()
                try:
                    body(self.wfile)
                except OSError:
                    pass  # the client gave up
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    current = Site(server)
    # A short poll interval keeps shutdown() from adding half a second per test.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield current
    finally:
        server.shutdown()
        server.server_close()


def message_item(text: str, item_id: str = "msg_1") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call_item(name: str, call_id: str = "call_1", **arguments: Any) -> dict[str, Any]:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def reasoning_item(item_id: str = "rs_1") -> dict[str, Any]:
    """What the API returns for a reasoning step when the request has store=False."""
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "gAAAA-opaque-reasoning-state",
    }


def response_body(
    *output: dict[str, Any],
    status: str = "completed",
    reason: str | None = None,
    input_tokens: int | None = None,
    cached_tokens: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    """A Responses API reply. Usage is left out unless input_tokens is given."""
    usage = input_tokens and {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + 7,
    }
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": status,
        "error": {"code": "server_error", "message": error} if error else None,
        "incomplete_details": {"reason": reason} if reason else None,
        "instructions": None,
        "model": "test-model",
        "output": list(output),
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "usage": usage or None,
    }


def stream_events(reply: dict[str, Any]) -> list[dict[str, Any]]:
    """The event sequence the API sends when streaming this reply."""
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": reply | {"status": "in_progress", "output": [], "usage": None},
        }
    ]
    for index, item in enumerate(reply["output"]):
        at = {"output_index": index, "item_id": item["id"]}
        if item["type"] == "message":
            text = item["content"][0]["text"]
            part = {"type": "output_text", "text": "", "annotations": []}
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item | {"content": [], "status": "in_progress"},
                },
                {
                    "type": "response.content_part.added",
                    **at,
                    "content_index": 0,
                    "part": part,
                },
                *(
                    {
                        "type": "response.output_text.delta",
                        **at,
                        "content_index": 0,
                        "delta": chunk,
                        "logprobs": [],
                    }
                    for chunk in (text[: len(text) // 2], text[len(text) // 2 :])
                ),
                {
                    "type": "response.output_text.done",
                    **at,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
                {
                    "type": "response.content_part.done",
                    **at,
                    "content_index": 0,
                    "part": item["content"][0],
                },
            ]
        elif item["type"] == "reasoning":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item | {"encrypted_content": None},
                }
            )
        else:
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item | {"arguments": "", "status": "in_progress"},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    **at,
                    "delta": item["arguments"],
                },
                {
                    "type": "response.function_call_arguments.done",
                    **at,
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            ]
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    # A streamed response finishes as "completed" or "incomplete".
    events.append({"type": f"response.{reply['status']}", "response": reply})
    return [event | {"sequence_number": n} for n, event in enumerate(events)]


class Api:
    """A fake Responses endpoint behind the real OpenAI client.

    `replies` is played back in order: a response_body() dict, an
    httpx2.Response for an HTTP error, or an exception to raise mid-request.
    `requests` records the JSON body of every request the client sent.
    """

    def __init__(self, settings: config.Config) -> None:
        self.settings = settings
        self.replies: list[Any] = []
        self.requests: list[dict[str, Any]] = []
        self.client = openai.OpenAI(
            api_key="test-key",
            base_url="http://api.test/v1",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )

    def agent(self, conversation: Context | None = None) -> Agent:
        """An Agent for the test's settings that talks to this endpoint.

        Tests always build agents this way: an Agent left to make its own
        client sets up TLS, which on macOS starts a system thread that outlives
        the test and makes the pseudo-terminal tests' fork unsafe.
        """
        agent = Agent(self.settings, self.client, build_toolbox(self.settings))
        if conversation is not None:
            agent.context = conversation
        return agent

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, httpx2.Response):
            return reply
        if body.get("stream"):
            lines = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in stream_events(reply)
            )
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, text=lines
            )
        return httpx2.Response(200, json=reply)


@pytest.fixture
def api(settings: config.Config) -> Api:
    return Api(settings)


def kinds(items: list[Any]) -> list[str]:
    """One short label per item, for history or for the input of a recorded request."""
    labels = []
    for item in items:
        data = item if isinstance(item, dict) else item.model_dump()
        kind = data.get("type")
        if kind == "function_call":
            labels.append(f"call:{data['call_id']}")
        elif kind == "function_call_output":
            result = json.loads(data["output"])
            outcome = "ok" if result["ok"] else result.get("message", "failed")
            labels.append(f"output:{data['call_id']}:{outcome}")
        elif data.get("role") == "user":
            labels.append(f"user:{data['content']}")
        else:
            labels.append(kind)
    return labels


CTRL_C = b"\x03"
CTRL_D = b"\x04"
PROMPT_AGAIN = r"\[Interrupted; returning to prompt\][\s\S]*› $"


def process_is_gone(pid: int, wait_seconds: float = 0) -> bool:
    """Whether the process has exited, waiting up to wait_seconds for it."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


class Terminal:
    """A program running on a pseudo-terminal, typed at like a user would."""

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # the child: become the agent, touching as little as possible
            os.chdir(cwd)
            # HOME is the workspace, so the input history file lands there.
            os.execve(argv[0], argv, {**env, "TERM": "dumb", "HOME": str(cwd)})
        self.screen = ""
        self.exit_code: int | None = None

    def read(self, seconds: float) -> None:
        if select.select([self.fd], [], [], seconds)[0]:
            try:
                self.screen += os.read(self.fd, 4096).decode("utf-8", "replace")
            except OSError:  # the child closed its side
                time.sleep(seconds)

    def expect(self, pattern: str, timeout: float = 20) -> None:
        deadline = time.monotonic() + timeout
        while not re.search(pattern, self.screen):
            assert time.monotonic() < deadline, (
                f"terminal never showed {pattern!r}; it shows:\n{self.screen}"
            )
            self.read(0.1)

    def wait_for(self, condition: Any, what: str, timeout: float = 20) -> None:
        """Wait for something outside the terminal, still draining its output."""
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, (
                f"{what}; terminal shows:\n{self.screen}"
            )
            self.read(0.05)

    def type(self, keys: bytes | str) -> None:
        self.screen = ""
        os.write(self.fd, keys if isinstance(keys, bytes) else f"{keys}\r".encode())

    def wait_for_exit(self, timeout: float = 20) -> int:
        deadline = time.monotonic() + timeout
        while self.exit_code is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.exit_code = os.waitstatus_to_exitcode(status)
            else:
                assert time.monotonic() < deadline, "the agent did not exit"
                self.read(0.05)
        return self.exit_code

    def close(self) -> None:
        if self.exit_code is None:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
        os.close(self.fd)


# Scripting a turn, shared by the loop and REPL tests.


def write(call_id: str, path: str = "out.txt") -> dict[str, Any]:
    return call_item("write_file", call_id, path=path, content=call_id, append=False)


def answer(text: str = "done", **kwargs: Any) -> dict[str, Any]:
    return response_body(message_item(text), **kwargs)


def run(
    api: Api,
    history: list[Any] | None = None,
    prompt: str = "new",
    conversation: Context | None = None,
) -> Context:
    """One turn as the terminal runs it: text and notices are printed, problems
    are reported rather than raised, and a blank line follows."""
    agent = api.agent(conversation or Context(history=history or []))
    Repl(agent).ask(prompt)
    return agent.context
