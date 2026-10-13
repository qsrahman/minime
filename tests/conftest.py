import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.responses import ResponseFunctionToolCall

import config
import main
import tools

ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "MAX_TURNS_PER_RUN",
    "COMMAND_TIMEOUT_SECONDS",
    "MAX_HISTORY_TURNS",
    "MAX_HISTORY_BYTES",
    "STREAMING",
    "BRAVE_API_KEY",
    "WORKSPACE_ROOT",
)


@pytest.fixture(autouse=True)
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> config.Config:
    """Give every test its own settings and workspace, isolated from .env and the shell."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_settings = config.Config(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        OPENAI_API_KEY="test-key",
        WORKSPACE_ROOT=workspace,
        STREAMING=False,
    )
    monkeypatch.setattr(tools, "get_config", lambda: test_settings)
    monkeypatch.setattr(main, "get_config", lambda: test_settings)
    tools.fetch_page.cache_clear()
    return test_settings


@pytest.fixture
def workspace(settings: config.Config) -> Path:
    return settings.workspace_root


def tool_call(name: str, call_id: str = "call_1", **arguments: Any) -> ResponseFunctionToolCall:
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
        self.routes: dict[str, tuple[int, dict[str, str], bytes | Callable[[Any], None]]] = {}
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


def response(
    text: str = "",
    calls: tuple[ResponseFunctionToolCall, ...] | list[ResponseFunctionToolCall] = (),
    status: str = "completed",
    reason: str | None = None,
) -> SimpleNamespace:
    """A stand-in for an OpenAI Response with just the fields run_turn reads."""
    output: list[Any] = list(calls)
    if text:
        output.append(SimpleNamespace(type="message", text=text))
    return SimpleNamespace(
        output_text=text,
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=output,
    )


class FakeStream:
    def __init__(self, final: SimpleNamespace) -> None:
        self.final = final

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(type="response.created")
        text = self.final.output_text
        for chunk in (text[: len(text) // 2], text[len(text) // 2 :]):
            if chunk:
                yield SimpleNamespace(type="response.output_text.delta", delta=chunk)

    def get_final_response(self) -> SimpleNamespace:
        return self.final


class FakeClient:
    """Plays back a script of responses (or exceptions) and records each request."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.responses = self

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def create(self, **kwargs: Any) -> Any:
        return self._next(kwargs)

    def stream(self, **kwargs: Any) -> FakeStream:
        return FakeStream(self._next(kwargs))


def kinds(history: list[Any]) -> list[str]:
    """Summarise history as one short label per item."""
    labels = []
    for item in history:
        if isinstance(item, dict) and "role" in item:
            labels.append(f"user:{item['content']}")
        elif isinstance(item, dict):
            result = json.loads(item["output"])["tool_result"]
            labels.append(f"output:{item['call_id']}:{'ok' if result['ok'] else result.get('message', 'failed')}")
        elif isinstance(item, ResponseFunctionToolCall):
            labels.append(f"call:{item.call_id}")
        else:
            labels.append(item.type)
    return labels
