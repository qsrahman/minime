"""The agent in a real terminal: Ctrl-C and Ctrl-D typed at a pseudo-terminal.

Every other test raises KeyboardInterrupt by hand. Here the real `main.py` loop
runs as a child process on a pty and gets the keystrokes a user would type, so
signal delivery, readline and process-group handling are all the real thing.
The model is a local HTTP server playing back scripted replies.
"""

import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from conftest import (
    CTRL_C,
    CTRL_D,
    PROMPT_AGAIN,
    Terminal,
    call_item,
    kinds,
    message_item,
    process_is_gone,
    response_body,
    stream_events,
)

pytestmark = [
    pytest.mark.pty,
    pytest.mark.skipif(os.name != "posix", reason="needs a pseudo-terminal"),
]

# Run the real loop, but take every setting from the environment the test
# builds instead of the developer's .env file.
LAUNCH = (
    "from minime import config, main; "
    "config.Config.model_config['env_file'] = None; main.main()"
)


class Endpoint:
    """A local Responses API that streams scripted replies.

    A reply wrapped in `held()` stops after its first piece of text and stays
    open, the way a long answer would, until the test ends.
    """

    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.release = threading.Event()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                endpoint.requests.append(json.loads(self.rfile.read(length)))
                reply = endpoint.replies.pop(0)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for event in stream_events(reply["body"]):
                        line = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                        if reply["held"] and event["type"].endswith("text.delta"):
                            endpoint.release.wait(30)
                            return
                except OSError:
                    pass  # the agent hung up, as it does on Ctrl-C

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self) -> None:
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        ).start()

    def script(self, *bodies: dict[str, Any]) -> None:
        self.replies += [
            body if "held" in body else {"body": body, "held": False} for body in bodies
        ]

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def held(body: dict[str, Any]) -> dict[str, Any]:
    return {"body": body, "held": True}


@pytest.fixture
def endpoint() -> Iterator[Endpoint]:
    server = Endpoint()
    yield server
    server.close()


@pytest.fixture
def terminal(endpoint: Endpoint, workspace: Path) -> Iterator[Terminal]:
    # Fork the agent before the server thread exists: forking a process that
    # has other threads running is unsafe. The port is already bound, so
    # requests made before start() simply wait.
    env = {
        **os.environ,
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": endpoint.url,
    }
    session = Terminal([sys.executable, "-c", LAUNCH], env, workspace)
    endpoint.start()
    try:
        session.expect(r"> $")
        yield session
    finally:
        session.close()


def test_ctrl_c_during_a_streamed_answer_returns_to_the_prompt(
    endpoint: Endpoint, terminal: Terminal
) -> None:
    endpoint.script(
        held(response_body(message_item("one two three four"))),
        response_body(message_item("still here")),
    )
    terminal.type("count for me")
    terminal.expect(r"one two")  # the answer has started to appear
    terminal.type(CTRL_C)
    terminal.expect(PROMPT_AGAIN)

    # The agent is still usable, and the abandoned request left no trace.
    terminal.type("are you there?")
    terminal.expect(r"still here[\s\S]*> $")
    assert kinds(endpoint.requests[1]["input"]) == ["user:are you there?"]


def test_ctrl_c_during_a_command_kills_it_and_returns_to_the_prompt(
    endpoint: Endpoint, terminal: Terminal, workspace: Path
) -> None:
    command = "echo $$ > pid; exec sleep 30"
    endpoint.script(
        response_body(call_item("run_command", cmd=command, timeout=None, cwd=None)),
        response_body(message_item("understood")),
    )
    terminal.type("run the slow thing")
    pid_file = workspace / "pid"
    terminal.wait_for(
        lambda: pid_file.exists() and pid_file.read_text().strip(),
        "the command never started",
    )
    command_pid = int(pid_file.read_text())

    started = time.monotonic()
    terminal.type(CTRL_C)
    terminal.expect(PROMPT_AGAIN)
    assert time.monotonic() - started < 5  # not the 30 seconds the command wanted
    terminal.wait_for(
        lambda: process_is_gone(command_pid), "the command survived Ctrl-C", timeout=5
    )

    # The next request tells the model its command was interrupted.
    terminal.type("what happened?")
    terminal.expect(r"understood[\s\S]*> $")
    assert kinds(endpoint.requests[1]["input"]) == [
        "user:run the slow thing",
        "call:call_1",
        "output:call_1:Interrupted by the user.",
        "user:what happened?",
    ]


@pytest.mark.parametrize("key", [CTRL_C, CTRL_D], ids=["ctrl-c", "ctrl-d"])
def test_quitting_from_the_prompt(terminal: Terminal, key: bytes) -> None:
    terminal.type(key)
    terminal.expect(r"Goodbye!")
    assert terminal.wait_for_exit() == 0


def test_exit_after_an_interrupted_request(
    endpoint: Endpoint, terminal: Terminal
) -> None:
    endpoint.script(held(response_body(message_item("one two three four"))))
    terminal.type("count for me")
    terminal.expect(r"one two")
    terminal.type(CTRL_C)
    terminal.expect(PROMPT_AGAIN)
    terminal.type("/exit")
    terminal.expect(r"Goodbye!")
    assert terminal.wait_for_exit() == 0
