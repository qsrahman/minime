import inspect
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from typing import IO, Any
from urllib.parse import urljoin

import httpx2
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.tool_param import ToolParam
from pydantic import BaseModel, ConfigDict, Field

from config import get_config

MAX_TOOL_OUTPUT_BYTES = 16 * 1024
MAX_FETCH_BYTES = 2 * 1024 * 1024
WEB_TIMEOUT_SECONDS = 20
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
# Bytes set aside for the "[... N bytes omitted ...]" line in clipped output.
OMISSION_MARKER_ROOM = 48


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
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """Read a file, optionally limited to an inclusive line range. Long files are returned in parts: to read the next part, pass next_start_line as start_line and, when it is not null, next_offset as offset."""
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be greater than zero")
    if offset is not None and offset < 0:
        raise ValueError("offset must not be negative")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be greater than zero")
    if start_line is not None and end_line is not None and end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line")

    file_path = resolve_workspace_path(path)
    first_line = 1 if start_line is None else start_line

    lines: list[str] = []
    size = 0
    next_start_line = None
    next_offset = None
    with file_path.open(encoding="utf-8") as file:
        numbered_lines = enumerate(islice(file, first_line - 1, end_line), first_line)
        for number, line in numbered_lines:
            if number == first_line and offset:
                line = line[offset:]
            size += len(line.encode("utf-8"))
            if size > MAX_TOOL_OUTPUT_BYTES:
                next_start_line = number
                if not lines:
                    # A single line longer than the limit is returned in pieces.
                    # errors="ignore" drops a character the byte limit cut in half.
                    piece = line.encode("utf-8")[:MAX_TOOL_OUTPUT_BYTES].decode(
                        "utf-8", errors="ignore"
                    )
                    lines.append(piece)
                    next_offset = (offset or 0) + len(piece)
                break
            lines.append(line)

    return {
        "ok": True,
        "content": "".join(lines),
        "next_start_line": next_start_line,
        "next_offset": next_offset,
    }


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
        "message": f"{operation} {len(content.encode('utf-8'))} bytes to {path}",
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
        # The usual near miss: right text, wrong indentation or line endings.
        hint = ""
        squeezed = "".join(old_content.split())
        if squeezed and squeezed in "".join(existing_content.split()):
            hint = (
                "; the same text is there with different whitespace (indentation, "
                "spacing or line endings), so copy it exactly from read_file"
            )
        raise ValueError(f"old_content not found in {path}{hint}")
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


class TextExtractor(HTMLParser):
    """Collect the title and readable text of an HTML document."""

    SKIPPED_TAGS = frozenset(
        "script style noscript template svg nav header footer aside".split()  # noqa: SIM905
    )
    # Sites that mark page chrome with ARIA roles on generic containers.
    SKIPPED_ROLES = frozenset(
        "navigation search banner contentinfo complementary".split()  # noqa: SIM905
    )
    ROLE_TAGS = frozenset({"div", "section", "form", "ul"})
    BLOCK_TAGS = frozenset(
        "address article blockquote br dd div dl dt figcaption figure "  # noqa: SIM905
        "form h1 h2 h3 h4 h5 h6 hr li main ol p pre section table tr ul".split()
    )

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.parts: list[str] = []
        self.title = ""
        self.in_title = False
        self.skip_tag = ""
        self.skip_depth = 0
        self.pre_depth = 0
        self.link_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_tag:
            self.skip_depth += tag == self.skip_tag
            return
        attributes = dict(attrs)
        if tag in self.SKIPPED_TAGS or (
            tag in self.ROLE_TAGS and attributes.get("role") in self.SKIPPED_ROLES
        ):
            self.skip_tag, self.skip_depth = tag, 1
            return

        self.in_title = tag == "title" and not self.title
        if tag == "pre":
            self.pre_depth += 1
        elif tag == "a" and not self.pre_depth:
            href = attributes.get("href") or ""
            if href and not href.startswith(("#", "javascript:")):
                self.link_url = urljoin(self.base_url, href)
                self.parts.append("[")
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_tag:
            self.skip_depth -= tag == self.skip_tag
            if not self.skip_depth:
                self.skip_tag = ""
            return

        self.in_title = False
        if tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
        elif tag == "a" and self.link_url:
            if self.parts[-1] == "[":  # a link with no text, such as an image
                self.parts.pop()
            else:
                self.parts.append(f"]({self.link_url})")
            self.link_url = ""
        # <br/> and <hr/> report an end as well as a start; one newline is enough.
        if tag in self.BLOCK_TAGS and tag not in ("br", "hr"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_tag:
            return
        if self.in_title:
            self.title += data
        elif self.pre_depth:
            self.parts.append(data)
        else:
            text = re.sub(r"\s+", " ", data)
            if not self.parts or self.parts[-1].endswith("\n"):
                text = text.lstrip()
            self.parts.append(text)


def html_to_text(markup: str, base_url: str = "") -> tuple[str, str]:
    """Return the title and visible text of an HTML document or fragment.

    Links are kept as [text](url), resolved against base_url.
    """
    extractor = TextExtractor(base_url)
    extractor.feed(markup)
    extractor.close()
    text = re.sub(r"[ \t]+\n", "\n", "".join(extractor.parts))
    return extractor.title.strip(), re.sub(r"\n{3,}", "\n\n", text).strip()


@lru_cache(maxsize=1)
def fetch_page(url: str) -> tuple[str, str, str, bool]:
    """Download a page and return its final URL, title, and text."""
    with httpx2.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=WEB_TIMEOUT_SECONDS,
        headers={"User-Agent": "minime/0.1"},
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if not (
            content_type.startswith("text/") or content_type.endswith(("json", "xml"))
        ):
            raise ValueError(f"unsupported content type: {content_type or 'unknown'}")

        # The client timeout is per read, so a slow trickle needs its own limit.
        deadline = time.monotonic() + WEB_TIMEOUT_SECONDS
        body = bytearray()
        truncated = False
        for chunk in response.iter_bytes():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"download took longer than {WEB_TIMEOUT_SECONDS} seconds"
                )
            remaining = MAX_FETCH_BYTES - len(body)
            if len(chunk) > remaining:
                body.extend(chunk[:remaining])
                truncated = True
                break
            body.extend(chunk)
        text = body[:MAX_FETCH_BYTES].decode(
            response.encoding or "utf-8", errors="replace"
        )
        title = ""
        if "html" in content_type:
            title, text = html_to_text(text, str(response.url))
        return str(response.url), title, text, truncated


def web_fetch(url: str, offset: int | None = None) -> dict[str, Any]:
    """Fetch a page in parts; pass next_offset as offset. truncated marks downloads over 2 MB."""
    offset = offset or 0
    if offset < 0:
        raise ValueError("offset must not be negative")
    if not offset:
        # Only continuation calls may reuse the cached copy of a page.
        fetch_page.cache_clear()

    final_url, title, text, truncated = fetch_page(url)
    # At most that many characters can fit; multi-byte text is then cut on
    # bytes, with errors="ignore" dropping a character split at the limit.
    content = (
        text[offset : offset + MAX_TOOL_OUTPUT_BYTES]
        .encode("utf-8")[:MAX_TOOL_OUTPUT_BYTES]
        .decode("utf-8", errors="ignore")
    )
    end = offset + len(content)
    return {
        "ok": True,
        "url": final_url,
        "title": title,
        "content": content,
        "total_chars": len(text),
        "next_offset": end if end < len(text) else None,
        "truncated": truncated,
    }


def web_search(query: str, count: int | None = None) -> dict[str, Any]:
    """Search the web and return result titles, URLs, and descriptions. Operators such as site: and quoted phrases go in query."""
    api_key = get_config().brave_api_key
    if not api_key:
        raise ValueError("BRAVE_API_KEY is not set")
    count = 5 if count is None else count
    if not 1 <= count <= 20:
        raise ValueError("count must be between 1 and 20")

    response = httpx2.get(
        BRAVE_SEARCH_URL,
        params={"q": query, "count": count},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=WEB_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return {
        "ok": True,
        "results": [
            {
                "title": html_to_text(result.get("title", ""))[1],
                "url": result.get("url"),
                "description": html_to_text(result.get("description", ""))[1],
            }
            for result in response.json().get("web", {}).get("results", [])
        ],
    }


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadFileArguments(ToolArguments):
    path: str = Field(description="Path to the file to read.")
    start_line: int | None = Field(description="Optional 1-based first line to read.")
    end_line: int | None = Field(description="Optional inclusive last line to read.")
    offset: int | None = Field(
        description="Optional character position within start_line to continue from."
    )


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


class WebFetchArguments(ToolArguments):
    url: str = Field(description="The http or https URL to fetch.")
    offset: int | None = Field(
        description="Optional character position to continue reading from."
    )


class WebSearchArguments(ToolArguments):
    query: str = Field(description="The search query.")
    count: int | None = Field(description="Number of results, 1 to 20. Defaults to 5.")


# The handler's docstring is the tool description sent to the model.
TOOLS: dict[str, tuple[Callable[..., dict[str, Any]], type[ToolArguments]]] = {
    "read_file": (read_file, ReadFileArguments),
    "write_file": (write_file, WriteFileArguments),
    "edit_file": (edit_file, EditFileArguments),
    "run_command": (run_command, RunCommandArguments),
    "web_fetch": (web_fetch, WebFetchArguments),
    "web_search": (web_search, WebSearchArguments),
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
        # web_search cannot work without a key, so it is not offered.
        if name != "web_search" or get_config().brave_api_key
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
