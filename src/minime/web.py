import re
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx2
from pydantic import Field

from minime.tools import MAX_TOOL_OUTPUT_BYTES, Tool, ToolArguments

MAX_FETCH_BYTES = 2 * 1024 * 1024
WEB_TIMEOUT_SECONDS = 20
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


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
        text = body.decode(response.encoding or "utf-8", errors="replace")
        title = ""
        if "html" in content_type:
            title, text = html_to_text(text, str(response.url))
        return str(response.url), title, text, truncated


class WebFetchArguments(ToolArguments):
    url: str = Field(description="The http or https URL to fetch.")
    offset: int | None = Field(
        description="Optional character position to continue reading from."
    )


class WebSearchArguments(ToolArguments):
    query: str = Field(description="The search query.")
    count: int | None = Field(description="Number of results, 1 to 20. Defaults to 5.")


class WebFetchTool(Tool[WebFetchArguments]):
    """Fetch a page in parts; pass next_offset as offset. truncated marks downloads over 2 MB."""

    name = "web_fetch"
    arguments_model = WebFetchArguments

    def __init__(self) -> None:
        # The last page downloaded, so continuation calls page through the
        # same copy: (requested URL, fetch_page result).
        self.last_page: tuple[str, tuple[str, str, str, bool]] | None = None

    def run(self, args: WebFetchArguments) -> dict[str, Any]:
        offset = args.offset or 0
        if offset < 0:
            raise ValueError("offset must not be negative")
        # Only continuation calls reuse the last download; a first part is always
        # fetched fresh, so a page that changed is never served stale.
        if not offset or self.last_page is None or self.last_page[0] != args.url:
            self.last_page = (args.url, fetch_page(args.url))
        final_url, title, text, truncated = self.last_page[1]
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


class WebSearchTool(Tool[WebSearchArguments]):
    """Search the web and return result titles, URLs, and descriptions. Operators such as site: and quoted phrases go in query."""

    name = "web_search"
    arguments_model = WebSearchArguments

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def run(self, args: WebSearchArguments) -> dict[str, Any]:
        count = 5 if args.count is None else args.count
        if not 1 <= count <= 20:
            raise ValueError("count must be between 1 and 20")

        response = httpx2.get(
            BRAVE_SEARCH_URL,
            params={"q": args.query, "count": count},
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
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
