import time
from typing import Any

import httpx2
import pytest

import config
import tools
import web
from agent import build_toolbox
from conftest import Site, Tools

LIMIT = tools.MAX_TOOL_OUTPUT_BYTES


def text_of(markup: str, base_url: str = "") -> str:
    return web.html_to_text(markup, base_url)[1]


class TestHtmlToText:
    def test_title_is_returned_separately(self) -> None:
        title, text = web.html_to_text(
            "<html><head><title> My &amp; Page </title></head><body><p>Body</p></body>"
        )
        assert (title, text) == ("My & Page", "Body")

    def test_no_title(self) -> None:
        assert web.html_to_text("<p>Body</p>") == ("", "Body")

    def test_only_the_first_title_counts(self) -> None:
        title, text = web.html_to_text(
            "<title>First</title><body><title>Second</title><p>Body</p>"
        )
        assert title == "First"

    @pytest.mark.parametrize(
        "tag",
        [
            "script",
            "style",
            "noscript",
            "template",
            "svg",
            "nav",
            "header",
            "footer",
            "aside",
        ],
    )
    def test_skipped_elements(self, tag: str) -> None:
        assert (
            text_of(f"<p>before</p><{tag}>HIDDEN <b>nested</b></{tag}><p>after</p>")
            == "before\n\nafter"
        )

    def test_nested_skipped_elements_of_the_same_tag(self) -> None:
        assert text_of("<nav>a<nav>b</nav>still hidden</nav><p>shown</p>") == "shown"

    def test_title_inside_svg_is_ignored(self) -> None:
        title, text = web.html_to_text("<svg><title>Icon</title></svg><p>Body</p>")
        assert (title, text) == ("", "Body")

    @pytest.mark.parametrize(
        "role", ["navigation", "search", "banner", "contentinfo", "complementary"]
    )
    def test_containers_with_chrome_roles_are_skipped(self, role: str) -> None:
        markup = f'<div role="{role}"><div>inner</div>HIDDEN</div><p>shown</p>'
        assert text_of(markup) == "shown"

    def test_other_roles_are_kept(self) -> None:
        assert text_of('<div role="main"><p>shown</p></div>') == "shown"

    def test_roles_on_void_elements_do_not_hide_the_rest(self) -> None:
        assert (
            text_of('<input role="search"><hr role="navigation"><p>shown</p>')
            == "shown"
        )

    def test_whitespace_is_collapsed(self) -> None:
        assert text_of("<p>one\n   two\t\tthree</p>") == "one two three"

    def test_inline_elements_keep_word_spacing(self) -> None:
        assert (
            text_of("<p>plain <b>bold</b> and <i>italic</i>.</p>")
            == "plain bold and italic."
        )

    def test_block_elements_start_new_lines(self) -> None:
        assert (
            text_of("<h1>Title</h1><p>One</p><ul><li>a</li><li>b</li></ul>")
            == "Title\n\nOne\n\na\n\nb"
        )

    def test_line_breaks(self) -> None:
        assert text_of("<p>one<br>two<br>three</p>") == "one\ntwo\nthree"

    @pytest.mark.parametrize("tag", ["<br>", "<br/>", "<br />"])
    def test_line_break_forms_are_equivalent(self, tag: str) -> None:
        assert text_of(f"<p>one{tag}two</p>") == "one\ntwo"

    def test_rule_separates_blocks(self) -> None:
        assert text_of("<p>one</p><hr/><p>two</p>") == "one\n\ntwo"

    def test_blank_lines_are_squeezed(self) -> None:
        assert (
            text_of("<div><div><div><p>a</p></div></div></div><div><p>b</p></div>")
            == "a\n\nb"
        )

    def test_entities_are_decoded(self) -> None:
        assert (
            text_of("<p>a &lt; b &amp;&amp; c &gt; d &copy; &#233;</p>")
            == "a < b && c > d © é"
        )

    def test_pre_is_kept_verbatim(self) -> None:
        assert (
            text_of("<pre>def f():\n    return  1\n</pre>") == "def f():\n    return  1"
        )

    def test_links_become_markdown(self) -> None:
        assert text_of(
            '<p>See <a href="https://example.com/a?b=1">the <b>docs</b></a>.</p>'
        ) == ("See [the docs](https://example.com/a?b=1).")

    def test_relative_links_are_resolved(self) -> None:
        base = "https://example.com/guide/page.html"
        assert text_of(
            '<a href="intro.html">x</a> <a href="/root">y</a> <a href="../up">z</a>',
            base,
        ) == (
            "[x](https://example.com/guide/intro.html) [y](https://example.com/root) [z](https://example.com/up)"
        )

    @pytest.mark.parametrize("href", ["#section", "javascript:void(0)", ""])
    def test_links_without_a_destination_are_plain_text(self, href: str) -> None:
        assert text_of(f'<p><a href="{href}">label</a></p>') == "label"

    def test_anchor_without_href(self) -> None:
        assert text_of("<p><a>label</a></p>") == "label"

    def test_link_without_text_is_dropped(self) -> None:
        assert text_of('<p>a <a href="/img"><img src="i.png"></a>b</p>') == "a b"

    def test_links_inside_pre_are_not_rewritten(self) -> None:
        assert text_of('<pre>call <a href="/api/f">f</a>()</pre>') == "call f()"

    def test_links_inside_skipped_elements_are_dropped(self) -> None:
        assert text_of('<nav><a href="/x">Home</a></nav><p>Body</p>') == "Body"

    def test_plain_text_fragment(self) -> None:
        assert web.html_to_text("just <strong>text</strong> &amp; more") == (
            "",
            "just text & more",
        )

    def test_empty_document(self) -> None:
        assert web.html_to_text("") == ("", "")

    def test_unclosed_tags(self) -> None:
        assert text_of("<p>one<p>two<li>three") == "one\ntwo\nthree"


PAGE = """<!doctype html><html><head><title>Fixture</title><script>var x = "SCRIPT";</script></head>
<body><nav><a href="/home">Home</a></nav>
<main><h1>Heading</h1><p>See <a href="guide/intro.html">the intro</a>.</p></main>
<footer>Footer</footer></body></html>"""


class TestWebFetch:
    def test_html_page(self, tool: Tools, site: Site) -> None:
        url = site.add("/page.html", PAGE)
        assert tool.web_fetch(url) == {
            "ok": True,
            "url": url,
            "title": "Fixture",
            "content": f"Heading\n\nSee [the intro]({site.url('/guide/intro.html')}).",
            "total_chars": len(
                f"Heading\n\nSee [the intro]({site.url('/guide/intro.html')})."
            ),
            "next_offset": None,
            "truncated": False,
        }

    def test_sends_a_user_agent(
        self, tool: Tools, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        real_stream = httpx2.stream

        def recording_stream(method: str, url: str, **kwargs: Any) -> Any:
            seen.update(kwargs, method=method)
            return real_stream(method, url, **kwargs)

        monkeypatch.setattr(httpx2, "stream", recording_stream)
        tool.web_fetch(site.add("/p", "<p>x</p>"))
        assert seen["method"] == "GET"
        assert seen["headers"] == {"User-Agent": "minime/0.1"}
        assert seen["follow_redirects"] is True
        assert seen["timeout"] == web.WEB_TIMEOUT_SECONDS

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("text/plain", "plain <b>not html</b>\n"),
            ("application/json", '{"a": 1}'),
            ("application/xml", "<root><a>1</a></root>"),
            ("application/ld+json", '{"@context": "x"}'),
            ("text/markdown; charset=utf-8", "# Title\n"),
        ],
    )
    def test_non_html_text_is_returned_as_is(
        self, tool: Tools, site: Site, content_type: str, body: str
    ) -> None:
        result = tool.web_fetch(site.add("/data", body, content_type))
        assert result["content"] == body
        assert result["title"] == ""

    def test_xhtml_is_treated_as_html(self, tool: Tools, site: Site) -> None:
        result = tool.web_fetch(
            site.add(
                "/x", "<html><body><p>Body</p></body></html>", "application/xhtml+xml"
            )
        )
        assert result["content"] == "Body"

    @pytest.mark.parametrize(
        "content_type", ["image/png", "application/pdf", "application/octet-stream"]
    )
    def test_binary_content_is_rejected(
        self, tool: Tools, site: Site, content_type: str
    ) -> None:
        with pytest.raises(
            ValueError, match=f"unsupported content type: {content_type}"
        ):
            tool.web_fetch(site.add("/file", b"\x89PNG\r\n", content_type))

    def test_missing_content_type_is_rejected(self, tool: Tools, site: Site) -> None:
        with pytest.raises(ValueError, match="unsupported content type: unknown"):
            tool.web_fetch(site.add("/file", b"data", content_type=None))

    def test_not_found(self, tool: Tools, site: Site) -> None:
        with pytest.raises(httpx2.HTTPStatusError, match="404"):
            tool.web_fetch(site.url("/missing"))

    def test_server_error(self, tool: Tools, site: Site) -> None:
        with pytest.raises(httpx2.HTTPStatusError, match="500"):
            tool.web_fetch(site.add("/broken", "oops", status=500))

    def test_connection_refused(self, tool: Tools, site: Site) -> None:
        with pytest.raises(httpx2.ConnectError):
            tool.web_fetch("http://127.0.0.1:1/")

    def test_unsupported_scheme(self, tool: Tools) -> None:
        with pytest.raises(httpx2.UnsupportedProtocol):
            tool.web_fetch("ftp://example.com/file")

    def test_redirects_are_followed_and_links_use_the_final_url(
        self, tool: Tools, site: Site
    ) -> None:
        final = site.add("/new/page.html", '<p><a href="other.html">other</a></p>')
        start = site.add("/old", status=302, headers={"Location": "/new/page.html"})
        result = tool.web_fetch(start)
        assert result["url"] == final
        assert result["content"] == f"[other]({site.url('/new/other.html')})"

    def test_declared_charset_is_used(self, tool: Tools, site: Site) -> None:
        url = site.add(
            "/latin", "café".encode("latin-1"), "text/plain; charset=iso-8859-1"
        )
        assert tool.web_fetch(url)["content"] == "café"

    def test_undecodable_bytes_are_replaced(self, tool: Tools, site: Site) -> None:
        url = site.add("/bad", b"ok \xff end", "text/plain; charset=utf-8")
        assert tool.web_fetch(url)["content"] == "ok � end"

    def test_negative_offset(self, tool: Tools, site: Site) -> None:
        with pytest.raises(ValueError, match="offset must not be negative"):
            tool.web_fetch(site.add("/p", "<p>x</p>"), -1)
        assert site.hits == []

    def test_long_page_is_returned_in_parts(self, tool: Tools, site: Site) -> None:
        body = "".join(f"line {i:05d}\n" for i in range(4000))  # 44 KB
        url = site.add("/long.txt", body, "text/plain")

        parts, offset = [], None
        while True:
            result = tool.web_fetch(url, offset)
            assert len(result["content"].encode()) <= LIMIT
            assert result["total_chars"] == len(body)
            parts.append(result["content"])
            if result["next_offset"] is None:
                break
            offset = result["next_offset"]

        assert len(parts) == 3
        assert "".join(parts) == body
        assert site.hits == ["/long.txt"]  # continuation calls reuse the download

    def test_multibyte_parts_stay_within_the_byte_limit(
        self, tool: Tools, site: Site
    ) -> None:
        body = "é" * 20000  # 40 KB as UTF-8
        url = site.add("/utf.txt", body, "text/plain; charset=utf-8")
        first = tool.web_fetch(url)
        assert first["content"] == "é" * (LIMIT // 2)
        assert first["next_offset"] == LIMIT // 2
        rest = tool.web_fetch(url, first["next_offset"])
        assert rest["content"].startswith("é")

    def test_offset_past_the_end(self, tool: Tools, site: Site) -> None:
        result = tool.web_fetch(site.add("/p", "short", "text/plain"), 100)
        assert (result["content"], result["next_offset"]) == ("", None)

    def test_first_part_is_always_downloaded_fresh(
        self, tool: Tools, site: Site
    ) -> None:
        url = site.add("/live", "version one", "text/plain")
        assert tool.web_fetch(url)["content"] == "version one"
        site.add("/live", "version two", "text/plain")
        assert tool.web_fetch(url)["content"] == "version two"
        assert tool.web_fetch(url, 0)["content"] == "version two"
        assert site.hits == ["/live"] * 3

    def test_fetching_another_page_replaces_the_cached_one(
        self, tool: Tools, site: Site
    ) -> None:
        first = site.add("/a", "a" * 100, "text/plain")
        second = site.add("/b", "b" * 100, "text/plain")
        tool.web_fetch(first)
        tool.web_fetch(second)
        assert tool.web_fetch(first, 50)["content"] == "a" * 50
        assert site.hits == ["/a", "/b", "/a"]

    def test_download_is_capped_and_reported(
        self, tool: Tools, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "MAX_FETCH_BYTES", 1000)
        result = tool.web_fetch(site.add("/big", "x" * 5000, "text/plain"))
        assert result["total_chars"] == 1000
        assert result["truncated"] is True

    def test_download_at_the_cap_is_not_reported_as_truncated(
        self, tool: Tools, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "MAX_FETCH_BYTES", 1000)
        result = tool.web_fetch(site.add("/exact", "x" * 1000, "text/plain"))
        assert (result["total_chars"], result["truncated"]) == (1000, False)

    def test_slow_download_hits_the_deadline(
        self, tool: Tools, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def drip(wfile: Any) -> None:
            for _ in range(100):
                wfile.write(b"drip\n")
                wfile.flush()
                time.sleep(0.05)

        monkeypatch.setattr(web, "WEB_TIMEOUT_SECONDS", 0.3)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="download took longer"):
            tool.web_fetch(site.add("/drip", drip, "text/plain"))
        assert time.monotonic() - started < 3

    def test_failed_fetch_is_not_cached(self, tool: Tools, site: Site) -> None:
        url = site.url("/later")
        with pytest.raises(httpx2.HTTPStatusError):
            tool.web_fetch(url)
        site.add("/later", "now here", "text/plain")
        assert tool.web_fetch(url)["content"] == "now here"


class FakeSearchResponse:
    def __init__(self, payload: Any, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error:
            raise self.error

    def json(self) -> Any:
        return self.payload


@pytest.fixture
def brave(settings: config.Config, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Enable web_search and replace the Brave request with a recorder."""
    settings.brave_api_key = "brave-key"
    state: dict[str, Any] = {
        "response": FakeSearchResponse({"web": {"results": []}}),
        "calls": [],
    }

    def fake_get(url: str, **kwargs: Any) -> FakeSearchResponse:
        state["calls"].append({"url": url, **kwargs})
        return state["response"]

    monkeypatch.setattr(httpx2, "get", fake_get)
    return state


class TestWebSearch:
    def test_is_only_offered_with_an_api_key(
        self, settings: config.Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            httpx2, "get", lambda *a, **k: pytest.fail("no request expected")
        )
        assert "web_search" not in build_toolbox(settings).tools
        settings.brave_api_key = "brave-key"
        assert "web_search" in build_toolbox(settings).tools

    def test_request(self, tool: Tools, brave: dict[str, Any]) -> None:
        tool.web_search("uv sync site:astral.sh")
        assert brave["calls"] == [
            {
                "url": "https://api.search.brave.com/res/v1/web/search",
                "params": {"q": "uv sync site:astral.sh", "count": 5},
                "headers": {
                    "X-Subscription-Token": "brave-key",
                    "Accept": "application/json",
                },
                "timeout": web.WEB_TIMEOUT_SECONDS,
            }
        ]

    @pytest.mark.parametrize("count", [1, 20])
    def test_count_is_passed_through(
        self, tool: Tools, brave: dict[str, Any], count: int
    ) -> None:
        tool.web_search("q", count)
        assert brave["calls"][0]["params"]["count"] == count

    @pytest.mark.parametrize("count", [0, -1, 21])
    def test_count_out_of_range(
        self, tool: Tools, brave: dict[str, Any], count: int
    ) -> None:
        with pytest.raises(ValueError, match="count must be between 1 and 20"):
            tool.web_search("q", count)
        assert brave["calls"] == []

    def test_results_are_cleaned(self, tool: Tools, brave: dict[str, Any]) -> None:
        brave["response"] = FakeSearchResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "A &amp; <strong>B</strong>",
                            "url": "https://example.com/1",
                            "description": "Use <strong>uv</strong> to sync &lt;deps&gt;.",
                            "extra_field": "ignored",
                        },
                        {"url": "https://example.com/2"},
                    ]
                }
            }
        )
        assert tool.web_search("q") == {
            "ok": True,
            "results": [
                {
                    "title": "A & B",
                    "url": "https://example.com/1",
                    "description": "Use uv to sync <deps>.",
                },
                {"title": "", "url": "https://example.com/2", "description": ""},
            ],
        }

    @pytest.mark.parametrize(
        "payload",
        [{}, {"web": {}}, {"web": {"results": []}}, {"query": {"original": "q"}}],
    )
    def test_no_results(
        self, tool: Tools, brave: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        brave["response"] = FakeSearchResponse(payload)
        assert tool.web_search("q") == {"ok": True, "results": []}

    def test_http_errors_propagate(self, tool: Tools, brave: dict[str, Any]) -> None:
        brave["response"] = FakeSearchResponse(
            None, error=RuntimeError("429 Too Many Requests")
        )
        with pytest.raises(RuntimeError, match="429"):
            tool.web_search("q")
