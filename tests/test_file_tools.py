from pathlib import Path

import pytest

from conftest import Tools
from minime import tools

LIMIT = tools.MAX_TOOL_OUTPUT_BYTES


class TestReadFile:
    @pytest.fixture
    def numbered(self, workspace: Path) -> Path:
        path = workspace / "numbers.txt"
        path.write_text("".join(f"line {i}\n" for i in range(1, 11)))
        return path

    def test_whole_file(self, tool: Tools, numbered: Path) -> None:
        result = tool.read_file("numbers.txt")
        assert result == {
            "ok": True,
            "content": numbered.read_text(),
            "next_start_line": None,
            "next_offset": None,
        }

    def test_inclusive_range(self, tool: Tools, numbered: Path) -> None:
        result = tool.read_file("numbers.txt", 3, 5)
        assert result["content"] == "line 3\nline 4\nline 5\n"
        # The file goes on, and the result says where.
        assert result["next_start_line"] == 6
        assert result["next_offset"] is None
        assert result["message"] == (
            "The file continues past the returned content; call read_file "
            "again with start_line=6 to read on."
        )

    @pytest.mark.parametrize("end", [10, 11, 500])
    def test_end_line_at_or_past_the_end_of_the_file(
        self, tool: Tools, numbered: Path, end: int
    ) -> None:
        result = tool.read_file("numbers.txt", 9, end)
        assert result["content"] == "line 9\nline 10\n"
        assert result["next_start_line"] is None
        assert "message" not in result

    def test_whole_file_has_no_continuation_message(
        self, tool: Tools, numbered: Path
    ) -> None:
        assert "message" not in tool.read_file("numbers.txt")

    def test_following_next_start_line_reads_everything_once(
        self, tool: Tools, numbered: Path
    ) -> None:
        parts, start = [], 1
        while start is not None:
            result = tool.read_file("numbers.txt", start, start + 2)
            parts.append(result["content"])
            start = result["next_start_line"]
        assert "".join(parts) == numbered.read_text()
        assert len(parts) == 4

    def test_start_only(self, tool: Tools, numbered: Path) -> None:
        assert tool.read_file("numbers.txt", 9)["content"] == "line 9\nline 10\n"

    def test_end_only(self, tool: Tools, numbered: Path) -> None:
        assert tool.read_file("numbers.txt", None, 2)["content"] == "line 1\nline 2\n"

    def test_single_line(self, tool: Tools, numbered: Path) -> None:
        assert tool.read_file("numbers.txt", 4, 4)["content"] == "line 4\n"

    def test_start_past_the_end(self, tool: Tools, numbered: Path) -> None:
        result = tool.read_file("numbers.txt", 50)
        assert result["content"] == ""
        assert result["next_start_line"] is None

    def test_empty_file(self, tool: Tools, workspace: Path) -> None:
        (workspace / "empty.txt").write_text("")
        assert tool.read_file("empty.txt")["content"] == ""

    def test_file_without_trailing_newline(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("a\nb")
        assert tool.read_file("f.txt")["content"] == "a\nb"

    @pytest.mark.parametrize(
        ("start", "end", "message"),
        [
            (0, None, "start_line\n  Input should be greater than or equal to 1"),
            (-1, None, "start_line\n  Input should be greater than or equal to 1"),
            (None, 0, "end_line\n  Input should be greater than or equal to 1"),
            (5, 4, "end_line must be greater than or equal to start_line"),
        ],
    )
    def test_invalid_ranges(
        self,
        tool: Tools,
        numbered: Path,
        start: int | None,
        end: int | None,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            tool.read_file("numbers.txt", start, end)

    def test_missing_file(self, tool: Tools) -> None:
        with pytest.raises(FileNotFoundError):
            tool.read_file("missing.txt")

    def test_directory(self, tool: Tools, workspace: Path) -> None:
        (workspace / "sub").mkdir()
        with pytest.raises(OSError):
            tool.read_file("sub")

    def test_outside_workspace(self, tool: Tools, workspace: Path) -> None:
        (workspace.parent / "x.txt").write_text("outside\n")
        assert tool.read_file("../x.txt")["content"] == "outside\n"

    def test_absolute_path(self, tool: Tools, tmp_path: Path) -> None:
        (tmp_path / "abs.txt").write_text("absolute\n")
        assert tool.read_file(str(tmp_path / "abs.txt"))["content"] == "absolute\n"

    def test_home_directory(
        self, tool: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "home.txt").write_text("at home\n")
        assert tool.read_file("~/home.txt")["content"] == "at home\n"

    def test_non_utf8_file(self, tool: Tools, workspace: Path) -> None:
        (workspace / "binary.bin").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(UnicodeDecodeError):
            tool.read_file("binary.bin")

    @pytest.fixture
    def big(self, workspace: Path) -> Path:
        """3000 ten-byte lines: 30 KB, so nearly two full parts."""
        path = workspace / "big.txt"
        path.write_text("".join(f"line {i:04d}\n" for i in range(1, 3001)))
        return path

    def test_long_file_is_paged_on_line_boundaries(
        self, tool: Tools, big: Path
    ) -> None:
        parts, start = [], None
        while True:
            result = tool.read_file("big.txt", start)
            assert len(result["content"].encode()) <= LIMIT
            assert result["content"].endswith("\n")
            parts.append(result["content"])
            if result["next_start_line"] is None:
                break
            start = result["next_start_line"]
        assert len(parts) == 2
        assert "".join(parts) == big.read_text()

    def test_first_part_reports_where_to_continue(self, tool: Tools, big: Path) -> None:
        result = tool.read_file("big.txt")
        lines_that_fit = LIMIT // 10
        assert result["content"].count("\n") == lines_that_fit
        assert result["next_start_line"] == lines_that_fit + 1

    def test_continuation_starts_at_the_requested_line(
        self, tool: Tools, big: Path
    ) -> None:
        result = tool.read_file("big.txt", 2000)
        assert result["content"].startswith("line 2000\n")
        assert result["next_start_line"] is None

    def test_range_within_the_limit_still_says_where_to_continue(
        self, tool: Tools, big: Path
    ) -> None:
        result = tool.read_file("big.txt", 3, 4)
        assert result["content"] == "line 0003\nline 0004\n"
        assert result["next_start_line"] == 5

    def test_range_over_the_limit_is_paged(self, tool: Tools, big: Path) -> None:
        result = tool.read_file("big.txt", 1, 2500)
        assert result["next_start_line"] == LIMIT // 10 + 1

    def read_all(self, tool: Tools, path: str) -> tuple[str, int]:
        """Follow next_start_line / next_offset to the end; return the text and call count."""
        parts, start, offset = [], None, None
        while True:
            result = tool.read_file(path, start, None, offset)
            assert len(result["content"].encode()) <= LIMIT
            parts.append(result["content"])
            if result["next_start_line"] is None:
                assert result["next_offset"] is None
                return "".join(parts), len(parts)
            start, offset = result["next_start_line"], result["next_offset"]

    def test_overlong_line_is_returned_in_pieces(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "long.txt").write_text("a\n" + "x" * (LIMIT * 2 + 5) + "\nlast\n")

        first = tool.read_file("long.txt")
        assert (first["content"], first["next_start_line"], first["next_offset"]) == (
            "a\n",
            2,
            None,
        )

        second = tool.read_file("long.txt", 2)
        assert second["message"].endswith(
            f"again with start_line=2 and offset={LIMIT} to read on."
        )
        assert second["content"] == "x" * LIMIT
        assert (second["next_start_line"], second["next_offset"]) == (2, LIMIT)

        third = tool.read_file("long.txt", 2, None, LIMIT)
        assert third["content"] == "x" * LIMIT
        assert (third["next_start_line"], third["next_offset"]) == (2, LIMIT * 2)

        # The rest of the line now fits, so reading carries on into later lines.
        fourth = tool.read_file("long.txt", 2, None, LIMIT * 2)
        assert fourth["content"] == "xxxxx\nlast\n"
        assert (fourth["next_start_line"], fourth["next_offset"]) == (None, None)

    def test_file_with_overlong_lines_can_be_read_completely(
        self, tool: Tools, workspace: Path
    ) -> None:
        text = "start\n" + "é" * LIMIT + "\nmiddle\n" + "z" * (LIMIT * 3) + "\nend\n"
        (workspace / "mixed.txt").write_text(text, encoding="utf-8")
        content, calls = self.read_all(tool, "mixed.txt")
        assert content == text
        assert calls > 4

    def test_minified_single_line_file(self, tool: Tools, workspace: Path) -> None:
        text = "{" + ",".join(f'"k{i}":{i}' for i in range(8000)) + "}"
        (workspace / "min.json").write_text(text)
        assert len(text) > LIMIT * 4
        assert self.read_all(tool, "min.json")[0] == text

    def test_offset_within_a_short_line(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("abcdef\nsecond\n")
        result = tool.read_file("f.txt", 1, None, 3)
        assert result["content"] == "def\nsecond\n"
        assert result["next_offset"] is None

    def test_offset_applies_only_to_the_first_line(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("abcdef\nghijkl\n")
        assert tool.read_file("f.txt", 2, None, 2)["content"] == "ijkl\n"

    def test_offset_past_the_end_of_the_line(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("abc\nnext\n")
        assert tool.read_file("f.txt", 1, None, 99)["content"] == "next\n"

    def test_negative_offset(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("abc\n")
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            tool.read_file("f.txt", 1, None, -1)

    def test_overlong_line_pieces_are_cut_between_characters(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "utf.txt").write_text(
            "é" * LIMIT, encoding="utf-8"
        )  # 2 x LIMIT bytes
        first = tool.read_file("utf.txt")
        assert first["content"] == "é" * (LIMIT // 2)
        assert first["next_offset"] == LIMIT // 2  # counted in characters

    def test_multibyte_text_is_measured_in_bytes(
        self, tool: Tools, workspace: Path
    ) -> None:
        line = "é" * 5000 + "\n"  # 10001 bytes, so only one line fits
        (workspace / "utf.txt").write_text(line * 2, encoding="utf-8")
        result = tool.read_file("utf.txt")
        assert result["content"] == line
        assert result["next_start_line"] == 2


class TestWriteFile:
    def test_creates_file_and_parent_directories(
        self, tool: Tools, workspace: Path
    ) -> None:
        result = tool.write_file("a/b/c.txt", "hello")
        assert result["ok"] is True
        assert (workspace / "a/b/c.txt").read_text() == "hello"

    def test_replaces_existing_content(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("old content")
        tool.write_file("f.txt", "new")
        assert (workspace / "f.txt").read_text() == "new"

    def test_append(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("one\n")
        result = tool.write_file("f.txt", "two\n", append=True)
        assert (workspace / "f.txt").read_text() == "one\ntwo\n"
        assert result["message"].startswith("appended 4 bytes")

    def test_append_creates_a_missing_file(self, tool: Tools, workspace: Path) -> None:
        tool.write_file("new.txt", "x", append=True)
        assert (workspace / "new.txt").read_text() == "x"

    def test_message_counts_bytes_not_characters(self, tool: Tools) -> None:
        assert tool.write_file("f.txt", "é")["message"].startswith("written 2 bytes")

    def test_message_names_the_path_as_given(self, tool: Tools) -> None:
        assert tool.write_file("a/b.txt", "xyz") == {
            "ok": True,
            "message": "written 3 bytes to a/b.txt",
        }

    def test_empty_content(self, tool: Tools, workspace: Path) -> None:
        tool.write_file("empty.txt", "")
        assert (workspace / "empty.txt").read_text() == ""

    def test_outside_workspace(self, tool: Tools, workspace: Path) -> None:
        tool.write_file("../escape.txt", "x")
        assert (workspace.parent / "escape.txt").read_text() == "x"

    def test_home_directory(
        self, tool: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        tool.write_file("~/new/home.txt", "x")
        assert (tmp_path / "new" / "home.txt").read_text() == "x"


class TestEditFile:
    def test_replaces_a_unique_match(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("alpha beta gamma")
        result = tool.edit_file("f.txt", "beta", "BETA")
        assert result == {"ok": True, "message": "replaced 1 match(es) in f.txt."}
        assert (workspace / "f.txt").read_text() == "alpha BETA gamma"

    def test_missing_text(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("alpha")
        with pytest.raises(ValueError, match="old_content not found"):
            tool.edit_file("f.txt", "beta", "x")
        assert (workspace / "f.txt").read_text() == "alpha"

    HINT = "the same text is there with different whitespace"

    def test_missing_text_gets_no_whitespace_hint(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("alpha")
        with pytest.raises(ValueError) as error:
            tool.edit_file("f.txt", "beta", "x")
        assert str(error.value) == "old_content not found in f.txt"

    @pytest.mark.parametrize(
        ("content", "old_content"),
        [
            ("def f():\n    return 1\n", "def f():\n  return 1"),  # indentation
            ("def f():\n\treturn 1\n", "def f():\n    return 1"),  # tabs vs spaces
            ("x = [1,  2]\n", "x = [1, 2]"),  # spacing
            ("one\n\ntwo\n", "one\ntwo"),  # blank line
        ],
    )
    def test_whitespace_near_miss_is_pointed_out(
        self, tool: Tools, workspace: Path, content: str | bytes, old_content: str
    ) -> None:
        path = workspace / "f.py"
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        before = path.read_bytes()
        with pytest.raises(ValueError, match=self.HINT):
            tool.edit_file("f.py", old_content, "replacement")
        assert path.read_bytes() == before

    def test_whitespace_only_old_content_gets_no_hint(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("alpha beta")
        with pytest.raises(ValueError) as error:
            tool.edit_file("f.txt", "\t\t", "x")
        assert self.HINT not in str(error.value)

    def test_ambiguous_match_is_rejected(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("x x x")
        with pytest.raises(ValueError, match="found 3 matches"):
            tool.edit_file("f.txt", "x", "y")
        assert (workspace / "f.txt").read_text() == "x x x"

    def test_replace_all(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("x x x")
        result = tool.edit_file("f.txt", "x", "y", replace_all=True)
        assert result["message"] == "replaced 3 match(es) in f.txt."
        assert (workspace / "f.txt").read_text() == "y y y"

    def test_replace_all_with_a_single_match(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("only")
        tool.edit_file("f.txt", "only", "one", replace_all=True)
        assert (workspace / "f.txt").read_text() == "one"

    def test_empty_old_content(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.txt").write_text("alpha")
        with pytest.raises(ValueError, match="must not be empty"):
            tool.edit_file("f.txt", "", "x")

    def test_delete_by_replacing_with_nothing(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("keep remove keep")
        tool.edit_file("f.txt", " remove", "")
        assert (workspace / "f.txt").read_text() == "keep keep"

    def test_multiline_match(self, tool: Tools, workspace: Path) -> None:
        (workspace / "f.py").write_text("def f():\n    return 1\n")
        tool.edit_file("f.py", "def f():\n    return 1", "def f():\n    return 2")
        assert (workspace / "f.py").read_text() == "def f():\n    return 2\n"

    def test_text_copied_from_read_file_edits_a_crlf_file(
        self, tool: Tools, workspace: Path
    ) -> None:
        path = workspace / "crlf.py"
        path.write_bytes(b"def f():\r\n    return 1\r\n\r\nprint(f())\r\n")
        shown = tool.read_file("crlf.py")["content"]
        assert "\r" not in shown  # read_file shows plain newlines

        old = "def f():\n    return 1\n"
        assert old in shown
        result = tool.edit_file("crlf.py", old, "def f():\n    return 2\n")
        assert result["ok"] is True
        # The new text takes the file's own line endings.
        assert path.read_bytes() == b"def f():\r\n    return 2\r\n\r\nprint(f())\r\n"

    def test_crlf_text_still_matches_exactly(
        self, tool: Tools, workspace: Path
    ) -> None:
        path = workspace / "crlf.txt"
        path.write_bytes(b"a\r\nb\r\n")
        tool.edit_file("crlf.txt", "a\r\nb", "a\r\nc")
        assert path.read_bytes() == b"a\r\nc\r\n"

    def test_new_text_with_crlf_is_not_doubled(
        self, tool: Tools, workspace: Path
    ) -> None:
        path = workspace / "crlf.txt"
        path.write_bytes(b"one\r\ntwo\r\n")
        tool.edit_file("crlf.txt", "one\ntwo\n", "uno\r\ndos\r\n")
        assert path.read_bytes() == b"uno\r\ndos\r\n"

    def test_lf_file_is_not_converted(self, tool: Tools, workspace: Path) -> None:
        path = workspace / "lf.txt"
        path.write_bytes(b"a\nb\n")
        tool.edit_file("lf.txt", "a\nb", "a\nc")
        assert path.read_bytes() == b"a\nc\n"

    def test_crlf_file_with_absent_text_still_fails(
        self, tool: Tools, workspace: Path
    ) -> None:
        (workspace / "crlf.txt").write_bytes(b"a\r\nb\r\n")
        with pytest.raises(ValueError, match="old_content not found"):
            tool.edit_file("crlf.txt", "x\ny", "z")

    def test_line_endings_are_preserved(self, tool: Tools, workspace: Path) -> None:
        (workspace / "crlf.txt").write_bytes(b"a\r\nb\r\n")
        tool.edit_file("crlf.txt", "b", "c")
        assert (workspace / "crlf.txt").read_bytes() == b"a\r\nc\r\n"

    def test_missing_file(self, tool: Tools) -> None:
        with pytest.raises(FileNotFoundError):
            tool.edit_file("missing.txt", "a", "b")

    def test_outside_workspace(self, tool: Tools, workspace: Path) -> None:
        (workspace.parent / "x.txt").write_text("a")
        tool.edit_file("../x.txt", "a", "b")
        assert (workspace.parent / "x.txt").read_text() == "b"

    def test_home_directory(
        self, tool: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "home.txt").write_text("a")
        tool.edit_file("~/home.txt", "a", "b")
        assert (tmp_path / "home.txt").read_text() == "b"
