from pathlib import Path

import pytest

import tools


class TestResolveWorkspacePath:
    def test_relative_path(self, workspace: Path) -> None:
        assert (
            tools.resolve_workspace_path("a/b.txt")
            == workspace.resolve() / "a" / "b.txt"
        )

    def test_absolute_path_inside(self, workspace: Path) -> None:
        target = workspace.resolve() / "inside.txt"
        assert tools.resolve_workspace_path(str(target)) == target

    def test_workspace_root_itself(self, workspace: Path) -> None:
        assert tools.resolve_workspace_path(".") == workspace.resolve()

    @pytest.mark.parametrize(
        "path", ["../outside.txt", "a/../../outside.txt", "/etc/hosts"]
    )
    def test_outside_is_rejected(self, path: str) -> None:
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.resolve_workspace_path(path)

    def test_sibling_with_same_prefix_is_rejected(self, workspace: Path) -> None:
        sibling = workspace.parent / (workspace.name + "-other")
        sibling.mkdir()
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.resolve_workspace_path(str(sibling / "x.txt"))

    def test_symlink_escape_is_rejected(self, workspace: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        (workspace / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.resolve_workspace_path("link/secret.txt")


LIMIT = tools.MAX_TOOL_OUTPUT_BYTES


class TestReadFile:
    @pytest.fixture
    def numbered(self, workspace: Path) -> Path:
        path = workspace / "numbers.txt"
        path.write_text("".join(f"line {i}\n" for i in range(1, 11)))
        return path

    def test_whole_file(self, numbered: Path) -> None:
        result = tools.read_file("numbers.txt")
        assert result == {
            "ok": True,
            "content": numbered.read_text(),
            "start_line": 1,
            "next_start_line": None,
        }

    def test_inclusive_range(self, numbered: Path) -> None:
        result = tools.read_file("numbers.txt", 3, 5)
        assert result["content"] == "line 3\nline 4\nline 5\n"
        assert result["start_line"] == 3
        assert result["next_start_line"] is None

    def test_start_only(self, numbered: Path) -> None:
        assert tools.read_file("numbers.txt", 9)["content"] == "line 9\nline 10\n"

    def test_end_only(self, numbered: Path) -> None:
        assert tools.read_file("numbers.txt", None, 2)["content"] == "line 1\nline 2\n"

    def test_single_line(self, numbered: Path) -> None:
        assert tools.read_file("numbers.txt", 4, 4)["content"] == "line 4\n"

    def test_start_past_the_end(self, numbered: Path) -> None:
        result = tools.read_file("numbers.txt", 50)
        assert result["content"] == ""
        assert result["next_start_line"] is None

    def test_empty_file(self, workspace: Path) -> None:
        (workspace / "empty.txt").write_text("")
        assert tools.read_file("empty.txt")["content"] == ""

    def test_file_without_trailing_newline(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("a\nb")
        assert tools.read_file("f.txt")["content"] == "a\nb"

    @pytest.mark.parametrize(
        ("start", "end", "message"),
        [
            (0, None, "start_line must be greater than zero"),
            (-1, None, "start_line must be greater than zero"),
            (None, 0, "end_line must be greater than zero"),
            (5, 4, "end_line must be greater than or equal to start_line"),
        ],
    )
    def test_invalid_ranges(
        self, numbered: Path, start: int | None, end: int | None, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            tools.read_file("numbers.txt", start, end)

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            tools.read_file("missing.txt")

    def test_directory(self, workspace: Path) -> None:
        (workspace / "sub").mkdir()
        with pytest.raises(OSError):
            tools.read_file("sub")

    def test_outside_workspace(self) -> None:
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.read_file("../x.txt")

    def test_non_utf8_file(self, workspace: Path) -> None:
        (workspace / "binary.bin").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(UnicodeDecodeError):
            tools.read_file("binary.bin")

    @pytest.fixture
    def big(self, workspace: Path) -> Path:
        """3000 ten-byte lines: 30 KB, so nearly two full parts."""
        path = workspace / "big.txt"
        path.write_text("".join(f"line {i:04d}\n" for i in range(1, 3001)))
        return path

    def test_long_file_is_paged_on_line_boundaries(self, big: Path) -> None:
        parts, start = [], None
        while True:
            result = tools.read_file("big.txt", start)
            assert len(result["content"].encode()) <= LIMIT
            assert result["content"].endswith("\n")
            parts.append(result["content"])
            if result["next_start_line"] is None:
                break
            start = result["next_start_line"]
        assert len(parts) == 2
        assert "".join(parts) == big.read_text()

    def test_first_part_reports_where_to_continue(self, big: Path) -> None:
        result = tools.read_file("big.txt")
        lines_that_fit = LIMIT // 10
        assert result["content"].count("\n") == lines_that_fit
        assert result["start_line"] == 1
        assert result["next_start_line"] == lines_that_fit + 1

    def test_continuation_starts_at_the_requested_line(self, big: Path) -> None:
        result = tools.read_file("big.txt", 2000)
        assert result["content"].startswith("line 2000\n")
        assert result["start_line"] == 2000
        assert result["next_start_line"] is None

    def test_range_within_the_limit_is_not_paged(self, big: Path) -> None:
        result = tools.read_file("big.txt", 3, 4)
        assert result["content"] == "line 0003\nline 0004\n"
        assert result["next_start_line"] is None

    def test_range_over_the_limit_is_paged(self, big: Path) -> None:
        result = tools.read_file("big.txt", 1, 2500)
        assert result["next_start_line"] == LIMIT // 10 + 1

    def test_overlong_line_is_cut_with_a_notice(self, workspace: Path) -> None:
        (workspace / "long.txt").write_text("a\n" + "x" * (LIMIT * 2) + "\nlast\n")

        first = tools.read_file("long.txt")
        assert (first["content"], first["next_start_line"]) == ("a\n", 2)

        second = tools.read_file("long.txt", 2)
        marker = "\n[line 2 truncated; output limit reached]"
        assert second["content"].endswith(marker)
        assert second["content"].startswith("x" * 100)
        assert len(second["content"].encode()) == LIMIT
        assert second["next_start_line"] == 3

        third = tools.read_file("long.txt", 3)
        assert (third["content"], third["next_start_line"]) == ("last\n", None)

    def test_multibyte_text_is_measured_in_bytes(self, workspace: Path) -> None:
        line = "é" * 5000 + "\n"  # 10001 bytes, so only one line fits
        (workspace / "utf.txt").write_text(line * 2, encoding="utf-8")
        result = tools.read_file("utf.txt")
        assert result["content"] == line
        assert result["next_start_line"] == 2


class TestWriteFile:
    def test_creates_file_and_parent_directories(self, workspace: Path) -> None:
        result = tools.write_file("a/b/c.txt", "hello")
        assert result["ok"] is True
        assert (workspace / "a/b/c.txt").read_text() == "hello"

    def test_replaces_existing_content(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("old content")
        tools.write_file("f.txt", "new")
        assert (workspace / "f.txt").read_text() == "new"

    def test_append(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("one\n")
        result = tools.write_file("f.txt", "two\n", append=True)
        assert (workspace / "f.txt").read_text() == "one\ntwo\n"
        assert result["message"].startswith("appended 4 bytes")

    def test_append_creates_a_missing_file(self, workspace: Path) -> None:
        tools.write_file("new.txt", "x", append=True)
        assert (workspace / "new.txt").read_text() == "x"

    def test_message_counts_bytes_not_characters(self) -> None:
        assert tools.write_file("f.txt", "é")["message"].startswith("written 2 bytes")

    def test_message_names_the_path_as_given(self) -> None:
        assert tools.write_file("a/b.txt", "xyz") == {
            "ok": True,
            "message": "written 3 bytes to a/b.txt",
        }

    def test_empty_content(self, workspace: Path) -> None:
        tools.write_file("empty.txt", "")
        assert (workspace / "empty.txt").read_text() == ""

    def test_outside_workspace(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.write_file("../escape.txt", "x")
        assert not (workspace.parent / "escape.txt").exists()


class TestEditFile:
    def test_replaces_a_unique_match(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("alpha beta gamma")
        result = tools.edit_file("f.txt", "beta", "BETA")
        assert result == {"ok": True, "message": "replaced 1 match(es) in f.txt."}
        assert (workspace / "f.txt").read_text() == "alpha BETA gamma"

    def test_missing_text(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("alpha")
        with pytest.raises(ValueError, match="old_content not found"):
            tools.edit_file("f.txt", "beta", "x")
        assert (workspace / "f.txt").read_text() == "alpha"

    def test_ambiguous_match_is_rejected(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("x x x")
        with pytest.raises(ValueError, match="found 3 matches"):
            tools.edit_file("f.txt", "x", "y")
        assert (workspace / "f.txt").read_text() == "x x x"

    def test_replace_all(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("x x x")
        result = tools.edit_file("f.txt", "x", "y", replace_all=True)
        assert result["message"] == "replaced 3 match(es) in f.txt."
        assert (workspace / "f.txt").read_text() == "y y y"

    def test_replace_all_with_a_single_match(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("only")
        tools.edit_file("f.txt", "only", "one", replace_all=True)
        assert (workspace / "f.txt").read_text() == "one"

    def test_empty_old_content(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("alpha")
        with pytest.raises(ValueError, match="must not be empty"):
            tools.edit_file("f.txt", "", "x")

    def test_delete_by_replacing_with_nothing(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("keep remove keep")
        tools.edit_file("f.txt", " remove", "")
        assert (workspace / "f.txt").read_text() == "keep keep"

    def test_multiline_match(self, workspace: Path) -> None:
        (workspace / "f.py").write_text("def f():\n    return 1\n")
        tools.edit_file("f.py", "def f():\n    return 1", "def f():\n    return 2")
        assert (workspace / "f.py").read_text() == "def f():\n    return 2\n"

    def test_line_endings_are_preserved(self, workspace: Path) -> None:
        (workspace / "crlf.txt").write_bytes(b"a\r\nb\r\n")
        tools.edit_file("crlf.txt", "b", "c")
        assert (workspace / "crlf.txt").read_bytes() == b"a\r\nc\r\n"

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            tools.edit_file("missing.txt", "a", "b")

    def test_outside_workspace(self) -> None:
        with pytest.raises(ValueError, match="outside the workspace"):
            tools.edit_file("../x.txt", "a", "b")
