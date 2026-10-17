from itertools import islice
from typing import Any

from pydantic import Field

from tools import MAX_TOOL_OUTPUT_BYTES, Tool, ToolArguments, Workspace


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


class ReadFileTool(Tool[ReadFileArguments]):
    """Read a file, optionally limited to an inclusive line range. When the file continues past the returned lines (because of end_line or the output limit), next_start_line is set: pass it as start_line to read on, together with next_offset as offset when that is set (a single line longer than the limit comes in pieces)."""

    name = "read_file"
    arguments_model = ReadFileArguments

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def run(self, args: ReadFileArguments) -> dict[str, Any]:
        if args.start_line is not None and args.start_line < 1:
            raise ValueError("start_line must be greater than zero")
        if args.offset is not None and args.offset < 0:
            raise ValueError("offset must not be negative")
        if args.end_line is not None and args.end_line < 1:
            raise ValueError("end_line must be greater than zero")
        if (
            args.start_line is not None
            and args.end_line is not None
            and args.end_line < args.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line")

        file_path = self.workspace.resolve(args.path)
        first_line = 1 if args.start_line is None else args.start_line

        lines: list[str] = []
        size = 0
        next_start_line = None
        next_offset = None
        with file_path.open(encoding="utf-8") as file:
            for number, line in enumerate(
                islice(file, first_line - 1, None), first_line
            ):
                if args.end_line is not None and number > args.end_line:
                    # The file goes on past end_line; say so, or the model
                    # takes end_line for the end of the file.
                    next_start_line = number
                    break
                if number == first_line and args.offset:
                    line = line[args.offset :]
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
                        next_offset = (args.offset or 0) + len(piece)
                    break
                lines.append(line)

        result: dict[str, Any] = {
            "ok": True,
            "content": "".join(lines),
            "next_start_line": next_start_line,
            "next_offset": next_offset,
        }
        if next_start_line is not None:
            continue_with = f"start_line={next_start_line}" + (
                f" and offset={next_offset}" if next_offset is not None else ""
            )
            result["message"] = (
                "The file continues past the returned content; call read_file "
                f"again with {continue_with} to read on."
            )
        return result


class WriteFileTool(Tool[WriteFileArguments]):
    """Write content to a file, replacing it by default or appending when append is true."""

    name = "write_file"
    arguments_model = WriteFileArguments

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def run(self, args: WriteFileArguments) -> dict[str, Any]:
        file_path = self.workspace.resolve(args.path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with file_path.open(mode, encoding="utf-8") as file:
            file.write(args.content)

        operation = "appended" if args.append else "written"
        return {
            "ok": True,
            "message": f"{operation} {len(args.content.encode('utf-8'))} bytes to {args.path}",
        }


class EditFileTool(Tool[EditFileArguments]):
    """Replace a unique text fragment, or replace all matches when replace_all is true."""

    name = "edit_file"
    arguments_model = EditFileArguments

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def run(self, args: EditFileArguments) -> dict[str, Any]:
        if not args.old_content:
            raise ValueError("old_content must not be empty")

        file_path = self.workspace.resolve(args.path)
        existing_content = file_path.read_text(encoding="utf-8", newline="")

        old_content, new_content = args.old_content, args.new_content
        match_count = existing_content.count(old_content)
        if match_count == 0 and "\r\n" in existing_content and "\r" not in old_content:
            # read_file shows CRLF files with plain "\n", so text copied from it
            # only matches once its line endings are converted back.
            old_content = old_content.replace("\n", "\r\n")
            new_content = new_content.replace("\r\n", "\n").replace("\n", "\r\n")
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
            raise ValueError(f"old_content not found in {args.path}{hint}")
        if match_count > 1 and not args.replace_all:
            raise ValueError(
                f"old_content must match exactly once in {args.path}; found {match_count} matches"
            )

        file_path.write_text(
            existing_content.replace(old_content, new_content),
            encoding="utf-8",
            newline="",
        )

        return {
            "ok": True,
            "message": f"replaced {match_count} match(es) in {args.path}.",
        }
