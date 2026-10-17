import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.tool_param import ToolParam
from pydantic import BaseModel, ConfigDict

# The most any tool returns at once; longer results come in parts or clipped.
MAX_TOOL_OUTPUT_BYTES = 16 * 1024


class Workspace:
    """The directory the agent works in; tool paths must resolve inside it."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        """Resolve a path and reject locations outside the workspace."""
        file_path = Path(path)
        resolved_path = (
            file_path if file_path.is_absolute() else self.root / file_path
        ).resolve()
        if not resolved_path.is_relative_to(self.root):
            raise ValueError(f"Path is outside the workspace: {path}")
        return resolved_path


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Tool[ArgumentsT: ToolArguments](ABC):
    """A function the model can call.

    A subclass's docstring is the tool description sent to the model, and its
    arguments_model supplies the JSON schema.
    """

    name: ClassVar[str]
    arguments_model: type[ArgumentsT]

    @abstractmethod
    def run(self, args: ArgumentsT) -> dict[str, Any]:
        """Run the tool and return a result with an "ok" key.

        Raises an exception for bad input; Toolbox turns it into an error result.
        """

    def schema(self) -> ToolParam:
        return {
            "type": "function",
            "name": self.name,
            "description": inspect.getdoc(self),
            "parameters": self.arguments_model.model_json_schema(),
            "strict": True,
        }


class Toolbox:
    """The tools offered to the model, looked up by name."""

    def __init__(self, tools: list[Tool[Any]]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[ToolParam]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(self, call: ResponseFunctionToolCall) -> dict[str, Any]:
        """Execute a single function call and normalize failures into a result payload."""
        tool = self.tools.get(call.name)
        if tool is None:
            return {"ok": False, "message": f"Unknown tool: {call.name}"}

        try:
            return tool.run(tool.arguments_model.model_validate_json(call.arguments))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
