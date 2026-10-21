import inspect
import json
from abc import ABC, abstractmethod
from textwrap import shorten
from typing import Any, ClassVar, cast

import openai
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.tool_param import ToolParam
from pydantic import BaseModel, ConfigDict

# The most any tool returns at once; longer results come in parts or clipped.
MAX_TOOL_OUTPUT_BYTES = 16 * 1024


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
        tool = openai.pydantic_responses_function_tool(
            self.arguments_model, name=self.name, description=inspect.getdoc(self)
        )
        # A plain dict: with the model attached, the SDK would parse the
        # arguments itself and abort the turn on invalid ones, instead of
        # Toolbox.execute reporting them to the model.
        return cast(ToolParam, dict(tool))


class Toolbox:
    """The tools offered to the model, looked up by name."""

    def __init__(self, tools: list[Tool[Any]]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[ToolParam]:
        return [tool.schema() for tool in self.tools.values()]

    def describe(self, call: ResponseFunctionToolCall) -> str:
        """The call's name and its first text argument in the tool's field
        order (the model's order varies), e.g. "write_file: src/x.py"."""
        tool = self.tools.get(call.name)
        try:
            arguments = json.loads(call.arguments)
        except ValueError:
            arguments = None
        if tool and isinstance(arguments, dict):
            for field in tool.arguments_model.model_fields:
                value = arguments.get(field)
                if isinstance(value, str) and value.strip():
                    return f"{call.name}: {shorten(value, 80, placeholder=' …')}"
        return call.name

    def execute(self, call: ResponseFunctionToolCall) -> dict[str, Any]:
        """Execute a single function call and normalize failures into a result payload."""
        tool = self.tools.get(call.name)
        if tool is None:
            return {"ok": False, "message": f"Unknown tool: {call.name}"}

        try:
            return tool.run(tool.arguments_model.model_validate_json(call.arguments))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
