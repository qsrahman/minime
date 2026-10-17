import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import raw_tool_call, tool_call
from minime import config, fs, tools
from minime.agent import build_toolbox

ALL_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "run_command",
    "web_fetch",
    "web_search",
]


def toolbox_for(settings: config.Config) -> tools.Toolbox:
    return build_toolbox(settings)


@pytest.fixture
def toolbox(settings: config.Config) -> tools.Toolbox:
    return toolbox_for(settings)


def schema_named(settings: config.Config, name: str) -> Any:
    settings.brave_api_key = "brave-key"
    return next(s for s in toolbox_for(settings).schemas() if s["name"] == name)


class TestToolSchemas:
    def test_web_search_is_left_out_without_a_key(self, toolbox: tools.Toolbox) -> None:
        assert [schema["name"] for schema in toolbox.schemas()] == ALL_TOOLS[:-1]

    def test_web_search_is_offered_with_a_key(self, settings: config.Config) -> None:
        settings.brave_api_key = "brave-key"
        assert [
            schema["name"] for schema in toolbox_for(settings).schemas()
        ] == ALL_TOOLS

    def test_empty_key_counts_as_unset(self, settings: config.Config) -> None:
        settings.brave_api_key = ""
        assert "web_search" not in toolbox_for(settings).tools

    def test_tools_are_looked_up_by_their_name(self, settings: config.Config) -> None:
        settings.brave_api_key = "brave-key"
        toolbox = toolbox_for(settings)
        assert list(toolbox.tools) == ALL_TOOLS
        assert all(name == tool.name for name, tool in toolbox.tools.items())

    def test_schemas_are_json_serialisable(self, settings: config.Config) -> None:
        settings.brave_api_key = "brave-key"
        json.dumps(toolbox_for(settings).schemas())

    @pytest.mark.parametrize("name", ALL_TOOLS)
    def test_schema_is_strict(self, name: str, settings: config.Config) -> None:
        schema = schema_named(settings, name)
        parameters = schema["parameters"]
        assert schema["type"] == "function"
        assert schema["strict"] is True
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        # OpenAI strict mode: every property must be listed as required.
        assert sorted(parameters["required"]) == sorted(parameters["properties"])
        assert all(
            prop.get("description") for prop in parameters["properties"].values()
        )

    @pytest.mark.parametrize("name", ALL_TOOLS)
    def test_description_is_the_tool_class_docstring(
        self, name: str, settings: config.Config
    ) -> None:
        schema = schema_named(settings, name)
        tool = toolbox_for(settings).tools[name]
        assert schema["description"] == inspect.getdoc(type(tool))
        assert schema["description"]

    @pytest.mark.parametrize("name", ALL_TOOLS)
    def test_argument_fields_have_no_defaults(
        self, name: str, settings: config.Config
    ) -> None:
        settings.brave_api_key = "brave-key"
        arguments_model = toolbox_for(settings).tools[name].arguments_model
        assert all(
            field.is_required() for field in arguments_model.model_fields.values()
        )

    def test_optional_arguments_are_nullable(self) -> None:
        properties = fs.ReadFileArguments.model_json_schema()["properties"]
        assert {"type": "null"} in properties["start_line"]["anyOf"]
        assert {"type": "integer"} in properties["start_line"]["anyOf"]


class TestExecuteToolCall:
    def test_success_returns_the_handler_result(
        self, toolbox: tools.Toolbox, workspace: Path
    ) -> None:
        (workspace / "f.txt").write_text("hello\n")
        result = toolbox.execute(
            tool_call(
                "read_file", path="f.txt", start_line=None, end_line=None, offset=None
            )
        )
        assert result == {
            "ok": True,
            "content": "hello\n",
            "next_start_line": None,
            "next_offset": None,
        }

    def test_arguments_reach_the_handler(
        self, toolbox: tools.Toolbox, workspace: Path
    ) -> None:
        toolbox.execute(
            tool_call("write_file", path="out.txt", content="data", append=False)
        )
        assert (workspace / "out.txt").read_text() == "data"

    def test_unknown_tool(self, toolbox: tools.Toolbox) -> None:
        assert toolbox.execute(tool_call("launch_rocket")) == {
            "ok": False,
            "message": "Unknown tool: launch_rocket",
        }

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ("{not json", "json_invalid"),
            ("", "json_invalid"),
            ("[]", "model_type"),
            ('"text"', "model_type"),
            ("null", "model_type"),
            ('{"path": "f.txt", "start_line": null}', "missing"),
            (
                '{"path": "f.txt", "start_line": null, "end_line": null, "offset": null, "extra": 1}',
                "extra_forbidden",
            ),
            (
                '{"path": 5, "start_line": null, "end_line": null, "offset": null}',
                "string_type",
            ),
            (
                '{"path": "f.txt", "start_line": "1", "end_line": null, "offset": null}',
                "int_type",
            ),
            (
                '{"path": "f.txt", "start_line": 1.5, "end_line": null, "offset": null}',
                "int_type",
            ),
        ],
    )
    def test_invalid_arguments(
        self, toolbox: tools.Toolbox, arguments: str, expected: str
    ) -> None:
        result = toolbox.execute(raw_tool_call("read_file", arguments))
        assert result["ok"] is False
        assert result["message"].startswith("ValidationError:")
        assert expected in result["message"]

    def test_booleans_are_not_coerced(self, toolbox: tools.Toolbox) -> None:
        arguments = json.dumps({"path": "f.txt", "content": "x", "append": "false"})
        result = toolbox.execute(raw_tool_call("write_file", arguments))
        assert result["ok"] is False
        assert "bool_type" in result["message"]

    def test_handler_errors_become_results(self, toolbox: tools.Toolbox) -> None:
        result = toolbox.execute(
            tool_call(
                "read_file",
                path="missing.txt",
                start_line=None,
                end_line=None,
                offset=None,
            )
        )
        assert result["ok"] is False
        assert result["message"].startswith("FileNotFoundError:")

    def test_workspace_violations_become_results(self, toolbox: tools.Toolbox) -> None:
        result = toolbox.execute(
            tool_call("write_file", path="../x.txt", content="x", append=False)
        )
        assert result == {
            "ok": False,
            "message": "ValueError: Path is outside the workspace: ../x.txt",
        }

    def test_web_search_without_a_key_is_an_unknown_tool(
        self, toolbox: tools.Toolbox
    ) -> None:
        result = toolbox.execute(tool_call("web_search", query="q", count=None))
        assert result == {"ok": False, "message": "Unknown tool: web_search"}

    def test_keyboard_interrupt_is_not_swallowed(
        self, toolbox: tools.Toolbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(args: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        monkeypatch.setattr(toolbox.tools["run_command"], "run", interrupted)
        with pytest.raises(KeyboardInterrupt):
            toolbox.execute(tool_call("run_command", cmd="x", timeout=None, cwd=None))
