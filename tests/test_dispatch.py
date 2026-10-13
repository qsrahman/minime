import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import raw_tool_call, tool_call

import config
import tools

ALL_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "run_command",
    "web_fetch",
    "web_search",
]


class TestToolSchemas:
    def test_web_search_is_left_out_without_a_key(self) -> None:
        assert [schema["name"] for schema in tools.build_tool_schemas()] == ALL_TOOLS[
            :-1
        ]

    def test_web_search_is_offered_with_a_key(self, settings: config.Config) -> None:
        settings.brave_api_key = "brave-key"
        assert [schema["name"] for schema in tools.build_tool_schemas()] == ALL_TOOLS

    def test_empty_key_counts_as_unset(self, settings: config.Config) -> None:
        settings.brave_api_key = ""
        assert "web_search" not in [
            schema["name"] for schema in tools.build_tool_schemas()
        ]

    def test_registry_lists_every_tool(self) -> None:
        assert list(tools.TOOLS) == ALL_TOOLS

    def test_schemas_are_json_serialisable(self, settings: config.Config) -> None:
        settings.brave_api_key = "brave-key"
        json.dumps(tools.build_tool_schemas())

    @pytest.mark.parametrize("name", ALL_TOOLS)
    def test_schema_is_strict(self, name: str, settings: config.Config) -> None:
        settings.brave_api_key = "brave-key"
        schema: Any = next(s for s in tools.build_tool_schemas() if s["name"] == name)
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
    def test_description_is_the_handler_docstring(
        self, name: str, settings: config.Config
    ) -> None:
        settings.brave_api_key = "brave-key"
        schema: Any = next(s for s in tools.build_tool_schemas() if s["name"] == name)
        handler, _ = tools.TOOLS[name]
        assert schema["description"] == inspect.getdoc(handler)
        assert schema["description"]

    @pytest.mark.parametrize("name", ALL_TOOLS)
    def test_argument_model_matches_the_handler_signature(self, name: str) -> None:
        handler, arguments_model = tools.TOOLS[name]
        assert list(arguments_model.model_fields) == list(
            inspect.signature(handler).parameters
        )

    @pytest.mark.parametrize("name", ALL_TOOLS)
    def test_argument_fields_have_no_defaults(self, name: str) -> None:
        _, arguments_model = tools.TOOLS[name]
        assert all(
            field.is_required() for field in arguments_model.model_fields.values()
        )

    def test_optional_arguments_are_nullable(self) -> None:
        properties = tools.ReadFileArguments.model_json_schema()["properties"]
        assert {"type": "null"} in properties["start_line"]["anyOf"]
        assert {"type": "integer"} in properties["start_line"]["anyOf"]


class TestExecuteToolCall:
    def test_success_returns_the_handler_result(self, workspace: Path) -> None:
        (workspace / "f.txt").write_text("hello\n")
        result = tools.execute_tool_call(
            tool_call("read_file", path="f.txt", start_line=None, end_line=None)
        )
        assert result == {
            "ok": True,
            "content": "hello\n",
            "start_line": 1,
            "next_start_line": None,
        }

    def test_arguments_reach_the_handler(self, workspace: Path) -> None:
        tools.execute_tool_call(
            tool_call("write_file", path="out.txt", content="data", append=False)
        )
        assert (workspace / "out.txt").read_text() == "data"

    def test_unknown_tool(self) -> None:
        assert tools.execute_tool_call(tool_call("launch_rocket")) == {
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
                '{"path": "f.txt", "start_line": null, "end_line": null, "extra": 1}',
                "extra_forbidden",
            ),
            ('{"path": 5, "start_line": null, "end_line": null}', "string_type"),
            ('{"path": "f.txt", "start_line": "1", "end_line": null}', "int_type"),
            ('{"path": "f.txt", "start_line": 1.5, "end_line": null}', "int_type"),
        ],
    )
    def test_invalid_arguments(self, arguments: str, expected: str) -> None:
        result = tools.execute_tool_call(raw_tool_call("read_file", arguments))
        assert result["ok"] is False
        assert result["message"].startswith("ValidationError:")
        assert expected in result["message"]

    def test_booleans_are_not_coerced(self) -> None:
        arguments = json.dumps({"path": "f.txt", "content": "x", "append": "false"})
        result = tools.execute_tool_call(raw_tool_call("write_file", arguments))
        assert result["ok"] is False
        assert "bool_type" in result["message"]

    def test_handler_errors_become_results(self) -> None:
        result = tools.execute_tool_call(
            tool_call("read_file", path="missing.txt", start_line=None, end_line=None)
        )
        assert result["ok"] is False
        assert result["message"].startswith("FileNotFoundError:")

    def test_workspace_violations_become_results(self) -> None:
        result = tools.execute_tool_call(
            tool_call("write_file", path="../x.txt", content="x", append=False)
        )
        assert result == {
            "ok": False,
            "message": "ValueError: Path is outside the workspace: ../x.txt",
        }

    def test_web_search_without_a_key_is_a_result_not_a_crash(self) -> None:
        result = tools.execute_tool_call(tool_call("web_search", query="q", count=None))
        assert result == {
            "ok": False,
            "message": "ValueError: BRAVE_API_KEY is not set",
        }

    def test_keyboard_interrupt_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(**kwargs: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        monkeypatch.setitem(
            tools.TOOLS, "run_command", (interrupted, tools.RunCommandArguments)
        )
        with pytest.raises(KeyboardInterrupt):
            tools.execute_tool_call(
                tool_call("run_command", cmd="x", timeout=None, cwd=None)
            )
