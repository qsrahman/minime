import json
from typing import Any

import pytest

import config
import main

PLACEHOLDER = (
    "[Previous request omitted because its history exceeded the context budget.]"
)


def user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def output(call_id: str, size: int) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": "x" * size}


def sizes(history: list[Any]) -> list[Any]:
    """Each item as its user text, or its output size, or OMITTED."""
    summary: list[Any] = []
    for item in history:
        if "role" in item:
            summary.append(item["content"])
        elif item["output"] == main.OMITTED_TOOL_OUTPUT:
            summary.append("OMITTED")
        else:
            summary.append(len(item["output"]))
    return summary


def test_history_within_limits_is_untouched() -> None:
    history = [user("one"), output("a", 100), user("two")]
    before = json.dumps(history)
    assert main.trim_history(history) == (False, False)
    assert json.dumps(history) == before


def test_empty_history() -> None:
    history: list[Any] = []
    assert main.trim_history(history) == (False, False)
    assert history == []


def test_oldest_turns_are_dropped_by_count(settings: config.Config) -> None:
    settings.max_history_turns = 2
    history = [
        user("one"),
        output("a", 10),
        user("two"),
        output("b", 10),
        user("three"),
    ]
    assert main.trim_history(history) == (True, False)
    assert sizes(history) == ["two", 10, "three"]


def test_oldest_turns_are_dropped_by_size(settings: config.Config) -> None:
    settings.max_history_bytes = 5000
    history = [user("one"), output("a", 4000), user("two"), output("b", 4000)]
    assert main.trim_history(history) == (True, False)
    assert sizes(history) == ["two", 4000]


def test_several_turns_can_be_dropped_at_once(settings: config.Config) -> None:
    settings.max_history_turns = 1
    history = [user("one"), user("two"), user("three"), output("c", 10)]
    assert main.trim_history(history) == (True, False)
    assert sizes(history) == ["three", 10]


def test_items_before_the_first_user_turn_go_with_it(settings: config.Config) -> None:
    settings.max_history_turns = 1
    history = [output("orphan", 10), user("one"), user("two")]
    main.trim_history(history)
    assert sizes(history) == ["two"]


def test_current_turn_sheds_its_oldest_tool_outputs(settings: config.Config) -> None:
    settings.max_history_bytes = 7000
    history = [user("go"), output("a", 3000), output("b", 3000), output("c", 3000)]
    assert main.trim_history(history) == (True, False)
    assert sizes(history) == ["go", "OMITTED", 3000, 3000]


def test_older_turns_go_before_current_tool_outputs(settings: config.Config) -> None:
    settings.max_history_bytes = 5000
    history = [user("old"), output("a", 4000), user("new"), output("b", 4000)]
    main.trim_history(history)
    assert sizes(history) == ["new", 4000]


def test_all_tool_outputs_can_be_omitted(settings: config.Config) -> None:
    settings.max_history_bytes = 600
    history = [user("go"), output("a", 3000), output("b", 3000)]
    assert main.trim_history(history) == (True, False)
    assert sizes(history) == ["go", "OMITTED", "OMITTED"]


def test_oversized_request_is_replaced_by_a_placeholder(
    settings: config.Config,
) -> None:
    settings.max_history_bytes = 1000
    history = [user("y" * 5000), output("a", 10)]
    assert main.trim_history(history) == (True, True)
    assert history == [user(PLACEHOLDER)]


def test_history_is_emptied_when_even_the_placeholder_does_not_fit(
    settings: config.Config,
) -> None:
    settings.max_history_bytes = 10
    history = [user("y" * 5000)]
    assert main.trim_history(history) == (True, True)
    assert history == []


def test_history_without_a_user_turn_cannot_be_trimmed(settings: config.Config) -> None:
    settings.max_history_bytes = 10
    history = [{"type": "message", "text": "z" * 500}]
    assert main.trim_history(history) == (False, False)
    assert len(history) == 1


def test_size_is_measured_in_utf8_bytes(settings: config.Config) -> None:
    settings.max_history_bytes = 300
    history = [user("old"), user("é" * 120)]  # 120 characters, 240 bytes
    assert main.trim_history(history) == (True, False)
    assert sizes(history) == ["é" * 120]


def test_non_dict_items_are_counted(settings: config.Config) -> None:
    class Item:
        def __str__(self) -> str:
            return "i" * 5000

    settings.max_history_bytes = 1000
    history = [user("old"), Item(), user("new")]
    assert main.trim_history(history) == (True, False)
    assert history == [user("new")]


def test_trimming_mutates_the_same_list(settings: config.Config) -> None:
    settings.max_history_turns = 1
    history = [user("one"), user("two")]
    alias = history
    main.trim_history(history)
    assert alias is history
    assert alias == [user("two")]


@pytest.mark.parametrize("turns", [1, 3])
def test_turn_limit_boundary(settings: config.Config, turns: int) -> None:
    settings.max_history_turns = turns
    history = [user(str(i)) for i in range(turns)]
    assert main.trim_history(history) == (False, False)
    history.append(user("extra"))
    assert main.trim_history(history) == (True, False)
    assert len(history) == turns
