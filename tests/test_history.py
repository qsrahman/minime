import json
from typing import Any

from conftest import OMITTED_TOOL_OUTPUT

import main

NO_LIMIT = 10**9
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
        elif item["output"] == OMITTED_TOOL_OUTPUT:
            summary.append("OMITTED")
        else:
            summary.append(len(item["output"]))
    return summary


def test_history_within_limits_is_untouched() -> None:
    history = [user("one"), output("a", 100), user("two")]
    before = json.dumps(history)
    assert main.Conversation(history).trim(NO_LIMIT) == (False, False)
    assert json.dumps(history) == before


def test_empty_history() -> None:
    history: list[Any] = []
    assert main.Conversation(history).trim(NO_LIMIT) == (False, False)
    assert history == []


def test_oldest_turns_are_dropped_by_size() -> None:
    history = [user("one"), output("a", 4000), user("two"), output("b", 4000)]
    assert main.Conversation(history).trim(6000) == (True, False)
    assert sizes(history) == ["two", 4000]


def test_several_turns_can_be_dropped_at_once() -> None:
    history = [
        user("one"),
        output("a", 3000),
        user("two"),
        output("b", 3000),
        user("three"),
        output("c", 10),
    ]
    assert main.Conversation(history).trim(500) == (True, False)
    assert sizes(history) == ["three", 10]


def test_trimming_cuts_back_to_three_quarters_of_the_limit() -> None:
    """Dropping just enough to fit would trim again on the very next request,
    and every trim discards the provider's prompt cache."""
    history = [item for n in range(10) for item in (user(f"t{n}"), output("o", 900))]
    limit = main.json_bytes(history) - 1  # one byte over
    assert main.Conversation(history).trim(limit) == (True, False)

    assert main.json_bytes(history) <= limit * main.TRIM_TARGET
    assert sizes(history)[0] == "t3"  # three turns dropped, not one
    # No more is dropped than the 75% mark requires.
    one_more = [user("t2"), output("o", 900), *history]
    assert main.json_bytes(one_more) > limit * main.TRIM_TARGET


def test_history_between_the_mark_and_the_limit_is_left_alone() -> None:
    history = [user("one"), output("a", 4000), user("two"), output("b", 4000)]
    size = main.json_bytes(history)
    assert size > (size + 100) * main.TRIM_TARGET
    assert main.Conversation(history).trim(size + 100) == (False, False)
    assert sizes(history) == ["one", 4000, "two", 4000]


def test_room_is_left_for_later_requests_after_a_trim() -> None:
    history = [item for n in range(10) for item in (user(f"t{n}"), output("o", 900))]
    limit = main.json_bytes(history) - 1
    main.Conversation(history).trim(limit)
    # Two more turns can be appended before anything is trimmed again.
    for n in (10, 11):
        history += [user(f"t{n}"), output("o", 900)]
        assert main.Conversation(history).trim(limit) == (False, False)


def test_items_before_the_first_user_turn_go_with_it() -> None:
    history = [output("orphan", 3000), user("one"), user("two")]
    main.Conversation(history).trim(200)
    assert sizes(history) == ["two"]


def test_current_turn_sheds_its_oldest_tool_outputs() -> None:
    history = [user("go"), output("a", 3000), output("b", 3000), output("c", 3000)]
    assert main.Conversation(history).trim(8800) == (True, False)
    assert sizes(history) == ["go", "OMITTED", 3000, 3000]


def test_tool_outputs_are_also_cut_back_to_the_mark() -> None:
    history = [user("go"), *(output(f"o{n}", 1000) for n in range(10))]
    limit = main.json_bytes(history) - 1
    assert main.Conversation(history).trim(limit) == (True, False)
    assert main.json_bytes(history) <= limit * main.TRIM_TARGET
    assert sizes(history)[1:4] == ["OMITTED"] * 3
    assert sizes(history)[-1] == 1000


def test_request_above_the_mark_but_within_the_limit_is_kept() -> None:
    """The mark applies to what can be dropped; the request itself only has to fit."""
    history = [user("p" * 4500), output("a", 1000)]
    assert main.Conversation(history).trim(5000) == (True, False)
    assert sizes(history) == ["p" * 4500, "OMITTED"]


def test_older_turns_go_before_current_tool_outputs() -> None:
    history = [user("old"), output("a", 4000), user("new"), output("b", 4000)]
    main.Conversation(history).trim(6000)
    assert sizes(history) == ["new", 4000]


def test_all_tool_outputs_can_be_omitted() -> None:
    history = [user("go"), output("a", 3000), output("b", 3000)]
    assert main.Conversation(history).trim(600) == (True, False)
    assert sizes(history) == ["go", "OMITTED", "OMITTED"]


def test_oversized_request_is_replaced_by_a_placeholder() -> None:
    history = [user("y" * 5000), output("a", 10)]
    assert main.Conversation(history).trim(1000) == (True, True)
    assert history == [user(PLACEHOLDER)]


def test_history_is_emptied_when_even_the_placeholder_does_not_fit() -> None:
    history = [user("y" * 5000)]
    assert main.Conversation(history).trim(10) == (True, True)
    assert history == []


def test_history_without_a_user_turn_cannot_be_trimmed() -> None:
    history = [{"type": "message", "text": "z" * 500}]
    assert main.Conversation(history).trim(10) == (False, False)
    assert len(history) == 1


def test_size_is_measured_in_utf8_bytes() -> None:
    history = [user("old"), user("é" * 120)]  # 120 characters, 240 bytes
    assert main.Conversation(history).trim(300) == (True, False)
    assert sizes(history) == ["é" * 120]


def test_non_dict_items_are_counted() -> None:
    class Item:
        def __str__(self) -> str:
            return "i" * 5000

    history = [user("old"), Item(), user("new")]
    assert main.Conversation(history).trim(1000) == (True, False)
    assert history == [user("new")]


def test_trimming_mutates_the_same_list() -> None:
    history = [user("x" * 3000), user("two")]
    alias = history
    main.Conversation(history).trim(500)
    assert alias is history
    assert alias == [user("two")]


def test_limit_is_inclusive() -> None:
    history = [user("old"), user("new")]
    size = main.json_bytes(history)
    assert main.Conversation(history).trim(size) == (False, False)
    assert main.Conversation(history).trim(size - 1) == (True, False)
    assert history == [user("new")]


def test_many_long_turns_are_kept_when_they_fit() -> None:
    """There is no cap on the number of turns, only on their total size."""
    history = [user(f"turn {i}") for i in range(200)]
    assert main.Conversation(history).trim(NO_LIMIT) == (False, False)
    assert len(history) == 200
