import json
from typing import Any

import pytest

from conftest import OMITTED_TOOL_OUTPUT, SUMMARY_MARKER
from minime.context import TRIM_TARGET, Context, json_bytes

NO_LIMIT = 10**9


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
    assert Context(history).trim(NO_LIMIT)[1:] == (False, False)
    assert json.dumps(history) == before


def test_empty_history() -> None:
    history: list[Any] = []
    assert Context(history).trim(NO_LIMIT)[1:] == (False, False)
    assert history == []


def test_oldest_turns_are_dropped_by_size() -> None:
    history = [user("one"), output("a", 4000), user("two"), output("b", 4000)]
    assert Context(history).trim(6000)[1:] == (True, False)
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
    assert Context(history).trim(500)[1:] == (True, False)
    assert sizes(history) == ["three", 10]


def test_trimming_cuts_back_to_three_quarters_of_the_limit() -> None:
    """Dropping just enough to fit would trim again on the very next request,
    and every trim discards the provider's prompt cache."""
    history = [item for n in range(10) for item in (user(f"t{n}"), output("o", 900))]
    limit = json_bytes(history) - 1  # one byte over
    assert Context(history).trim(limit)[1:] == (True, False)

    assert json_bytes(history) <= limit * TRIM_TARGET
    assert sizes(history)[0] == "t3"  # three turns dropped, not one
    # No more is dropped than the 75% mark requires.
    one_more = [user("t2"), output("o", 900), *history]
    assert json_bytes(one_more) > limit * TRIM_TARGET


def test_history_between_the_mark_and_the_limit_is_left_alone() -> None:
    history = [user("one"), output("a", 4000), user("two"), output("b", 4000)]
    size = json_bytes(history)
    assert size > (size + 100) * TRIM_TARGET
    assert Context(history).trim(size + 100)[1:] == (False, False)
    assert sizes(history) == ["one", 4000, "two", 4000]


def test_room_is_left_for_later_requests_after_a_trim() -> None:
    history = [item for n in range(10) for item in (user(f"t{n}"), output("o", 900))]
    limit = json_bytes(history) - 1
    Context(history).trim(limit)
    # Two more turns can be appended before anything is trimmed again.
    for n in (10, 11):
        history += [user(f"t{n}"), output("o", 900)]
        assert Context(history).trim(limit)[1:] == (False, False)


def test_items_before_the_first_user_turn_go_with_it() -> None:
    history = [output("orphan", 3000), user("one"), user("two")]
    dropped, _, _ = Context(history).trim(200)
    assert sizes(dropped) == [3000, "one"]
    assert sizes(history) == ["two"]


def test_current_turn_sheds_its_oldest_tool_outputs() -> None:
    history = [user("go"), output("a", 3000), output("b", 3000), output("c", 3000)]
    assert Context(history).trim(8800)[1:] == (True, False)
    assert sizes(history) == ["go", "OMITTED", 3000, 3000]


def test_tool_outputs_are_also_cut_back_to_the_mark() -> None:
    history = [user("go"), *(output(f"o{n}", 1000) for n in range(10))]
    limit = json_bytes(history) - 1
    assert Context(history).trim(limit)[1:] == (True, False)
    assert json_bytes(history) <= limit * TRIM_TARGET
    assert sizes(history)[1:4] == ["OMITTED"] * 3
    assert sizes(history)[-1] == 1000


def test_request_above_the_mark_but_within_the_limit_is_kept() -> None:
    """The mark applies to what can be dropped; the request itself only has to fit."""
    history = [user("p" * 4500), output("a", 1000)]
    assert Context(history).trim(5000)[1:] == (True, False)
    assert sizes(history) == ["p" * 4500, "OMITTED"]


def test_older_turns_go_before_current_tool_outputs() -> None:
    history = [user("old"), output("a", 4000), user("new"), output("b", 4000)]
    Context(history).trim(6000)
    assert sizes(history) == ["new", 4000]


def test_all_tool_outputs_can_be_omitted() -> None:
    history = [user("go"), output("a", 3000), output("b", 3000)]
    assert Context(history).trim(600)[1:] == (True, False)
    assert sizes(history) == ["go", "OMITTED", "OMITTED"]


def test_oversized_request_empties_the_history() -> None:
    history = [user("old"), user("y" * 5000), output("a", 10)]
    assert Context(history).trim(1000) == ([], True, True)
    assert history == []


def test_history_without_a_user_turn_cannot_be_trimmed() -> None:
    history = [{"type": "message", "text": "z" * 500}]
    assert Context(history).trim(10)[1:] == (False, False)
    assert len(history) == 1


def test_size_is_measured_in_utf8_bytes() -> None:
    history = [user("old"), user("é" * 120)]  # 120 characters, 240 bytes
    assert Context(history).trim(300)[1:] == (True, False)
    assert sizes(history) == ["é" * 120]


def test_non_dict_items_are_counted() -> None:
    class Item:
        def __str__(self) -> str:
            return "i" * 5000

    history = [user("old"), Item(), user("new")]
    assert Context(history).trim(1000)[1:] == (True, False)
    assert history == [user("new")]


def test_trimming_mutates_the_same_list() -> None:
    history = [user("x" * 3000), user("two")]
    alias = history
    Context(history).trim(500)
    assert alias is history
    assert alias == [user("two")]


def test_limit_is_inclusive() -> None:
    history = [user("old"), user("new")]
    size = json_bytes(history)
    assert Context(history).trim(size)[1:] == (False, False)
    assert Context(history).trim(size - 1)[1:] == (True, False)
    assert history == [user("new")]


def test_many_long_turns_are_kept_when_they_fit() -> None:
    """There is no cap on the number of turns, only on their total size."""
    history = [user(f"turn {i}") for i in range(200)]
    assert Context(history).trim(NO_LIMIT)[1:] == (False, False)
    assert len(history) == 200


MARKER = "[Summary of the earlier conversation, which was removed to save space]"


def test_summary_goes_first_with_its_marker() -> None:
    context = Context([user("now")])
    context.add_summary("  Earlier: the user asked for X.\n", NO_LIMIT)
    assert context.history == [
        user(f"{MARKER}\nEarlier: the user asked for X."),
        user("now"),
    ]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_summary_is_left_out(text: str) -> None:
    context = Context([user("now")])
    context.add_summary(text, NO_LIMIT)
    assert context.history == [user("now")]


def test_summary_that_would_not_fit_is_left_out() -> None:
    context = Context([user("now")])
    room = json_bytes([user(f"{MARKER}\nshort"), user("now")])
    context.add_summary("short", room)
    assert len(context.history) == 2
    context = Context([user("now")])
    context.add_summary("short", room - 1)
    assert context.history == [user("now")]


def test_summary_is_trimmed_like_any_turn() -> None:
    context = Context([user("now")])
    context.add_summary("s" * 3000, NO_LIMIT)
    context.history.append(user("next"))
    assert context.trim(500)[1:] == (True, False)
    assert context.history == [user("now"), user("next")]


def test_trim_returns_the_dropped_turns_in_order() -> None:
    history = [user("one"), output("a", 3000), user("two"), user("three")]
    context = Context(list(history))
    dropped, _, _ = context.trim(500)
    assert dropped == history[:2]
    assert context.history == [user("two"), user("three")]


def test_byte_budget_converts_tokens_and_leaves_out_the_overhead() -> None:
    assert Context().byte_budget(1000, 300) == 4000 - 300  # 4 bytes per token
    assert Context(tokens_per_byte=0.5).byte_budget(1000, 300) == 2000 - 300


def test_learned_ratio_replaces_the_default() -> None:
    context = Context()
    context.learn_tokens_per_byte(500, 1000)
    assert context.tokens_per_byte == 0.5
    assert context.byte_budget(1000, 0) == 2000


@pytest.mark.parametrize(("input_tokens", "expected"), [(1, 0.1), (10**7, 1.0)])
def test_implausible_ratios_are_bounded(input_tokens: int, expected: float) -> None:
    context = Context()
    context.learn_tokens_per_byte(input_tokens, 1000)
    assert context.tokens_per_byte == expected


def test_undo_drops_the_last_turn_only() -> None:
    history = [user("one"), output("a", 10), user("two"), output("b", 10)]
    context = Context(list(history))
    assert context.undo()
    assert context.history == history[:2]
    assert context.undo()
    assert context.history == []
    assert not context.undo()


def test_turns_counts_user_messages() -> None:
    assert Context([user("one"), output("a", 10), user("two")]).turns() == 2
    assert Context().turns() == 0


def test_replace_with_summary() -> None:
    context = Context([user("one"), output("a", 10)])
    assert context.replace_with_summary("short", NO_LIMIT)
    assert context.history == [user(f"{SUMMARY_MARKER}\nshort")]


@pytest.mark.parametrize(("summary", "limit"), [("  ", NO_LIMIT), ("s" * 500, 100)])
def test_replace_with_an_unusable_summary_keeps_the_history(
    summary: str, limit: int
) -> None:
    history = [user("one"), output("a", 10)]
    context = Context(list(history))
    assert not context.replace_with_summary(summary, limit)
    assert context.history == history
