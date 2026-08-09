"""Format detection and the render path the backends actually consume.

A misdetected format doesn't error — it silently trains on the wrong text, so
these assert the detection table and the as_chat/to_texts/split output shapes.
"""

from __future__ import annotations

import pytest

from shadowlm.data import (CHAT, INSTRUCTION, PREFERENCE, RAW, SHAREGPT, TEXT,
                           Dataset, _detect_format, _instruction_cols,
                           _resolve_format)


# ---- detection ---------------------------------------------------------------

@pytest.mark.parametrize("row,expected", [
    ({"messages": [{"role": "user", "content": "hi"}]}, CHAT),
    ({"conversations": [{"from": "human", "value": "hi"}]}, SHAREGPT),
    ({"prompt": "p", "chosen": "a", "rejected": "b"}, PREFERENCE),
    ({"instruction": "i", "output": "o"}, INSTRUCTION),
    ({"question": "q", "answer": "a"}, INSTRUCTION),      # gsm8k naming
    ({"prompt": "p", "response": "r"}, INSTRUCTION),
    ({"text": "just text"}, TEXT),
    ({"foo": "bar"}, RAW),
])
def test_detection_table(row, expected):
    assert _detect_format([row]) == expected


def test_empty_dataset_is_raw():
    assert _detect_format([]) == RAW


def test_preference_wins_over_instruction_columns():
    # a preference row also carries a prompt column; it must not read as instruction
    assert _detect_format([{"prompt": "p", "chosen": "a", "rejected": "b",
                            "output": "o"}]) == PREFERENCE


def test_instruction_needs_both_a_prompt_and_a_response_column():
    assert _instruction_cols({"instruction"}) is None
    assert _instruction_cols({"output"}) is None
    assert _instruction_cols({"instruction", "output"}) == ("instruction", "output")


def test_explicit_format_overrides_detection():
    rows = [{"text": "x"}]
    assert _resolve_format(rows, "raw") == RAW
    assert _resolve_format(rows, "auto") == TEXT
    assert _resolve_format(rows, None) == TEXT


def test_unknown_format_override_lists_the_valid_ones():
    with pytest.raises(ValueError, match="expected one of"):
        _resolve_format([{"text": "x"}], "parquet")


# ---- as_chat -------------------------------------------------------------------

def test_sharegpt_speakers_become_chat_roles():
    ds = Dataset.from_list([{"conversations": [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "hello"}]}])
    assert ds.format == SHAREGPT
    assert ds.as_chat().rows[0]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}]


def test_instruction_becomes_one_exchange():
    ds = Dataset.from_list([{"instruction": "add", "output": "4"}])
    assert ds.as_chat().rows[0]["messages"] == [
        {"role": "user", "content": "add"},
        {"role": "assistant", "content": "4"}]


def test_alpaca_input_column_is_appended_to_the_prompt():
    ds = Dataset.from_list([{"instruction": "sum", "input": "2+2", "output": "4"}])
    user = ds.as_chat().rows[0]["messages"][0]["content"]
    assert "sum" in user and "2+2" in user


def test_chat_is_already_chat():
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": "hi"}]}])
    assert ds.as_chat() is ds


def test_raw_rows_cannot_be_chatified():
    ds = Dataset.from_list([{"foo": "bar"}])
    with pytest.raises(ValueError, match="cannot convert"):
        ds.as_chat()


# ---- to_texts / as_text ----------------------------------------------------------

def test_text_rows_render_verbatim():
    assert Dataset.from_list([{"text": "abc"}]).to_texts() == ["abc"]


def test_chat_rows_render_role_tagged():
    ds = Dataset.from_list([{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"}]}])
    assert ds.to_texts() == ["user: hi\nassistant: yo"]


def test_instruction_rows_render_alpaca_style():
    text = Dataset.from_list([{"instruction": "add", "output": "4"}]).to_texts()[0]
    assert "add" in text and "4" in text


def test_as_text_converts_and_relabels():
    ds = Dataset.from_list([{"instruction": "add", "output": "4"}]).as_text()
    assert ds.format == TEXT
    assert set(ds.rows[0]) == {"text"}


# ---- split ----------------------------------------------------------------------

def test_split_by_fraction_is_disjoint_and_complete():
    ds = Dataset.from_list([{"text": str(i)} for i in range(10)])
    train, ev = ds.split(test_size=0.2)
    assert (len(train), len(ev)) == (8, 2)
    assert {r["text"] for r in train.rows} | {r["text"] for r in ev.rows} \
        == {str(i) for i in range(10)}


def test_split_by_absolute_count():
    ds = Dataset.from_list([{"text": str(i)} for i in range(10)])
    assert len(ds.split(test_size=3)[1]) == 3


def test_split_is_reproducible_for_a_seed():
    ds = Dataset.from_list([{"text": str(i)} for i in range(20)])
    assert [r["text"] for r in ds.split(seed=7)[1]] \
        == [r["text"] for r in ds.split(seed=7)[1]]


def test_split_always_leaves_a_training_row():
    ds = Dataset.from_list([{"text": "a"}, {"text": "b"}])
    train, ev = ds.split(test_size=0.99)
    assert len(train) >= 1 and len(ev) >= 1
