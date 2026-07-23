import pytest

from data.conversation_utils import extract_system_prompt, extract_user_prompt
from scripts.analysis.utils import (
    count_conversation_tokens,
    extract_chat_template_messages,
    extract_conversation_text,
    render_token_representation,
)


def test_extracts_role_and_from_conversation_variants():
    conversations = [
        {"from": "system", "value": "System rule"},
        {"role": "user", "content": "User request"},
    ]

    assert extract_system_prompt(conversations) == "System rule"
    assert extract_user_prompt(conversations) == "User request"


def test_preserves_empty_and_last_message_fallbacks():
    assert extract_user_prompt([]) == ""
    assert extract_system_prompt([]) is None
    assert (
        extract_user_prompt([{"role": "assistant", "content": "Fallback"}])
        == "Fallback"
    )


class _TokenCounter:
    def __init__(self):
        self.template_messages = None

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text)))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is False
        self.template_messages = messages
        return [10, 11, 12] if tokenize else "<chat-template>"


def test_token_representations_are_explicit_and_preserve_their_distinct_inputs():
    record = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi"},
        ]
    }
    tokenizer = _TokenCounter()

    assert render_token_representation(record, "serialized") == (
        '[{"from":"human","value":"Hello"},{"from":"gpt","value":"Hi"}]'
    )
    assert render_token_representation(record, "conversation_text") == "Hello\nHi"
    assert (
        render_token_representation(record, "chat_template", tokenizer=tokenizer)
        == "<chat-template>"
    )
    assert tokenizer.template_messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert count_conversation_tokens(record, tokenizer, "serialized") == 62
    assert count_conversation_tokens(record, tokenizer, "conversation_text") == 8
    assert count_conversation_tokens(record, tokenizer, "chat_template") == 3


def test_chat_template_rejects_non_message_records_and_unknown_representations():
    tokenizer = _TokenCounter()
    with pytest.raises(ValueError, match="messages.*conversations"):
        extract_chat_template_messages({"text": "not a message list"})
    with pytest.raises(ValueError, match="Unknown token representation"):
        count_conversation_tokens({"conversations": []}, tokenizer, "implicit")

    # Plain text retains its intentional JSON fallback for arbitrary records.
    assert extract_conversation_text({"metadata": "only"}) == '{"metadata": "only"}'
