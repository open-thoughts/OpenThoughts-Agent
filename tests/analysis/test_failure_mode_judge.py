import pytest

from scripts.analysis.failure_mode_judge import (
    batched,
    request_json_array,
    strip_json_code_fence,
)


class _FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self.payload})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, payload):
        self.completions = _FakeCompletions(payload)
        self.chat = type("Chat", (), {"completions": self.completions})()


class _FlakyCompletions(_FakeCompletions):
    def __init__(self, payload):
        super().__init__(payload)
        self.failures_remaining = 1

    def create(self, **kwargs):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary failure")
        return super().create(**kwargs)


class _FlakyClient:
    def __init__(self, payload):
        self.completions = _FlakyCompletions(payload)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_shared_judge_strips_fences_and_forwards_output_limit():
    client = _FakeClient('```json\n[{"trial_name": "one", "analysis": "ok"}]\n```')

    result = request_json_array(
        client,
        model="judge",
        temperature=0.0,
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=100,
    )

    assert result == [{"trial_name": "one", "analysis": "ok"}]
    assert client.completions.calls[0]["max_completion_tokens"] == 100


def test_shared_judge_batches_and_rejects_non_array_payloads():
    assert list(batched([1, 2, 3], 2)) == [[1, 2], [3]]
    assert strip_json_code_fence("```\n[]\n```") == "[]"

    with pytest.raises(ValueError, match="JSON array"):
        request_json_array(
            _FakeClient('{"not": "an array"}'),
            model="judge",
            temperature=0.0,
            system_prompt="system",
            user_prompt="user",
        )


def test_shared_judge_retries_transient_client_errors():
    client = _FlakyClient("[]")
    retries = []

    assert (
        request_json_array(
            client,
            model="judge",
            temperature=0.0,
            system_prompt="system",
            user_prompt="user",
            max_attempts=2,
            on_retry=lambda attempt, exc: retries.append((attempt, str(exc))),
        )
        == []
    )
    assert retries == [(1, "temporary failure")]
    assert len(client.completions.calls) == 1
