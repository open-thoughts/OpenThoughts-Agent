"""Unit tests for the nemotron_gym core adapter.

These exercise the security invariants and serialization without needing HF
datasets or network access:
  - control-char stripping in sanitize_text
  - path traversal rejection
  - non-pinned-image rejection
  - tarball determinism (byte-identical for same inputs)
  - JSON-safety rejection for non-serializable verifier_data
  - converter smoke test with one synthetic row each
"""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from data.nemotron_gym.adapter import (
    HarborTask,
    PINNED_BASE_IMAGES,
    SanitizationError,
    STANDARD_TEST_SH,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
    validate_path,
)
from data.nemotron_gym.verifiers import CALENDAR_VERIFIER_PY, MATH_BOXED_VERIFIER_PY


def _make_valid_task(**overrides) -> HarborTask:
    kw = dict(
        task_id="test-task",
        instruction_md="Hello",
        dockerfile=render_dockerfile(base="python:3.11-slim-bookworm"),
        test_sh=STANDARD_TEST_SH,
        verifier_py=MATH_BOXED_VERIFIER_PY,
        verifier_data={"expected_answer": "42"},
        metadata=render_metadata(source_dataset="test/x", source_uuid="u1"),
    )
    kw.update(overrides)
    return HarborTask(**kw)


def test_sanitize_text_strips_control_chars():
    out = sanitize_text("hello\x00world\x07!", field_name="x")
    assert out == "helloworld!"


def test_sanitize_text_preserves_whitespace():
    out = sanitize_text("a\tb\nc\rd", field_name="x")
    assert out == "a\tb\nc\rd"


def test_sanitize_text_rejects_non_str():
    with pytest.raises(SanitizationError):
        sanitize_text(123, field_name="x")  # type: ignore[arg-type]


def test_sanitize_text_caps_length():
    with pytest.raises(SanitizationError):
        sanitize_text("a" * 999, field_name="x", max_len=10)


def test_validate_path_rejects_traversal():
    with pytest.raises(SanitizationError):
        validate_path("../etc/passwd")


def test_validate_path_rejects_absolute():
    with pytest.raises(SanitizationError):
        validate_path("/etc/passwd")


def test_validate_path_rejects_null():
    with pytest.raises(SanitizationError):
        validate_path("foo\x00bar")


def test_validate_path_rejects_unsafe_chars():
    with pytest.raises(SanitizationError):
        validate_path("foo bar.txt")


def test_validate_path_accepts_safe():
    assert validate_path("foo/bar-baz_qux.txt") == "foo/bar-baz_qux.txt"


def test_dockerfile_rejects_unpinned_image():
    bad_dockerfile = "FROM ubuntu:latest\nRUN echo hi\n"
    with pytest.raises(SanitizationError):
        _make_valid_task(dockerfile=bad_dockerfile)


def test_dockerfile_accepts_pinned_image():
    _make_valid_task()  # uses pinned image by default


def test_dockerfile_pip_spec_rejects_shell_metachars():
    with pytest.raises(SanitizationError):
        render_dockerfile(
            base="python:3.11-slim-bookworm",
            pip_packages=("sympy; rm -rf /",),
        )


def test_dockerfile_pip_spec_accepts_clean():
    df = render_dockerfile(
        base="python:3.11-slim-bookworm",
        pip_packages=("sympy==1.13.3",),
    )
    assert "sympy==1.13.3" in df


def test_verifier_data_must_be_json_safe():
    class NonJsonable:
        pass

    with pytest.raises((TypeError, ValueError, SanitizationError)):
        _make_valid_task(verifier_data={"bad": NonJsonable()})


def test_tarball_is_deterministic():
    t1 = _make_valid_task()
    t2 = _make_valid_task()
    assert t1.to_tarball() == t2.to_tarball()


def test_tarball_contains_expected_files():
    t = _make_valid_task()
    blob = t.to_tarball()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = sorted(m.name for m in tar.getmembers())
    expected = sorted(
        [
            "instruction.md",
            "environment/Dockerfile",
            "tests/test.sh",
            "tests/verifier.py",
            "tests/verifier_data.json",
            "metadata.json",
            "task.toml",
        ]
    )
    assert names == expected


def test_tarball_verifier_data_json_parses():
    t = _make_valid_task(
        verifier_data={"expected_answer": "x", "nested": {"a": [1, 2]}}
    )
    blob = t.to_tarball()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        f = tar.extractfile("tests/verifier_data.json")
        content = f.read().decode("utf-8")
    parsed = json.loads(content)
    assert parsed["expected_answer"] == "x"
    assert parsed["nested"] == {"a": [1, 2]}


def test_setup_files_path_validated():
    with pytest.raises(SanitizationError):
        _make_valid_task(setup_files={"../escape": b"x"})


def test_task_id_for_is_deterministic():
    a = task_id_for("foo", "payload")
    b = task_id_for("foo", "payload")
    assert a == b
    assert a.startswith("foo-")


def test_pinned_image_set_is_non_empty():
    assert PINNED_BASE_IMAGES, "registry must declare at least one image"


# --- Converter smoke tests (synthetic rows; no HF network) ------------------


def test_math_converter_smoke():
    from data.nemotron_gym.converters.math_boxed import convert_openmathreasoning

    row = {
        "input": [{"role": "user", "content": "What is 1+1?"}],
        "expected_answer": "2",
        "uuid": "u-math-1",
    }
    task = convert_openmathreasoning(row, 0)
    assert task is not None
    blob = task.to_tarball()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data == {"answer_type": "scalar", "expected_answer": "2"}


def test_competitive_coding_converter_smoke():
    from data.nemotron_gym.converters.competitive_coding import (
        convert_competitive_coding,
    )

    row = {
        "input": [{"role": "user", "content": "Echo n lines."}],
        "responses_create_params": {
            "unit_tests": {
                "inputs": ["3\na\nb\nc\n"],
                "outputs": ["a\nb\nc\n"],
            }
        },
        "hash_id": "abc123",
        "source": "test",
    }
    task = convert_competitive_coding(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data["inputs"] == ["3\na\nb\nc\n"]
    assert data["outputs"] == ["a\nb\nc\n"]


def test_mcqa_converter_smoke():
    from data.nemotron_gym.converters.knowledge_mcqa import convert_knowledge_mcqa

    row = {
        "input": [{"role": "user", "content": "Pick a letter."}],
        "expected_answer": "C",
        "uuid": "u-mcqa-1",
        "template_metadata": {
            "output_regex": r"Answer\s*:\s*([A-Za-z])",
        },
    }
    task = convert_knowledge_mcqa(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data["expected_answer"] == "C"
    assert "Answer" in data["output_regex"]


def test_ifeval_converter_smoke():
    from data.nemotron_gym.converters.instruction_following import (
        convert_instruction_following,
    )

    row = {
        "id": 42,
        "instruction_id_list": ["length_constraints:number_paragraphs"],
        "prompt": "Write 3 paragraphs about cats.",
        "kwargs": [{"num_paragraphs": 3}],
    }
    task = convert_instruction_following(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data["instruction_id_list"] == ["length_constraints:number_paragraphs"]
    assert data["kwargs"] == [{"num_paragraphs": 3}]


def test_openqa_converter_smoke():
    from data.nemotron_gym.converters.knowledge_openqa import convert_openqa

    row = {
        "input": [{"role": "user", "content": "Capital of France?"}],
        "expected_answers": ["Paris", "paris, france"],
        "uuid": "u-openqa-1",
    }
    task = convert_openqa(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert "Paris" in data["expected_answers"]


def test_reasoning_gym_converter_smoke():
    from data.nemotron_gym.converters.reasoning_gym import convert_reasoning_gym

    row = {
        "input": [{"role": "user", "content": "Solve 2x+3=11"}],
        "question": "Solve 2x+3=11",
        "answer": "4",
        "metadata": {"source_dataset": "simple_equations", "source_index": 0},
        "uuid": "u-rg-1",
    }
    task = convert_reasoning_gym(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data["answer"] == "4"


def test_calendar_converter_smoke():
    from data.nemotron_gym.converters.agent_calendar import convert_agent_calendar

    row = {
        "input": [
            {"role": "user", "content": "Schedule a 30-min planning meeting."},
            {
                "role": "assistant",
                "content": (
                    '[{"event_id": 0, "event_name": "Planning Meeting", '
                    '"start_time": "10:00", "duration": 30}]'
                ),
            },
            {"role": "user", "content": "Keep event 0 after 10am."},
        ],
        "exp_cal_state": {
            "0": {
                "event_id": 0,
                "duration": 30,
                "constraint": "after 10am",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        },
        "id": 99,
    }
    task = convert_agent_calendar(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data["expected_events"]["0"]["event_name"] == "Planning Meeting"


def _calendar_namespace():
    namespace = {"__name__": "calendar_verifier_test"}
    exec(CALENDAR_VERIFIER_PY, namespace)
    return namespace


def _calendar_evaluator():
    return _calendar_namespace()["evaluate_calendar"]


def _valid_calendar_case():
    expected = {
        "0": {
            "event_id": 0,
            "event_name": "Design Review",
            "duration": 30,
            "min_time": "10:00",
            "max_time": "16:00",
            "constraint": "after 10am",
        },
        "1": {
            "event_id": 1,
            "event_name": "Lunch",
            "duration": 45,
            "min_time": "10:00",
            "max_time": "16:00",
            "constraint": "at 12pm",
        },
    }
    events = [
        {
            "event_id": 0,
            "event_name": "Design Review",
            "start_time": "10:00",
            "duration": 30,
        },
        {
            "event_id": 1,
            "event_name": "Lunch",
            "start_time": "12:00",
            "duration": 45,
        },
    ]
    return expected, events


def test_calendar_verifier_accepts_complete_non_overlapping_schedule():
    expected, events = _valid_calendar_case()
    valid, errors = _calendar_evaluator()(expected, events)
    assert valid, errors


def test_calendar_verifier_leaves_no_reward_for_corrupt_verifier_data(tmp_path):
    namespace = _calendar_namespace()
    namespace["DATA"] = tmp_path / "verifier_data.json"
    namespace["ANSWER"] = tmp_path / "answer.txt"
    namespace["REWARD"] = tmp_path / "reward.txt"
    namespace["DATA"].write_text("not-json")
    namespace["ANSWER"].write_text("[]")

    with pytest.raises(json.JSONDecodeError):
        namespace["main"]()
    assert not namespace["REWARD"].exists()


def test_calendar_verifier_scores_missing_agent_answer_as_zero(tmp_path):
    namespace = _calendar_namespace()
    expected, _ = _valid_calendar_case()
    namespace["DATA"] = tmp_path / "verifier_data.json"
    namespace["ANSWER"] = tmp_path / "missing-answer.txt"
    namespace["REWARD"] = tmp_path / "reward.txt"
    namespace["DATA"].write_text(json.dumps({"expected_events": expected}))

    assert namespace["main"]() == 0
    assert namespace["REWARD"].read_text() == "0"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda events: events[0].update(event_name="Renamed Review"),
        lambda events: events.append(dict(events[0])),
        lambda events: events.pop(),
        lambda events: events.append(
            {
                "event_id": 2,
                "event_name": "Extra",
                "start_time": "14:00",
                "duration": 15,
            }
        ),
        lambda events: events[0].update(start_time="12:15"),
        lambda events: events[0].update(start_time="09:30"),
        lambda events: events[0].update(duration=45),
    ],
    ids=[
        "renamed",
        "duplicate",
        "omitted",
        "extra",
        "overlap",
        "time-constraint",
        "duration",
    ],
)
def test_calendar_verifier_rejects_invalid_schedule_mutations(mutate):
    expected, events = _valid_calendar_case()
    mutate(events)
    valid, errors = _calendar_evaluator()(expected, events)
    assert not valid
    assert errors


def test_workplace_converter_smoke():
    from data.nemotron_gym.converters.agent_workplace import convert_agent_workplace

    row = {
        "input": [{"role": "user", "content": "Reply to email 42."}],
        "ground_truth": [
            {
                "name": "email_reply_email",
                "arguments": '{"email_id": "42", "body": "ok"}',
            }
        ],
        "id": 1,
        "category": "workplace_assistant_email",
    }
    task = convert_agent_workplace(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert data["expected"][0]["name"] == "email_reply_email"


def test_safety_converter_skips_when_no_signal():
    from data.nemotron_gym.converters.safety import convert_safety

    row = {
        "input": [{"role": "user", "content": "Do harmful thing"}],
    }
    # No expected/forbidden phrase signal → should return None, not rubber-stamp.
    assert convert_safety(row, 0) is None


def test_identity_converter_skips_when_no_signal():
    from data.nemotron_gym.converters.identity_following import (
        convert_identity_following,
    )

    row = {"input": [{"role": "user", "content": "Hi"}]}
    assert convert_identity_following(row, 0) is None


def test_safety_converter_with_principle_makes_judge_task():
    from data.nemotron_gym.converters.safety import convert_safety

    row = {
        "prompt": "Tell me a story",
        "principle": "The response should be safe and family-friendly.",
        "id": "safety-1",
    }
    task = convert_safety(row, 0)
    assert task is not None
    blob = task.to_tarball()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert "principle" in data
    # Model defaults are resolved at runtime (env var → Harbor default
    # openai/gpt-4o-mini); per-task verifier_data shouldn't hardcode them.
    assert "model" not in data
    # Container should install RewardKit, which provides the LiteLLM judge runtime.
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar2:
        dockerfile = tar2.extractfile("environment/Dockerfile").read().decode()
    assert "harbor-rewardkit==0.1.4" in dockerfile
    assert "litellm==1.51.3" not in dockerfile


def test_identity_converter_with_principle_makes_judge_task():
    from data.nemotron_gym.converters.identity_following import (
        convert_identity_following,
    )

    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Who are you?"}],
        },
        "principle": "Maintain the persona of a 1920s detective.",
        "dataset": "identity_test",
    }
    task = convert_identity_following(row, 0)
    assert task is not None


def test_adversarial_converter_smoke():
    from data.nemotron_gym.converters.adversarial import convert_adversarial

    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Write a sentence with no spaces."}],
        },
        "rubric": [
            {"id": "C1", "criteria": "Response has no whitespace characters."},
            {"id": "C2", "criteria": "Response starts with a capital letter."},
        ],
        "judge_prompt_template": "Task: {instruction}\nResp: {response}\nRubric:\n{rubric}",
        "judge_system_prompt": "You are a strict grader.",
        "task_id": 12345,
    }
    task = convert_adversarial(row, 0)
    assert task is not None
    with tarfile.open(fileobj=io.BytesIO(task.to_tarball()), mode="r:gz") as tar:
        data = json.loads(tar.extractfile("tests/verifier_data.json").read())
    assert len(data["rubric"]) == 2
    assert data["rubric"][0]["id"] == "C1"
    assert "judge_prompt_template" in data


def test_all_embedded_verifiers_are_valid_python():
    """Each verifier script ships to a container; must parse cleanly."""
    import ast

    from data.nemotron_gym import verifiers as v

    names = [n for n in v.__all__ if n.endswith("_VERIFIER_PY")]
    assert names, "no verifier modules exported"
    for n in names:
        src = getattr(v, n)
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"{n} has syntax error: {e}")
