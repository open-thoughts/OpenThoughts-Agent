from __future__ import annotations

import pytest

from data.nemotron_gym.converters.math_boxed import answer_type
from data.nemotron_gym.verifiers.math_boxed import VERIFIER_PY


@pytest.fixture(scope="module")
def verifier() -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "math_boxed_test"}
    exec(compile(VERIFIER_PY, "generated_math_boxed_verifier.py", "exec"), namespace)
    return namespace


def test_interval_comparison_rejects_missing_excluded_point(verifier) -> None:
    matches = verifier["_matches"]
    expected = r"(-\infty,-4] \cup [-2,-1) \cup (-1,\infty)"
    actual = r"(-\infty,-4] \cup [-2,\infty)"

    assert not matches(actual, expected, "interval")


def test_typed_equivalences(verifier) -> None:
    matches = verifier["_matches"]

    assert matches("{2, 1}", "{1, 2}", "set")
    assert matches("{(-7, 8), (2, 3)}", "{(2, 3), (-7, 8)}", "set")
    assert matches("{O-, A+, x=1}", "{x=1, A+, O-}", "set")
    assert matches("(1/2, 2)", "(0.5, 2)", "tuple")
    assert matches("[1/2, 2]", "[0.5, 2]", "list")
    assert matches("2*x = 2", "x = 1", "equation")
    assert matches("1/2", "0.5", "scalar")
    assert matches("1/2", "50%", "scalar")
    assert matches("No solution", "no   solution.", "text")
    assert matches(r"\(u=x^2\).", r"\(u=x^2\)", "text")


def test_declared_type_fails_closed(verifier) -> None:
    matches = verifier["_matches"]
    unsupported = verifier["UnsupportedAnswer"]

    with pytest.raises(unsupported, match="invalid interval syntax"):
        matches("all real numbers", "(-oo, oo)", "interval")
    with pytest.raises(unsupported, match="unsupported answer_type"):
        matches("1", "2", "unknown")


@pytest.mark.parametrize(
    ("expected", "kind"),
    [
        (r"(-\infty,-4] \cup [-2,-1)", "interval"),
        ("[0, 1)", "interval"),
        ("{1, 2}", "set"),
        ("(1, 2)", "tuple"),
        ("[1, 2]", "list"),
        ("2*x = 2", "equation"),
        (r"z = (0, e^{\pi i / 3}, e^{2 \pi i / 3})", "text"),
        (r"2t = \frac{\pi}{4} + k\pi, k \in \mathbb{Z}", "text"),
        ("3/4", "scalar"),
        ("No solution", "text"),
    ],
)
def test_answer_type(expected: str, kind: str) -> None:
    assert answer_type(expected) == kind
