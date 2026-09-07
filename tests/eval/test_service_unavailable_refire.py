from database.unified_db.infra_errors import (
    INFRA_ERROR_TYPES,
    compute_infra_error_stats,
)
from eval.cloud.launch_eval_iris import DEFAULT_REFIRE_ERROR_TYPES


def test_infrastructure_failures_are_refired() -> None:
    for error_type in (
        "AgentKilledBySignalError",
        "DaytonaConflictError",
        "DaytonaSandboxStopError",
        "ServiceUnavailableError",
    ):
        assert error_type in INFRA_ERROR_TYPES
        assert error_type in DEFAULT_REFIRE_ERROR_TYPES


def test_infrastructure_failures_are_counted_as_infrastructure() -> None:
    stats = {
        "evals": {
            "reward": {
                "exception_stats": {
                    "AgentKilledBySignalError": ["trial-d"],
                    "DaytonaConflictError": ["trial-g"],
                    "DaytonaSandboxStopError": ["trial-e", "trial-f"],
                    "ServiceUnavailableError": ["trial-a", "trial-b"],
                    "AgentTimeoutError": ["trial-c"],
                }
            }
        }
    }

    assert compute_infra_error_stats(stats) == (
        6,
        {
            "AgentKilledBySignalError": 1,
            "DaytonaConflictError": 1,
            "DaytonaSandboxStopError": 2,
            "ServiceUnavailableError": 2,
        },
    )
