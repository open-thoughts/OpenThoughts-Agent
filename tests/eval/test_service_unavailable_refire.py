from database.unified_db.infra_errors import (
    INFRA_ERROR_TYPES,
    compute_infra_error_stats,
)
from eval.cloud.launch_eval_iris import DEFAULT_REFIRE_ERROR_TYPES


def test_api_and_signal_failures_are_refired() -> None:
    for error_type in ("AgentKilledBySignalError", "ServiceUnavailableError"):
        assert error_type in INFRA_ERROR_TYPES
        assert error_type in DEFAULT_REFIRE_ERROR_TYPES


def test_service_unavailable_is_counted_as_infrastructure() -> None:
    stats = {
        "evals": {
            "reward": {
                "exception_stats": {
                    "AgentKilledBySignalError": ["trial-d"],
                    "ServiceUnavailableError": ["trial-a", "trial-b"],
                    "AgentTimeoutError": ["trial-c"],
                }
            }
        }
    }

    assert compute_infra_error_stats(stats) == (
        3,
        {"AgentKilledBySignalError": 1, "ServiceUnavailableError": 2},
    )
