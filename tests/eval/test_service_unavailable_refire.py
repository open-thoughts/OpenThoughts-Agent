from database.unified_db.infra_errors import (
    INFRA_ERROR_TYPES,
    compute_infra_error_stats,
)
from eval.cloud.launch_eval_iris import DEFAULT_REFIRE_ERROR_TYPES


def test_service_unavailable_is_refired() -> None:
    assert "ServiceUnavailableError" in INFRA_ERROR_TYPES
    assert "ServiceUnavailableError" in DEFAULT_REFIRE_ERROR_TYPES


def test_service_unavailable_is_counted_as_infrastructure() -> None:
    stats = {
        "evals": {
            "reward": {
                "exception_stats": {
                    "ServiceUnavailableError": ["trial-a", "trial-b"],
                    "AgentTimeoutError": ["trial-c"],
                }
            }
        }
    }

    assert compute_infra_error_stats(stats) == (
        2,
        {"ServiceUnavailableError": 2},
    )
