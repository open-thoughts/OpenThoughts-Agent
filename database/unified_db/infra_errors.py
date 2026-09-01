"""Single source of truth for infrastructure-error classification.

`INFRA_ERROR_TYPES` is the set of exception types that represent INFRASTRUCTURE
failures (Daytona/sandbox/environment/verification-wrapper) rather than genuine
agent/task failures. Harbor's resume filters retry these, and the eval listener's
disk-based resume scanner counts them.

This module is the canonical definition: both `eval/unified_eval_listener.py`
(the resume scanner) and `database/unified_db/utils.py` (the DB write-point that
persists the count onto `sandbox_jobs.stats`) import from here. Do NOT duplicate
the set elsewhere.

NOTE: harbor's trial finalizer no longer flattens an upstream infra failure into
`VerificationNotCompletedError` (VNC). Since the harbor `feuer/trial-not-scored-error`
change, when a single-step verification-enabled trial finalizes unscored:
  - an informative upstream error (e.g. `EnvironmentStartTimeoutError`) is PRESERVED
    as-is and bucketed under its true type (already in this set);
  - VNC now means strictly "the verifier was reached but produced no result";
  - a silent early fall-through with no prior error gets the catch-all
    `TrialNotScoredError` base.
All three (`EnvironmentStartTimeoutError`, `VerificationNotCompletedError`,
`TrialNotScoredError`) are infra and listed here. `exception_stats` keys off the
leaf class-name STRING, so the base needs its own entry. Old runs (pre-change)
that bucketed env-start timeouts as VNC keep the same total infra count; future
runs re-attribute to the true type for a truer breakdown.

This module is dependency-free (stdlib only) so it can be imported from anywhere.
"""

from typing import Any, Dict, Mapping, Tuple

# Infrastructure errors that harbor's resume filters will retry.
# MUST stay in sync with the sbatch resume branch's --filter-error-type list
# (eval/tacc/eval_harbor.sbatch).
INFRA_ERROR_TYPES = {
    "DaytonaError",
    "DaytonaAuthenticationError",
    "DaytonaAuthorizationError",
    "DaytonaNotFoundError",
    "EnvironmentStartTimeoutError",
    "DaytonaRateLimitError",
    "ServiceUnavailableError",
    "CancelledError",
    "SandboxBuildFailedError",
    "AgentEnvironmentTimeoutError",
    "VerificationNotCompletedError",
    "TrialNotScoredError",
    # Verification-wrapper for a sandbox/toolbox failure while uploading the
    # tests dir on the VERIFY path (harbor verifier.py wraps the underlying
    # Daytona 401 / httpx error as AddTestsDirError). This is the leaf that
    # surfaces after the harbor last-error-wins fix; a Daytona-side upload
    # failure is infra (verifier could not reach the sandbox), so it must be
    # re-run — NOT a genuine agent/task failure.
    "AddTestsDirError",
}


def compute_infra_error_stats(stats: Mapping[str, Any]) -> Tuple[int, Dict[str, int]]:
    """Compute the infrastructure-error count + per-type breakdown from a Harbor
    `stats` blob.

    Reads `stats.evals.<key>.exception_stats` (a `{error_type: [trial_ids]}` map),
    keeps only types in `INFRA_ERROR_TYPES`, and sums their counts. The count for a
    type is `len(ids)` when `ids` is a list, else 1 (defensive).

    Args:
        stats: The Harbor `stats` dict (i.e. `result["stats"]`).

    Returns:
        (n_infra_errors, infra_error_breakdown) where breakdown is
        `{error_type: count}` containing only infra types with a non-zero count.
        Types not present / not infra are omitted from the breakdown.
    """
    n_infra = 0
    breakdown: Dict[str, int] = {}
    if not isinstance(stats, Mapping):
        return 0, {}
    evals = stats.get("evals")
    if not isinstance(evals, Mapping):
        return 0, {}
    for eval_data in evals.values():
        if not isinstance(eval_data, Mapping):
            continue
        exception_stats = eval_data.get("exception_stats")
        if not isinstance(exception_stats, Mapping):
            continue
        for exc_type, ids in exception_stats.items():
            if exc_type not in INFRA_ERROR_TYPES:
                continue
            n = len(ids) if isinstance(ids, list) else 1
            n_infra += n
            breakdown[exc_type] = breakdown.get(exc_type, 0) + n
    return n_infra, breakdown


def filter_error_type_flags() -> str:
    """Render `INFRA_ERROR_TYPES` as repeatable harbor `--filter-error-type` flags.

    Returns a single space-joined string, deterministically sorted, e.g.:
        "--filter-error-type AgentEnvironmentTimeoutError --filter-error-type CancelledError ..."

    This is the canonical representation of the infra set as harbor `jobs resume`
    filter args, so the cluster sbatches can derive their resume filter from the
    same set the listener's resume manager imports — no hand-maintained subset that
    can drift. The set is non-empty by construction; this function never emits an
    empty string (callers rely on that — emitting zero flags would silently change
    harbor's resume scope to its `["CancelledError"]` default).
    """
    return " ".join(f"--filter-error-type {t}" for t in sorted(INFRA_ERROR_TYPES))


def _main(argv) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m database.unified_db.infra_errors",
        description=(
            "Emit the canonical INFRA_ERROR_TYPES set as harbor "
            "--filter-error-type resume filter flags."
        ),
    )
    parser.add_argument(
        "--filter-flags",
        action="store_true",
        help="Print the set as repeatable `--filter-error-type <T>` flags "
        "(deterministically sorted, space-joined, on one line).",
    )
    args = parser.parse_args(argv)

    if args.filter_flags:
        flags = filter_error_type_flags()
        # Defensive: the set is non-empty by construction, but never emit an empty
        # line — a caller that word-splits an empty string into zero flags would
        # silently change harbor's resume scope.
        if not flags.strip():
            print(
                "ERROR: INFRA_ERROR_TYPES is empty; refusing to emit zero filter flags.",
                file=sys.stderr,
            )
            return 2
        print(flags)
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
