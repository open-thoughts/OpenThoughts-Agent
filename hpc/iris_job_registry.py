"""SQLite registry of submitted iris jobs + their GCS output destinations.

The OT-Agent iris launcher writes one row here per submission. The fetch
daemon (``hpc.iris_fetch_daemon``, planned) polls the iris controller for
each non-terminal row and, on terminal state, copies
``gcs_output_dir/`` down to ``local_dest/``. Decoupling the registry
from both ends means the launcher can submit while the daemon is
offline, and the daemon can be installed/uninstalled without losing
state.

Schema is intentionally tiny; add columns only when a real consumer
needs them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from hpc.local_paths import PATHS, ensure as ensure_local_paths


DB_PATH: Path = PATHS.state / "iris_jobs.db"


# Possible status values written by the launcher / daemon. Strings (not
# enums) so the SQLite column reads naturally in `sqlite3 ... .dump`.
STATUS_SUBMITTED = "submitted"     # launcher INSERT, before daemon sees it
STATUS_RUNNING = "running"         # daemon observed an active iris state
STATUS_SUCCEEDED = "succeeded"     # iris terminal SUCCEEDED; fetch may not be done
STATUS_FAILED = "failed"           # iris terminal FAILED/CANCELLED
STATUS_FETCHING = "fetching"       # daemon is currently `gcloud storage cp`-ing
STATUS_FETCHED = "fetched"         # outputs live at local_dest/
STATUS_FETCH_FAILED = "fetch_failed"  # gcloud cp errored; user can retry


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    job_name          TEXT NOT NULL,
    submitted_at      TEXT NOT NULL,        -- ISO-8601 UTC
    gcs_output_dir    TEXT NOT NULL,        -- gs://.../<job-name>
    local_dest        TEXT NOT NULL,        -- ~/.ot-agent/runs/<job-name>
    cluster_config    TEXT NOT NULL,
    status            TEXT NOT NULL,
    last_polled_at    TEXT,
    fetched_at        TEXT,
    exit_code         INTEGER,
    error_msg         TEXT,
    iris_attempt_id   INTEGER,
    bytes_fetched     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, last_polled_at);
"""


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    job_name: str
    submitted_at: str
    gcs_output_dir: str
    local_dest: str
    cluster_config: str
    status: str
    last_polled_at: Optional[str]
    fetched_at: Optional[str]
    exit_code: Optional[int]
    error_msg: Optional[str]
    iris_attempt_id: Optional[int]
    bytes_fetched: Optional[int]


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open (and on first use, create) the registry database.

    Uses ``isolation_level=None`` + ``BEGIN IMMEDIATE`` for explicit
    transaction control; SQLite's default deferred transactions are
    surprising under multi-writer (launcher + daemon) load.
    """
    ensure_local_paths(PATHS.home, PATHS.state)
    conn = sqlite3.connect(str(DB_PATH), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        job_name=row["job_name"],
        submitted_at=row["submitted_at"],
        gcs_output_dir=row["gcs_output_dir"],
        local_dest=row["local_dest"],
        cluster_config=row["cluster_config"],
        status=row["status"],
        last_polled_at=row["last_polled_at"],
        fetched_at=row["fetched_at"],
        exit_code=row["exit_code"],
        error_msg=row["error_msg"],
        iris_attempt_id=row["iris_attempt_id"],
        bytes_fetched=row["bytes_fetched"],
    )


def register_submission(
    *,
    job_id: str,
    job_name: str,
    submitted_at_iso: str,
    gcs_output_dir: str,
    local_dest: Path,
    cluster_config: str,
) -> None:
    """Insert a new row for a freshly submitted iris job.

    Idempotent on ``job_id`` — re-submission with the same ID overwrites
    the row (the iris job IDs are timestamp-suffixed so collisions
    indicate a deliberate re-register, not a duplicate insert).
    """
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_name, submitted_at, gcs_output_dir, local_dest,
                cluster_config, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                job_name=excluded.job_name,
                submitted_at=excluded.submitted_at,
                gcs_output_dir=excluded.gcs_output_dir,
                local_dest=excluded.local_dest,
                cluster_config=excluded.cluster_config,
                status=excluded.status
            """,
            (job_id, job_name, submitted_at_iso, gcs_output_dir, str(local_dest),
             cluster_config, STATUS_SUBMITTED),
        )
        conn.execute("COMMIT")


def list_pending() -> list[JobRecord]:
    """Return jobs the daemon should keep polling."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?, ?) ORDER BY submitted_at",
            (STATUS_SUBMITTED, STATUS_RUNNING, STATUS_FETCHING),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def list_all(limit: Optional[int] = None) -> list[JobRecord]:
    """Return every registered job, most-recent first. For status views."""
    sql = "SELECT * FROM jobs ORDER BY submitted_at DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with _connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [_row_to_record(r) for r in rows]


def update_status(
    job_id: str,
    *,
    status: str,
    last_polled_at_iso: Optional[str] = None,
    fetched_at_iso: Optional[str] = None,
    exit_code: Optional[int] = None,
    error_msg: Optional[str] = None,
    iris_attempt_id: Optional[int] = None,
    bytes_fetched: Optional[int] = None,
) -> None:
    """Patch only the columns the caller passes (None = leave alone)."""
    updates: list[tuple[str, object]] = [("status", status)]
    if last_polled_at_iso is not None:
        updates.append(("last_polled_at", last_polled_at_iso))
    if fetched_at_iso is not None:
        updates.append(("fetched_at", fetched_at_iso))
    if exit_code is not None:
        updates.append(("exit_code", exit_code))
    if error_msg is not None:
        updates.append(("error_msg", error_msg))
    if iris_attempt_id is not None:
        updates.append(("iris_attempt_id", iris_attempt_id))
    if bytes_fetched is not None:
        updates.append(("bytes_fetched", bytes_fetched))

    set_clause = ", ".join(f"{col} = ?" for col, _ in updates)
    values: list[object] = [v for _, v in updates]
    values.append(job_id)

    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
        conn.execute("COMMIT")


def get(job_id: str) -> Optional[JobRecord]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_record(row) if row is not None else None


def get_latest_by_job_name(job_name: str) -> Optional[JobRecord]:
    """Return the most recently submitted record for ``job_name``.

    Multiple iris job_ids can share a job_name (e.g. when the same
    short name is launched twice). Returning the latest matches the
    intent of ``--resume-from <job-name>``: pick up where the last
    attempt left off.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_name = ? ORDER BY submitted_at DESC LIMIT 1",
            (job_name,),
        ).fetchone()
    return _row_to_record(row) if row is not None else None
