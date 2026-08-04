# Debug log: Iris Harbor eval identity loss

## Goal

Make resumed eval trial counts and means reliable when Finelog delivers the early Harbor identity record after a watcher's incremental cursor has passed it.

## Initial status

- The watcher advances a local wall-clock cursor and appends only later Finelog records.
- Resumed eval aggregation depends on a one-time `starting Harbor job ... jobs_dir=...` record.
- A clean full-history read recovers all affected identities, while the incremental cache can permanently miss them.
- Missing identity silently falls back to lifecycle counts rendered as `N/?`, which can look authoritative.

## Hypotheses

1. A bounded overlap plus exact-line deduplication prevents records delivered slightly late from being lost.
2. An earliest-first full-history repair when identity is absent recovers records delivered outside that overlap.
3. Reusing identity from the durable bundle manifest prevents regressions on later watcher runs.
4. Explicit lifecycle-only trial and health labels prevent fallback counts from being mistaken for Harbor aggregates.

## Changes and results

- Added a five-minute overlap to incremental Finelog reads and deduplicate exact lines when merging into the local cache.
- Capture the next cursor before issuing the read, so records created during the request cannot fall between cursors.
- If a callable eval still lacks Harbor identity, reread its job tree from submission time; Iris does not expose a parent-task-only log endpoint.
- Persist the recovered identity and Finelog cursor in the bundle manifest, and reuse that identity on later watcher runs.
- Prefer a newly observed Finelog identity over the manifest, allowing a newer resume identity to replace stale persisted state.
- Label fallback trial counts as `(lifecycle only)` and use `lifecycle-only` health states instead of presenting them as normal Harbor advancement.
- Record `completion_source` in both `latest.json` state and bundle manifests.
- Regression tests started with three failures, then passed after the implementation. The late-record test proves the repaired identity selects the resumed S3 prefix and restores `76/300` with mean reward `0.671`.
- Validation: Ruff check and format passed; focused watcher tests passed (`26 passed`); repository tests passed (`556 passed, 2 skipped`).

## Future work

- If Iris gains a parent-task-only log API, replace the full job-tree repair read with that narrower query.
