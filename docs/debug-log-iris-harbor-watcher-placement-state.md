# Debugging log for Iris Harbor placement state

Determine whether `watch_iris_harbor.py` reports queued CoreWeave datagen jobs as running and turns their absent worker pods into monitor errors.

## Initial status

The 2026-07-31 17:06 UTC report labeled 13 `cw-rno2a` datagen jobs as `running` even though they had no Harbor output. Seven jobs older than the two-hour startup grace also emitted `No running Iris pod found` monitor errors.

## Hypothesis 1

The watcher displays the root Iris job state and does not inspect the current task state. Iris can keep a root job in state 3 (`running`) while its only task is state 2 (`building`) and waiting for Kueue placement.

## Changes to make

Query the controller for the root task state, derive the displayed state from the root job and task, and avoid worker-log health probes until the task reaches state 3 (`running`).

## Results

The live controller confirmed the hypothesis. Jobs 33, 44, 45, 50, 51, 52, 55, 59, 60, 61, 62, 63, and 64 had root `job_state=3`, current `task_state=2`, and no attempt start timestamp. `iris job summary` reported `building=1` with `SchedulingGated`, and Kubernetes reported the corresponding pods as `SchedulingGated`.

Three regression tests failed on `main`: the parsed job had no task state, Ray/vLLM collection still searched for a running pod, and health reported `output-unavailable`. The watcher now queries the aggregate active task state, displays `awaiting placement` while the root task is pending or building, skips Ray/vLLM collection until placement, and retains the root controller state separately in JSON and bundle manifests.

The focused watcher suite passes 10 tests, the analysis suite passes 36 tests, and the full repository suite passes 470 tests with 2 existing skips.

## Future work

- [x] Record the regression test result and final watcher behavior.
