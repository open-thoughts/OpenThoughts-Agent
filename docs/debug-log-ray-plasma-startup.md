# Debugging log for Ray plasma startup

Keep large-object-store Ray clusters from failing nondeterministically while the local plasma store initializes.

## Initial status

Jupiter job 1262271 started GCS and registered all five worker nodes, but its head raylet exited before the
cluster became usable. The preserved `raylet.out` records ten failed attempts to connect to the local
`plasma_store` socket, followed by a fatal `No such file or directory`. The launcher then reached its bounded
cluster-startup deadline and preserved the Ray logs.

## Hypothesis 1

The existing `RAY_raylet_start_wait_time_s=120` setting also governs the local plasma-socket connection.

## Results

Refuted by Ray 2.51.1 source. GCS registration reads `raylet_start_wait_time_s`; the plasma client instead reads
`raylet_client_num_connect_attempts`, whose default is ten attempts at one-second intervals.

The immediately preceding job 1262020 used the same 40 GiB object-store size and runtime. Its plasma store became
ready after about eight seconds. In job 1262271, initialization exceeded ten seconds and the raylet aborted. This
paired observation supports a startup race rather than an unreachable head node.

## Hypothesis 2

Passing `RAY_raylet_client_num_connect_attempts=120` through the shared Apptainer prefix will bound the same
plasma-socket startup phase without weakening the existing GCS-registration deadline.

## Changes to make

- Preserve `RAY_raylet_start_wait_time_s=120` for GCS registration.
- Add the independent plasma-socket attempt limit to every in-container Ray invocation.
- Test both defaults and explicit operator overrides through `build_apptainer_prefix`.

## Results

The focused regression test failed before the change because the plasma-socket control was absent and passes
after both independent controls are emitted.

Jupiter job 1264144 validated the runtime contract against Ray 2.51.1 in the production r5 SIF. It enabled
plasma page preallocation, created a 40 GiB object store, connected a driver, observed all four GH200 GPUs, and
shut Ray down cleanly. Slurm completed the job with exit code 0 in 42 seconds.

## Future work

- [ ] Confirm the next six-node TaskTrove resume forms the complete Ray cluster before trainer initialization.
