# Iris Cluster Ops

Two Marin-managed clusters reached through the `iris` SDK: the **CoreWeave GPU** cluster
(`cw-us-east-02a`, 8× H100-80GB + InfiniBand/node — the GPU-RL path) and the **Google TPU**
cluster (`marin` — datagen/eval). This doc is the ACCESS/HARDWARE/SCHEDULING reference for
both; the launch HOW-TOs live in the **`rl-agentic-launch-iris`** / `datagen-launch-iris` /
`eval-agentic-launch-iris` skills (ops = particulars, skill = procedure).

---

## §0 Preamble

Shared env/setup every iris operation needs.

- **Python = the otagent py3.12 conda env, FULL interpreter path:**
  `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` (symlinks fail in the sandbox).
- **`iris` binary = `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`** — the reliable
  default. The marin `.venv` ships a **broken `kubernetes`** (dist-info present, module not
  importable) → every **CoreWeave** `iris` command dies with
  `ImportError: Install iris[controller]` (cw is a k8s controller backend). The otagent env has
  working `kubernetes` 35.0.0 + editable iris. For the **TPU `marin`** cluster the marin `.venv`
  iris works: `IRIS=/Users/benjaminfeuer/Documents/marin/.venv/bin/iris`
  (or `conda activate marin && uv run iris`).
- **`source "$DC_AGENT_SECRET_ENV"`** before any submit/mirror script — the launcher forwards
  `HF_TOKEN` / `WANDB_*` / `DAYTONA_*` from the launch host's env into the task pod. Without
  `DAYTONA_*` every agentic trajectory finalizes `VerificationNotCompletedError` reward 0; without
  `HF_TOKEN` weight/data resolution fails.
- **`export KUBECONFIG=~/.kube/coreweave-iris-gpu`** is a HARD PREREQUISITE for every
  **CoreWeave** job/query. This Mac's default `~/.kube/config` points at a different context
  (TPU/`marin`/other); without the export, `kubectl` inspects the wrong cluster and `iris` cw
  commands fail with misleading "0 pods / not found" / auth errors. Re-export in any fresh shell
  or background call. (rno2a uses `~/.kube/coreweave-iris` — see §2.)
  `scripts/iris/list_iris_jobs.py` is the exception: it selects the canonical kubeconfig for
  `cw-us-east-02a` or `cw-rno2a` itself, so its no-argument local invocation is safe even when
  an inherited `KUBECONFIG` points elsewhere.
  - The stale Beta/k3d `KUBECONFIG` export is disabled in `~/.zshrc`. CoreWeave commands still
    need the region-specific kubeconfig above when they do not select one explicitly.
- **All `iris`/`kubectl` calls SYNCHRONOUS — never background them.**
- **`--cluster=cw-us-east-02a` is a TOP-LEVEL flag** (BEFORE the subcommand:
  `iris --cluster=cw-us-east-02a job logs …`), not a per-subcommand option. Bare
  `job stop <jid>` errors "No controller specified"; `job stop <jid> --cluster …` errors
  "No such option". `stop`/`kill` are aliases → prints `Terminated jobs:`.
- **⚠ SECRET-BEARING kubectl (all iris clusters):** `kubectl describe pod <task-pod>` and
  `kubectl get pod <task-pod> -o yaml` dump `IRIS_JOB_ENV` — the pod's **live API keys**
  (`DAYTONA_*`, `HF_TOKEN`, `WANDB_API_KEY`, `OPENAI_API_KEY`, …) — into their output. Do NOT run
  `describe`/`-o yaml` on a task pod when you only need state; use `kubectl … get pods` (state)
  or read ONE var via `kubectl -n iris exec <pod> -- printenv <VAR>` (never echo the value).

---

## §1 Hardware

### CoreWeave H100 (`cw-us-east-02a`)

`cw-us-east-02a` is homogeneous: **one node shape, whole-node-exclusive** (`H100x8`, one iris
task per node, no co-tenants). Node composition `[obs]`:
- **8× NVIDIA H100-80GB SXM5** per node, **x86_64** host (NOT aarch64/GH200).
- **~128 CPU cores/node**; **~64–68 are system/daemonset overhead** → ~48–60 free (why
  `--cpu 48` admits a multi-node gang and `--cpu 64` does not).
- **~2 TB host DRAM/node** allocatable.
- **NVLink (NVSwitch) intra-node + InfiniBand inter-node.** A TP=8 vLLM engine places
  intra-node on one 8-GPU node (no cross-node TP).
- **~36 H100 nodes total** `[obs]` → ceiling ~4 simultaneous 8-node gangs (minus other tenants).

Per-chip **H100-80GB SXM5** spec (NVIDIA datasheet; dense regime — we don't run 2:4 sparsity,
so sparsity-doubled figures don't apply):
- **80 GiB HBM3, 3.35 TB/s** bandwidth
- **bf16 / fp16 tensor: ~989 TFLOPs/s** dense (1979 w/ sparsity)
- **FP8 tensor: ~1979 TFLOPs/s** dense (3958 w/ sparsity) — native FP8 (Transformer Engine)
- **int8 tensor: ~1979 TOPS** dense
- **TF32 tensor: ~495 TFLOPs/s** dense · **FP64 tensor: ~67 TFLOPs/s**
- **NVLink: 900 GB/s** per GPU (4th-gen, full all-to-all via NVSwitch on-node)
- 700 W TDP · **sm_90** (Hopper → `TORCH_CUDA_ARCH_LIST="9.0"` for from-source builds)

H100x8 node totals:

| metric | per GPU | × 8 GPUs |
|---|---|---|
| HBM | 80 GiB | 640 GiB |
| HBM bandwidth | 3.35 TB/s | 26.8 TB/s aggregate |
| bf16 FLOPs/s | ~989 TFLOPS | ~7.9 PFLOPs/s (dense) |
| FP8 FLOPs/s | ~1979 TFLOPS | ~15.8 PFLOPs/s (dense) |
| int8 OPs/s | ~1979 TOPS | ~15.8 POPS (dense) |
| NVLink BW | 900 GB/s | on-node all-to-all (NVSwitch) |
| host CPU | — | ~128 cores (~48–60 free) |
| host DRAM | — | ~2 TB/node |

**Interconnect (what shapes the parallelism):**
- **Intra-node = NVLink/NVSwitch, 900 GB/s/GPU, full all-to-all.** TP=8 (and TP=8 + DCP=2)
  belongs ON ONE NODE: decode/EP all-reduce + all-to-all ride NVLink, not the slower fabric.
  **Use NCCL defaults** — the GH200/SIF disables (`NCCL_P2P_DISABLE` / `NVLS=0` / `COLLNET=0`)
  would cripple this on-node path.
- **Inter-node = InfiniBand**, gang-scheduled in one IB leaf fabric (Kueue
  `topology 'infiniband'`, all-or-nothing — see §2). Typical CoreWeave H100 config is
  **8× 400 Gb/s NDR** (one NIC/GPU, GPUDirect RDMA) ≈ 3.2 Tb/s/node — *datasheet-typical, not
  re-measured on `cw-us-east-02a`.* Cross-node collectives (FSDP all-gather/reduce-scatter on
  the policy mesh, inter-engine) go over IB → keep TP intra-node, shard the slower axes
  (FSDP/CP) across nodes.

**RL geometry implications:**
- **640 GiB HBM/node** hosts a TP=8 vLLM engine for a 30B–35B-class MoE at long context: the
  131k MoE arm runs **4 engines × TP=8 / DCP=2** (32 inference GPU = 4 nodes) — each engine one
  node, KV + weights under 640 GiB at `gpu_memory_utilization 0.80`.
- **H100-80GB < GH200-96GB HBM** → porting a GH200 config, the per-GPU budget tightens: drop
  `gpu_memory_utilization` (0.80→0.75) / `max_num_seqs` first on a KV-bind OOM.
- **~2 TB host DRAM** is generous for `cpu_offload`, but host-RAM OOM at FSDP weight-load on
  the policy nodes is still possible at 131k + EP8 + offload; reduce `n_concurrent` / the
  rollout-worker count if it recurs.
- **sm_90** everywhere → from-source builds (vLLM fork, flash-attn) target
  `TORCH_CUDA_ARCH_LIST="9.0"` (baked in the gpu-rl image; canonical build in **MarinSkyRL**
  `docker/Dockerfile.gpu-rl` + `docker/build_gpu_rl_kaniko.sh` — see `build-gpu-rl-image-iris`).

### Google TPU (`marin`)

#### v5p-32

v5p-32 slice composition (from iris workers table):
- 4 workers (= 4 hosts), each total_tpu_count=4 chips, chips_per_host_bounds=2,2,1 → **16 chips/slice**
  (confirms [v5p_naming_cores_not_chips]: v5p-N = N cores = N/2 chips)
- Per-host: 464.7 GB DRAM, 207 CPU cores

Per-chip v5p spec (Google datasheet):
- 95 GiB HBM2e, 2765 GB/s bandwidth
- 459 TFLOPs/s bf16 · 918 TOPS int8
- No native FP8 (a v6e feature). Qwen122B-FP8 weights dequantize to bf16 before matmul on v5p, so bf16 is the relevant compute number.

v5p-32 slice totals:

| metric | per chip | × 16 chips |
|---|---|---|
| HBM | 95 GiB | 1,520 GiB (~1.49 TiB) |
| HBM bandwidth | 2765 GB/s | 44.24 TB/s aggregate |
| bf16 FLOPs/s | 459 TFLOPS | 7.34 PFLOPs/s |
| int8 OPs/s | 918 TOPS | 14.69 POPS |
| host DRAM | — | ~1,859 GB across 4 hosts |

#### v6e-8

Per-chip v6e (Trillium) spec (Google datasheet):
- 32 GiB HBM3, 1640 GB/s bandwidth
- 918 TFLOPs/s bf16 · 1836 TOPS int8
- Native FP8 at 1836 TFLOPs/s — v6e's edge over v5p (which must dequant FP8 to bf16)

v6e-8 slice totals (8 chips, single host):

| metric | per chip | × 8 chips |
|---|---|---|
| HBM | 32 GiB | 256 GiB |
| HBM bandwidth | 1640 GB/s | 13.12 TB/s |
| bf16 FLOPs/s | 918 TFLOPS | 7.34 PFLOPs/s |
| FP8 / int8 OPs/s | 1836 TFLOPS | 14.69 PFLOPs/s |
| host DRAM | — | ~1,410 GiB |

v6e-8 vs v5p-32 — same nominal bf16 throughput, very different memory:

| | v6e-8 (1 host, 8 chips) | v5p-32 (4 hosts, 16 chips) |
|---|---|---|
| HBM total | 256 GiB | 1,520 GiB (5.9× more) |
| HBM bandwidth | 13.12 TB/s | 44.24 TB/s (3.4× more) |
| bf16 PFLOPs/s | 7.34 | 7.34 (same) |
| native FP8 | yes (14.7 PFLOPs/s) | no (must dequant → 7.34 bf16) |
| host count | 1 | 4 (cross-host comms cost) |
| host DRAM | 1,410 GiB | 1,859 GiB (across 4) |

122B-FP8 fits on v5p-32 but not v6e-8: 122B weights × 1 byte ≈ 122 GiB > the 256 GiB v6e-8
budget once activations, MoE fixed footprint, and compile-time peaks are subtracted.

**Serving gotcha — prefer single-host DP=1 (marin#6136 multi-host decode bug):** ⚠️ Multi-host TPU
serving hits a decode bug (marin#6136) that forces `max_num_seqs` down (e.g. seqs=2), crippling
throughput. **Single-host `DP=1` dodges it entirely.** For 122B-FP8 @131k the validated operating point
is **v5p-8 A2: TP=4 / DP=1 / EP-on / max_num_seqs=32, fp8 kv**, `--load-format runai_streamer`,
`MODEL_IMPL_TYPE=vllm` — ~600 mean / 794 peak gen tok/s, **199 tok/s per chip** (best per-chip
efficiency), decode-CLEAN. For aggregate throughput run keep-N single-host jobs in parallel rather than
one multi-host slice. (`v5p-8` = 4 chips, so TP=4 is the per-chip ceiling — see the CORES-not-chips
note in §3.)

---

## §2 CoreWeave GPU cluster (`cw-us-east-02a`)

The **GPU RL path**: driven via `python -m cloud.iris.launch_rl_iris` (run **from the
MarinSkyRL repo root**, `cd ~/Documents/MarinSkyRL`) + the `gpu-rl` Docker image. The
self-contained **MarinSkyRL launcher** `cloud/iris/launch_rl_iris.py` is the sole live entry
point (`python -m cloud.iris.launch_rl_iris` with `--rl_config cloud/iris/configs/<cfg>.yaml`);
the old OT-Agent copy `python -m rl.cloud.launch_rl_iris` has been **REMOVED**. The controller
+ helpers (`start_rl_iris_controller.py`, `tilelang_cache_sync.py`, `run_rl.py`, config
translation) are `cloud/iris/*` in MarinSkyRL, synced to `/app`; `PYTHONPATH=/app:/opt/skyrl/skyrl-train`
(skyrl-train baked in the image at `/opt/skyrl`).

**Access:** Launch from the **local Mac** (preamble in §0), **`cd ~/Documents/MarinSkyRL`**
first (the `-m cloud.iris.launch_rl_iris` module resolves from the MarinSkyRL repo root; the
`iris` SDK itself is installed in the otagent env). The local MarinSkyRL checkout must be on
**`main`** (or a branch containing `cloud/iris/`) — the current working branch
`feuer/megatron-backend-transformers5` does NOT have the port. There is **no cluster login / no
SSH** — talk to the cluster through the `iris` SDK over a controller tunnel; the launcher uploads
your **local MarinSkyRL** workspace to `/app` (a local commit takes effect on the next launch
immediately). Runtime is self-contained: `/app` provides `cloud.iris.*` + configs;
`skyrl_train` imports from the baked `/opt/skyrl` (swap with `--skyrl-ref`).

**Cluster config** auto-resolves to `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`
(`launch_rl_iris.py:_resolve_cluster_config_default`); override with `--cluster-config` only if
it moved.

### Sibling CoreWeave GPU cluster: `cw-rno2a` (RNO2A / Reno) — 512 H100, added 2026-07 (marin PR #6909)

Second, larger CoreWeave GPU cluster. Same KubernetesProvider / Kueue-gang model; different
region, kubeconfig, and node shape.

- **Fleet:** **64× `gd-8xh100ib-i128` (8× H100-80GB + IB each) = 512 H100** + 1 `turin-gp-l`
  CPU controller node (65 nodes total). Pool **pinned fully warm** (`buffer_slices: 64` = whole
  reservation stays provisioned even when idle) → gang admits without cold node-provision wait.
  **4× the East cluster** (~36 nodes / ~256 GPU).
- **Access — DIFFERENT kubeconfig + context from East (don't reuse East's):**
  - `KUBECONFIG=~/.kube/coreweave-iris` (NOTE: **no `-gpu` suffix** — East is `~/.kube/coreweave-iris-gpu`).
    This is a 5-context file (a fresh `…-july-token` covering East/West/rno2a/oa-*); perms 600.
  - `kube_context: marin-rn02a_RNO2A` → endpoint `https://208261-6670debc.k8s.rno2a.coreweave.com`.
  - Query nodes/pods exactly like East: `KUBECONFIG=~/.kube/coreweave-iris kubectl --context marin-rn02a_RNO2A get nodes`.
  - Namespace `iris` (controller `iris-controller-*` + `finelog-cw-rno2a-*` both run there).
- **Cluster config** = `lib/iris/config/cw-rno2a.yaml` (on marin **origin/main**). ⚠ To drive it
  via the `iris` SDK the **local marin clone must be on `main`** (or a branch carrying the new
  `IrisClusterConfig` schema) — an older clone's pydantic **rejects** the new fields
  (`kube_context`, `external_object_storage_endpoint`, `checkpoint_interval_seconds`,
  `signing_key`, `trusted_cidrs`) → `extra_forbidden` at config-load. Direct `kubectl` (above)
  works regardless. Object store: `object_storage_endpoint: http://cwlota.com`,
  `external_object_storage_endpoint: https://cwobject.com`. Finelog server: `finelog deploy up cw-rno2a`.
- **⚠ NCCL_SOCKET_IFNAME on rno2a — verbatim value (use in `extra_env`):**
  `NCCL_SOCKET_IFNAME: "^ibs,ibp,lo,docker,veth,cilium,lxc"` — an EXCLUDE pattern (`^`): skip
  IB/IPoIB + virtual ifaces so the bootstrap socket falls back to the host ethernet PF. An exact
  `=enpXsYnpZ` pin is fragile per-node/region. Source of truth: `cw-rno2a.yaml:128`; also carried
  by `128GPU_80B_A3B_next_cp1.yaml:275` + `tasktrove_dq_sweep_30b.yaml`. ⚠ ANY RL config launched
  on rno2a MUST set this — omitting it lets NCCL **and Ray's raylet/GCS** (all ifaces visible under
  `host_network:true`) mis-detect → raylets `Failed to establish connection` → the 4-node gang
  never forms → silent 0-progress wedge. No-op on cw-us-east-02a (East uses `enp157s0np0`).
  (#6940 + #6950 closed; large-gang MoE stability still shaken out — see below.)
- **Ownership:** this is a **marin-team** reservation (rjpower/dlwh/Rafal Wojdyla run grug/jaxpp
  MoE experiments on it). Not our dedicated cluster — coordinate before parking a large long run.
- **Status snapshot (2026-07-13):** 512 H100 / 64 nodes **entirely idle** (0 GPUs in use; only
  controller + finelog pods in `iris`). Verified via the `kubectl … get pods -A` GPU-request tally.

### Node shape & storage (see §1 for chip specs)

Whole-node-exclusive ⇒ REQUEST ALL the node's allocatable resources (under-requesting is wasted
capacity AND a footgun). Node allocatable ≈ **128 CPU / ~2014 GiB mem / 8 GPU**. Launcher defaults
(`launch_rl_iris.py`): **`--cpu 48`** (max-admittable — >~60 fails the IB gang), **`--memory 1400GB`**
(≈ full ~2 TB leaving daemonset headroom — see Binding gotchas for the validated middle),
**`--gpus_per_node 8`**, `--disk 512GB` (rendezvous/ckpts go to the CW object store
`s3://marin-us-east-02a`, not node-local).

- **⚡ IB (NET/IB GPUDirect RDMA) — userspace BAKED (gpu-rl-7d15b25a, 2026-07-08).** CoreWeave
  auto-exposes the RDMA devices into every pod (`/dev/infiniband/{uverbs0..8,rdma_cm}`,
  `/sys/class/infiniband/mlx5_0..8`, all 9 ports `state=ACTIVE` @ `100 Gb/sec 4X EDR`) — no
  pod/config change needed. EVERY gpu-rl image BEFORE gpu-rl-7d15b25a shipped **no IB verbs
  userspace**, so NCCL silently fell back to `NET/Socket` (TCP). Expected on-launch signal with
  `NCCL_DEBUG=INFO`: `NET/IB : Using [0]mlx5_0:1/IB` + `GPU Direct RDMA Enabled`. **No external
  `libnccl-net.so`/OFI plugin is needed on Mellanox IB — the "Could not find libnccl-net.so"
  health-probe line is BENIGN.**
- **NCCL DEFAULTS — use them.** Do NOT set the GH200/SIF disables
  (`NCCL_P2P_DISABLE` / `NCCL_NVLS_ENABLE=0` / `NCCL_COLLNET_ENABLE=0`): they cripple the
  intra-node NVLink all-reduce a TP=8 (DCP) engine depends on. NCCL defaults give NVLink
  intra-node + IB inter-node. Keep the observability/raised-timeout env (`NCCL_DEBUG=INFO`,
  `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`, `TORCH_NCCL_*`).
- **Egress: CoreWeave nodes have internet.** Models/data pulled from HF **online** — do NOT set
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (contrast Leonardo/Jupiter compute nodes). The cost is
  the transient HF-weight-resolution flake below.
- **Storage/scratch:** ephemeral per-node disk via `--disk` (default `512GB`); multi-node Ray
  rendezvous + banked traces go through the CW `s3://` object store (`s3://marin-us-east-02a`;
  store moved R2→CW 2026-07-05, see Scheduling), not node-local disk. No shared persistent POSIX
  scratch like Leonardo's `$WORK` — checkpoints/exports go to HF / the object store.
- **The `gpu-rl` image is deps-only; source is synced at runtime.** The image
  (`ghcr.io/open-thoughts/openthoughts-agent`, pinned by **immutable `@sha256:` digest** in
  `launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE` — NOT the floating `:gpu-rl` tag, which stale-caches
  under `imagePullPolicy: IfNotPresent`) bakes the RL conda venv (`/opt/openthoughts/envs/rl`:
  torch 2.11 + the **vLLM fork built from source** + flash-attn 2.8.3), **MarinSkyRL editable** at
  `/opt/skyrl`, and harbor. The launcher syncs the **local OT-Agent workspace to `/app`** (first on
  PYTHONPATH) → first-party edits live on the next launch **without an image rebuild**. A
  MarinSkyRL fix that landed after the image build can be picked up live via `--skyrl-ref <commit>`;
  only the compiled vLLM fork requires an image rebuild (then **bump the digest**, using the
  immutable `:gpu-rl-<gitsha>` tag's digest).
- **Iris environment flags use two arguments:** write `-e KEY VALUE` (for example,
  `-e GITSHA "$GITSHA"`), **not** `-e KEY=VALUE`. The current Iris CLI rejects the latter before
  submitting a job.
- **⚠ BUILD THE IMAGE MULTI-LAYER (`SINGLE_SNAPSHOT=0`) — a single >8 GB layer is UN-PULLABLE cold
  over the CoreWeave→ghcr egress.** A kaniko `--single-snapshot` build collapses everything kaniko
  adds into ONE ~16.6 GB layer; the first **fresh** pull of that single-stream layer never
  completes — containerd restarts each attempt from byte 0 (`short read: expected … got <N>`) and
  dies at 8–11 GB every time, so all pods sit `ImagePullBackOff` indefinitely (NOT transient —
  don't wait it out; a relaunch re-pulls the same blob). **Fix = re-layer:** build `SINGLE_SNAPSHOT=0`
  (per-instruction layers) and split the big torch/nvidia-CUDA installs into a few pinned
  pre-install RUNs so **no single layer exceeds ~8 GB** (validated r5 image `gpu-rl-efd77b98` = 48
  layers, max 3.46 GB → pulled clean, Running in ~5 min). Quality-gate any new image with a
  throwaway 1-pod pull test BEFORE swapping a live job onto it.

### Scheduling & multi-node particulars

- **Gang scheduling.** `--num-nodes N` → `replicas=N` whole `H100x8` tasks. For GPUs with
  replicas>1, `resolve_multinode_defaults` returns **`CoschedulingConfig(group_by="leafgroup")`**
  — all N nodes co-scheduled on **one InfiniBand leaf fabric**, all-or-nothing. `cw-us-east-02a`
  enables **Kueue gang admission** (`kueue.cluster_queue: iris-cq`, `host_network: true` for
  NCCL/IB), so the N-task gang admits **atomically** (all N whole nodes granted, or it queues).
  Pods sit **SchedulingGated** (normal Kueue gang pre-admit) until admitted.
- **⚠ JUST SUBMIT — do NOT pre-check free-node count and withhold the submit.** A gang sitting
  `SchedulingGated`/pending because a transient `cw-hpc-verification`/`nhc-*` health-check sweep
  or another tenant's job holds nodes is NORMAL — **Kueue admits it atomically the moment N whole
  nodes free, with zero babysitting.** Polling the free-node count and refusing to submit until
  ≥N are free keeps the job OUT of the queue entirely, so it never gets its turn. **Submit, let
  Iris/Kueue schedule, report the id + state (running or SchedulingGated — both fine), move on.**
  The health checks are NOT blocking — the scheduler handles them. The ONLY real "doomed gang"
  is a CONFIG mis-size that can't fit one IB leaf (`--cpu 64` → the `topology 'infiniband' allows
  to fit only 2 out of N` message); `--cpu 48` avoids it.
- **`--cpu 48` is the single-IB-leaf gang constraint.** The gang must fit on ONE IB leaf; with
  `--cpu 64` only ~2/32 nodes have ≥64 free cores (daemonset overhead), so an N-node single-leaf
  gang sits SchedulingGated forever with a Kueue `topology 'infiniband' allows to fit only 2 out
  of N pod(s)` message. `--cpu 48` fits all nodes → admits immediately (QuotaReserved=True).
- **Multi-node Ray rendezvous via an `s3://` object-store bucket.** `--num-nodes>1` REQUIRES
  `--rendezvous-dir` (the launcher hard-errors otherwise). Use an `s3://` URI under the cluster's
  default bucket (`marin-us-east-02a`), e.g. `s3://marin-us-east-02a/iris/rl-<slug>/<run>`.
  **✅ DURABLE PATTERN — DON'T hardcode the region bucket; derive the storage root from
  `marin_prefix()`** (`rigging.filesystem`, returns `data_config().resolved_root()`), which
  AUTO-RESOLVES to the active cluster's correct bucket. Build `--rendezvous-dir` /
  `--s3-output-dir` / `--gcs-output-dir` off `marin_prefix()` so a launch follows a store
  migration automatically. Region helpers: `marin_prefix_for_region(region)`
  (`marin.rl.placement`), `marin_region()`. Hardcode `s3://marin-us-east-02a` /
  `gs://marin-models-us` only as a fallback.
  **⚠ Store moved R2 (`s3://marin-na`) → CW (`s3://marin-us-east-02a`) on 2026-07-05 (marin
  `c7caecc95a`):** pods inject CW creds + `AWS_ENDPOINT_URL=cwlota.com` and can NO LONGER reach
  `s3://marin-na` (R2) — a `marin-na` rendezvous PUT resolves to the nonexistent
  `marin-na.cwlota.com` and STALLS. The cluster injects working creds via the **`iris-task-env`
  k8s Secret** (`envFrom`, because `storage.remote_state_dir` is an `s3://` URI), so **no external
  creds needed** — and you must **NOT forward `AWS_*`/`R2_*`**: explicit container `env` overrides
  `envFrom`, so forwarding the launch host's `AWS_*` (different account, no `AWS_ENDPOINT_URL`)
  clobbers the pod's injected creds and silently targets real AWS S3. Use a **fresh sub-path per
  run** so a stale head file isn't picked up. Mechanism: one `start_rl_iris_controller.py` per
  node; rank 0 writes `ray_head.json`, workers poll + join; rank 0 publishes `ray_head.done`.
  **Inspecting a `marin-us-east-02a` object FROM THE MAC — YES, via the EXTERNAL endpoint (validated 2026-07-18).**
  The cluster config exposes BOTH `object_storage_endpoint: http://cwlota.com` (in-cluster LOTA — the pods'
  injected `AWS_ENDPOINT_URL`, unroutable from the laptop) AND **`external_object_storage_endpoint: https://cwobject.com`**
  (Mac-reachable). So the store is NOT laptop-inaccessible — use the EXTERNAL endpoint with the CW creds. Recipe
  (validated): extract the CW creds from the `iris-task-env` k8s Secret (holds `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL` / `FSSPEC_S3`) and use them with `endpoint_url=https://cwobject.com`
  + `addressing_style=virtual` (CW/LOTA **rejects path-style**, boto3's default):
  ```
  export KUBECONFIG=~/.kube/coreweave-iris-gpu
  AKID=$(kubectl -n iris get secret iris-task-env -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
  ASK=$(kubectl -n iris get secret iris-task-env -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)
  AWS_ACCESS_KEY_ID=$AKID AWS_SECRET_ACCESS_KEY=$ASK python -c 'import boto3;from botocore.config import Config;s3=boto3.client("s3",endpoint_url="https://cwobject.com",config=Config(s3={"addressing_style":"virtual"}));print([p["Prefix"] for p in s3.list_objects_v2(Bucket="marin-us-east-02a",Prefix="<prefix>/",Delimiter="/").get("CommonPrefixes",[])])'
  ```
  ⚠ SECRET-BEARING: extract creds into shell vars/env only, NEVER echo the values (you're reading a secret k8s Secret).
  `s3fs`/fsspec Mac-side: `storage_options={"client_kwargs":{"endpoint_url":"https://cwobject.com"},"config_kwargs":{"s3":{"addressing_style":"virtual"}}}`.
  (subdirs are `CommonPrefixes`→`["Prefix"]`; top-level files are `Contents`→`["Key"]`.) The in-pod
  `kubectl -n iris exec <own task pod>` route (using the internal `AWS_ENDPOINT_URL=cwlota.com`) still works when
  you're already operating in-cluster; do NOT exec into someone else's active training pod (e.g. dlwh's `…grug-train-cw…`).

  **⚡ RAY ACTOR LOGS — the ONLY place an engine-side / actor-side fatal traceback survives (ALWAYS fetch these on an RL death/wedge).** The iris `job logs` / RL finelog show the *teardown* (`ray-log-sync uploaded …`, then `Killed ray::IDLE / raylet / gcs_server`), NOT the root exception — a colocated-engine RL job can die `exit=0` with NO traceback in the job log while the real killer (a vLLM `EngineCore` fatal, a Ray `ObjectLostError`, an actor crash) is only in the per-actor Ray logs. Those are uploaded to S3 at **`<rendezvous-dir>/ray_session_logs/`** — the `--rendezvous-dir` path, NOT the job-name `…/<job>/trace_jobs` path (e.g. rendezvous `s3://marin-us-east-02a/iris/<slug>/run-<ts>/` → logs at `.../run-<ts>/ray_session_logs/`). ⚠ The controller's `[ray-log-sync] uploaded … -> …/ray` line UNDERSTATES the prefix; the real subdir is `ray_session_logs/` (list the run prefix to confirm; also `term_artifacts/`).
  - **Fetch** (same CW creds + external endpoint; bounded — a keep-1 run ≈ 225 MB / ~6k files):
    ```
    export AWS_ACCESS_KEY_ID=$(kubectl -n iris get secret iris-task-env -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
    export AWS_SECRET_ACCESS_KEY=$(kubectl -n iris get secret iris-task-env -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)
    export AWS_S3_ADDRESSING_STYLE=virtual   # CW rejects path-style
    aws s3 sync s3://marin-us-east-02a/iris/<slug>/run-<ts>/ray_session_logs/ <scratch>/ray/ --endpoint-url https://cwobject.com --quiet
    ```
  - **The application tracebacks live in `worker-*.out` / `worker-*.err`** (the Ray actor stdout/stderr — RolloutCoordinator, `AsyncVLLMInferenceEngine`/`EngineCore`, run_rl driver), NOT `python-core-*.log` (Ray's C++ core event logs, no Python traceback) and NOT the 1172× `event_*` files. Locate with `grep -rlE "Traceback \(most recent call last\)|<ErrorClass>" <scratch>/ray`, then read the `worker-*.out/.err` — the exception-terminating lines (`^[A-Za-z_.]*Error:`) are the root cause. The rank subdir + `session_latest/` holds the freshest.
  - **Worked example (keep1-v22, 2026-07-19):** a silent ~34-min "death" (`exit=0`, clean teardown) was explained ONLY by the ray logs: vLLM `EngineDeadError` from `_C_cache_ops::reshape_and_cache_flash: attempted to run this operator with Meta tensors` (a KV-cache write against meta-device model tensors during the first-policy-step weight-sync) → un-pickleable `_pickle.PicklingError: Can't pickle RayTaskError(EngineDeadError)` → driver cascade → orderly teardown. NONE of this appeared in `iris job logs`.
  (Aside — the on-disk shape of a marin/Levanter *training* checkpoint there: `<run>-<hash>/.executor_info`
  + `checkpoints/step-N/{metadata.json, manifest.ocdbt, d/<content-addressed blobs>}` = **Orbax OCDBT /
  TensorStore JAX** format, NOT HF transformers — no `config.json`/`*.safetensors`/`tokenizer.json`;
  needs an explicit Levanter→HF export to become loadable by `transformers`.)

### CoreWeave datagen diagnostic capture (vLLM logs + trial sample)

For a **live** `launch_tracegen_iris.py` GPU datagen job, use the bounded collector instead of
ad-hoc `kubectl cp`/S3 commands:

```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  scripts/iris/coreweave_ops.py \
  /benjaminfeuer/<job> --cluster cw-rno2a \
  --output-dir /Users/benjaminfeuer/Documents/experiments/active/<experiment> \
  --sample-size 100 --seed 0
```

It resolves the active task pod, derives the trial S3 prefix from PID 1's
`--experiments_dir` and `--job_name`, and writes a timestamped `diagnostics/` directory with a
credential-free `manifest.json`, the task-log tail, selected Ray/vLLM logs, and a deterministic
sample of completed trial directories. It obtains the object-store credentials from the
in-cluster `iris-task-env` Secret and never writes them to disk. The default object limits are
20 MiB per trial file and 100 MiB per Ray/vLLM log; skipped objects are listed in the manifest.
Pass `--pod`, `--trials-s3`, or cluster/kubeconfig overrides only when discovery is ambiguous.

**Terminal-job evidence policy:** when a datagen job is FAILED, KILLED, or confirmed absent,
collect its logs and bounded trial evidence into that job's experiment directory **before**
diagnosis, recovery, or relaunch. Debug from the local synchronized copy, not from a changing
live pod or a second ad-hoc S3 listing. For a CoreWeave tracegen job whose pod still exists, run
the collector above immediately; it preserves `pod-task-tail.log`, Ray/vLLM logs, and sampled
`exception.txt` files under `<experiment>/diagnostics/<job>-<timestamp>/`. If the pod has already
been garbage-collected, synchronize the durable controller/Finelog history locally instead. Get
`<submitted_at_ms>` from the jobs table before it is pruned:

```bash
OUT=/Users/benjaminfeuer/Documents/experiments/active/<experiment>/diagnostics
mkdir -p "$OUT"
env -u KUBECONFIG /Users/benjaminfeuer/Documents/marin/.venv/bin/iris \
  --cluster=cw-rno2a job logs /benjaminfeuer/<job> \
  --since-ms <submitted_at_ms> --max-lines 50000 --no-tail \
  | tee "$OUT/<job>-controller.log"
```

`analyze_iris_harbor_job.py` currently resolves only `gs://` output roots, so do **not** use it for a
terminal CoreWeave S3 job. For TPU datagen, use its
`--output <experiment>/diagnostics/<job>-history.md --resync` collector; it paginates durable
Finelog history and reports durable trial progress. Record the local diagnostic directory in the
incident/update before taking recovery action. A successful state-4 completion does not require
this failure forensics capture.

For rno2a controller checks, a shell that has sourced `secrets.env` may inherit the unrelated
`~/.kube/lambdaconfig`. Use `env -u KUBECONFIG` with `iris --cluster=cw-rno2a ...` and with
`scripts/iris/iris_ops.py`; do not treat that kubeconfig error as an Iris/DNS failure.

**Datagen status tables must include trial completion explicitly.** Report `completed / total`
and `remaining` for every arm, where `completed` is the count of direct
`<trial>/result.json` objects and `remaining = total - completed`. A `result.json` records a
terminal trial, not necessarily a usable trace: report verifier rewards and exception counts
alongside it when assessing quality. Do not use raw trial-directory count as completion; it also
includes metadata and incomplete directories.

### Observability (verify, monitor, fetch logs)

**Verify access** (cheap, before submitting):
```bash
# iris-side: my live jobs (JobState enum — see §5 iris_ops.py)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a query \
  "SELECT job_id,state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%'" -f csv

# k8s-side: H100 node headroom (an N-node gang needs N WHOLE free 8-GPU nodes)
kubectl get nodes        # with KUBECONFIG=~/.kube/coreweave-iris-gpu  (Ready count ONLY — see trap below)
```

**Are nodes actually free? Use Kueue + a per-node free-GPU count — NOT `kubectl get nodes` or a
pod request-sum.** `kubectl get nodes` shows *Ready*, not *free*; the naive "sum
`requests.nvidia.com/gpu` over running pods vs 36×8" is wrong twice over: (a) allocatable GPUs is
**~256, not 288** — only **~32 of the ~36 Ready nodes carry 8 GPUs** (the rest are util/control
nodes with 0 GPU), and (b) the request-sum **undercounts** because some pods declare GPUs via
`limits`, not `requests`. Both errors make a busy cluster look free → relaunch into contention and
get **preempted by higher-priority `/power` jobs** (interactive < production). Use the two
authoritative signals instead:

```bash
# (1) Kueue ClusterQueue = the SCHEDULER'S OWN accounting — literally what decides gang admission.
kubectl get clusterqueue                 # PENDING WORKLOADS column: 0 = no admission backlog (good sign)
kubectl get clusterqueue iris-cq -o json | python3 -c 'import json,sys; d=json.load(sys.stdin)["status"]; \
print("admitted:",d.get("admittedWorkloads"),"pending:",d.get("pendingWorkloads")); \
print([{r["name"]:r.get("total") for r in f["resources"]} for f in d.get("flavorsUsage",[])])'
#   (nominalQuota lives under .spec.resourceGroups[].flavors[].resources[].nominalQuota; quotas mix units
#    like "1G" for memory — parse GPU separately, don't int() the whole map.)

# (2) CORRECT free whole-nodes = per-node (allocatable_gpu - sum of bound-pod GPU req/limit), count nodes with >=8 free.
kubectl get nodes -o json | python3 -c '
import json,sys,subprocess
nodes=json.load(sys.stdin)["items"]
alloc={n["metadata"]["name"]:int(n["status"]["allocatable"].get("nvidia.com/gpu",0)) for n in nodes}
pods=json.loads(subprocess.run(["kubectl","get","pods","-A","-o","json"],capture_output=True,text=True).stdout)["items"]
used={}
for p in pods:
    if p.get("status",{}).get("phase") in ("Succeeded","Failed"): continue
    nn=p.get("spec",{}).get("nodeName")
    if not nn: continue
    g=0
    for c in p["spec"].get("containers",[]):
        r=c.get("resources",{}); req=r.get("requests",{}) or {}; lim=r.get("limits",{}) or {}
        g+=int(req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu",0)))
    used[nn]=used.get(nn,0)+g
gpu_nodes=sum(1 for a in alloc.values() if a>0)
free=sum(1 for n,a in alloc.items() if a>0 and a-used.get(n,0)>=8)
print(f"GPU nodes:{gpu_nodes}  fully-free 8-GPU nodes:{free}  total free GPUs:{sum(max(0,a-used.get(n,0)) for n,a in alloc.items())}")'
```
Decision rule for relaunching an idle gang: only submit when **`pendingWorkloads == 0`** AND
**fully-free 8-GPU nodes ≥ N** (the gang size; e.g. ≥16 for a 30B+35B pair). If `/power` is
bursting (free nodes oscillating), wait for it to drain or escalate to `--priority production` —
do NOT churn-relaunch at interactive into contention. The in-container invocation the launcher
ultimately drives is `uv run iris --cluster=cw-us-east-02a job run …` (the SDK `IrisClient.submit`
path); you do not type that by hand — `python -m cloud.iris.launch_rl_iris` builds it.

**⚑ Priority bands + node cap (operator 2026-07-16).** `--priority {production, interactive, batch}`
(`cloud/iris/launch_rl_iris.py` → `PRIORITY_BAND_*`; default `interactive`). Bands are ordered
production > interactive > batch; a higher band preempts a lower one.
- **`--priority batch` → NO node cap. Surge freely.** Batch is the lowest band, fully preemptible;
  submit as many batch nodes as the clusters will hold across BOTH CW clusters.
- **Non-batch (`interactive`/`production`) → the ~24-node soft cap applies** across ALL my
  non-batch CW jobs combined (24 is the current operator-set value — treat the latest authorization
  as binding).
- **Accounting:** sum only **non-batch** node usage against the ~24 cap; batch jobs don't count.
- **Cross-cluster moves are autonomous.** Relocate a job between `cw-us-east-02a` and `cw-rno2a`
  freely when one cluster is packed and the other has room. Placement is launch-flag-level
  (`--cluster` / `--cluster-config` / `--rendezvous-dir`); the rl_config is mostly cluster-agnostic.
  **Before any move, grep the config for hardcoded interface/PF/host/region values in `extra_env`.**
  The trap is **`NCCL_SOCKET_IFNAME`**: East's Ethernet PF `enp157s0np0` does not exist on rno2a,
  so a hardcoded PF dies upstream of everything with NCCL `Bootstrap: no socket interface found`.
  Use the cluster-PORTABLE exclusion list `"^ibs,ibp,lo,docker,veth,cilium,lxc"` (matches
  cw-rno2a.yaml's default; auto-picks Ethernet on either cluster) instead of a hardcoded PF.
  Kubeconfigs: East `~/.kube/coreweave-iris-gpu` (`--cluster=cw-us-east-02a`), rno2a
  `~/.kube/coreweave-iris` (`--cluster=cw-rno2a`).
- **⚑ Priority preemption works (marin #7207 + #7206, 2026-07-16).** Kueue is MANDATORY on the
  k8s backend (`kubernetes_provider.kueue.cluster_queue` must be set — for us `iris-cq`); a single-pod
  GPU job that bypassed Kueue used to silently defeat preemption of the gangs beside it.
  `preemption.withinClusterQueue: LowerPriority` makes higher bands evict lower; #7206 makes
  preempt-and-place **atomic** (the preemptor binds to the worker its victim frees — no gap where
  the freed worker is stranded, stolen by a solo first-fit task, or over-preempts fresh victims
  tick after tick; gangs re-form on the freed slice). The "surge freely on batch, it never blocks
  anyone" model is now genuinely sound — batch WILL be evicted the moment a higher band needs the
  nodes — so the ckpt→s3 durable-resume fix below is load-bearing for any long batch run.
- **⚠ Preemption trade-off — checkpoint-resume is NOT yet plumbed for CoreWeave preemption
  (2026-07-16).** A batch job WILL get preempted when a higher band needs the nodes. But our RL
  runs write the resumable checkpoint to **ephemeral pod-local disk**, not s3: `ckpt_path` is null
  in `hpc/skyrl_yaml/iris/*.yaml` → auto-derives to `{experiments_dir}/{job}/checkpoints` and
  `experiments_dir` defaults to the in-container `/app/experiments` (`cloud/iris/launch_rl_iris.py`),
  which a preempt/re-bring-up wipes. The trainer's save+resume is correct and s3-capable
  (`resume_mode: latest` is set; fsspec supports s3://), so the ONE fix is to point
  `trainer.ckpt_path` at a **stable per-job** `s3://marin-us-east-02a/iris/<job_name>/checkpoints`
  (auto-derive in the launcher, mirroring the `trials_dir` pattern at `launch_rl_iris.py:879`, OR
  set it in the YAMLs). Until that lands, a preempted batch job **restarts from step 0** — so
  batch is safe only for short/smoke runs; do NOT park a long run on batch expecting resume.
  (SLURM resumes fine because `experiments_dir` there is durable `$WORK`.) Verify the s3-ckpt
  round-trip at 80B FSDP scale at a smoke before trusting it in production.

**Monitor liveness — state-poll, NOT a log-string watch.** Poll the authoritative iris job
lifecycle state, never grep rank-0 logs for a content string. A clean kill / eviction /
preemption / early crash often emits **no** terminal log line, and the pods are reaped, so a
content-watch sits idle while the job is gone. The watch primitive:
```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
$PY scripts/iris/iris_ops.py /benjaminfeuer/<job> --once --json    # authoritative state now
$PY scripts/iris/iris_ops.py /benjaminfeuer/<job> --interval 60     # watch until terminal
#   wraps `iris job summary --json` (auth) + SQL `query` fallback + kubectl pod cross-check;
#   "no record AND 0 pods" => terminal `absent`. Importable: get_job_state() / watch().
```
Log-content greps (`scripts/iris/analyze_iris_harbor_job.py`) are for sel_rows / EPDIag / throughput
**science only** — never for liveness/terminal detection. (The launch HOW-TO's §8 carries the full
monitoring rule.)

**⚠ Empty job-state ≠ still-running — the `jobs` table is PRUNED, not a durable ledger.
Absence-after-existence is TERMINAL; never hand-roll a `state IN (4,5,6)` watcher.** A completion
watcher that polls `iris query "SELECT state FROM jobs WHERE job_id='...'"` and only fires on a
terminal `state` (4 SUCCEEDED / 5 FAILED / 6 KILLED) will **poll forever past a real completion**:
the controller's background pruner DELETES a terminal job's row, so once pruned the query returns
an **EMPTY result** (no `state` row) and the terminal case never matches. (`job summary` reads
the same DB, so it too returns "no record" once the row is pruned.)
- **Pruning mechanics (marin `lib/iris/src/iris/cluster/controller/`):** pruner
  (`pruner.py:_prune_terminal_jobs`) runs every `prune_interval` (**1 h**, `controller.py:236`)
  and deletes terminal jobs older than `job_retention` (**7 days** default, `controller.py:239`;
  **no override in `lib/iris/config/cw-us-east-02a.yaml`**). **A FEDERATED job ages out sooner +
  less predictably:** `find_prunable_job` (`reads.py:488-505`) excludes non-local rows
  (`jobs.cluster == 'local'`, `types.py:107`) from the 7-day time-prune — a federated job's
  parent-side **mirror row is instead deleted the moment the PEER issues a tombstone**
  (`federation_changelog.tombstone=1`, `schema.py:641`) that federation-sync mirrors. A job
  submitted via the `iris.oa.dev` meta-scheduler is exactly this case: its row can vanish right
  after the peer prunes it, well inside a 7-day window.
- **ROBUST recipe:**
  1. **Prefer `iris_ops.py` — it already handles absence-as-terminal.** Its order is
     `job summary` → `query` fallback → if BOTH have no record **AND** `kubectl` reports **0 pods**,
     it returns `state="absent"` (`is_terminal=True`, exit **2**). It never reads an empty row as
     "running." **REQUIRES a correct `KUBECONFIG`** (`~/.kube/coreweave-iris-gpu` for cw-us-east-02a;
     the **peer's** kubeconfig for a federated job whose pods live on the peer) — if kubectl can't
     run, `count_live_pods` returns `None`, `get_job_state` RAISES a *transient* error, and the
     watch loop **keeps waiting** (absence can't be confirmed). Export the right kubeconfig, and
     for `--once`/`watch` do NOT pass `--no-pods` when you need the absent verdict.
  2. **If you must hand-roll:** treat the row present with a terminal `state` **OR the row ABSENT
     (empty result) with 0 live pods** as terminal→inspect. Never treat empty as "keep waiting."
- **⚠ `absent` proves the job ENDED, not that it SUCCEEDED** — the terminal state (4/5/6) was
  pruned along with the row, so a watcher that raced the prune learns only "gone." For a pass/fail
  verdict either read the state **inside** the retention window, or use the artifact signal below.
- **✅ Artifact-producing jobs (HF export, s3/gcs write) — watch the ARTIFACT directly; it is the
  ground-truth completion signal AND it is DURABLE (the job row is not).** For an HF export, poll
  the repo/revision existence rather than the ephemeral job-state:
  ```bash
  PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
  $PY -c 'from huggingface_hub import HfApi; import sys; print(HfApi().repo_exists(sys.argv[1]))' <repo_id>
  # stricter — require the expected files landed (not just an empty repo shell):
  $PY -c 'from huggingface_hub import HfApi; import sys; print(HfApi().list_repo_files(sys.argv[1], revision="main"))' <repo_id>
  ```
  For an s3/gcs object write, poll the object path exists (for `marin-us-east-02a`, in-pod boto3
  with the LOTA `virtual` addressing style — see §Scheduling). Prefer the artifact watch for these
  jobs: it fires on the thing you actually want (artifact published), it is immune to job-row
  aging, and it works even after the job-state has been pruned to `absent`.

**Finelog retains the FULL job log** — retrievable by **time-window pagination** with the
cw-capable iris binary (`/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`). The only real
truncation is `--tail`'s line cap; `--since-ms <submitted_at_ms> --no-tail` returns everything:
```bash
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris   # KUBECONFIG=~/.kube/coreweave-iris-gpu
$IRIS --cluster cw-us-east-02a query \
  "SELECT job_id,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code FROM jobs WHERE job_id LIKE '%<job>%'"
$IRIS --cluster cw-us-east-02a job logs /benjaminfeuer/<job> --since-ms <submitted_at_ms> --max-lines 500000 --no-tail
```
(e.g. on dead `rl-131k-cpdcp2r3-v2` one `--no-tail --since-ms <submit>` recovered all 2275 lines
spanning the full 10:09:08 → 10:16:21 lifetime, revealing the rank-0 fatal
`ModuleNotFoundError: No module named 'torchtitan'` in `fsdp_utils.py:667 apply_ep` — the MoE EP
path needs torchtitan, not in the gpu-rl image.)

**⚠ Poll / tooling pitfalls (esp. cw-rno2a + pure-RL jobs) — learned 2026-07-14:**
- **`job summary` is FLAKY on cw-rno2a** (intermittent `execute_unary` controller blips). Drive
  rno2a liveness through `iris query` / `iris_ops.py --once` instead; retry-or-fall-back
  rather than trusting one `job summary` failure.
  - **⚠ The blip can masquerade as a false `absent` / "job not found" — even `iris_ops.py --once`
    is not safe as a single shot.** Observed 2026-07-20: 4 freshly-launched (and 1 long-running) rno2a RL
    jobs ALL read `state=absent src=absent pods=0 'no controller record AND 0 pods (disappeared)'` — the
    summary RPC hit `execute_unary` 3× (→ a definitive-looking `ConnectError: Job … not found`) AND that
    same invocation's kubectl cross-check returned pods=0, so iris_ops concluded "disappeared." It
    was a lie: **`iris query` SQL showed all 5 at `state=3`, and a fresh `kubectl -n iris get pods` showed
    4 Running pods per job.** So on rno2a, a `not found`/`absent` verdict is NOT terminal proof — a launcher
    that printed `Submitted` + a controller blip is the likelier story. **Ground-truth a scary rno2a
    absent/terminal read with BOTH `iris query "SELECT job_id,state FROM jobs WHERE state IN (1,2,3) AND
    job_id LIKE '/<user>/<prefix>%'" -f csv` AND a fresh standalone `kubectl -n iris get pods -o wide |
    grep <job-substr>` before believing a job died or never registered.**
- **⚠ `iris job kill/stop <id>` is a PREFIX kill — it defaults to `--include-children`, terminating
  EVERY job whose id starts with `<id>`.** So `job kill /benjaminfeuer/rl-opencode-keep1` ALSO kills
  `rl-opencode-keep1-lp`, `rl-opencode-keep1-v2`, etc. (learned the hard way 2026-07-21: killing the old
  `rl-opencode-keep1` also reaped the freshly-relaunched `rl-opencode-keep1-lp`). **To kill exactly one
  job whose name is a prefix of another live job, pass `--no-include-children`.** Always `--dry-run`
  first when the target name could be a prefix of a job you want to keep. Corollary for NAMING: don't
  suffix a relaunch onto a live job's exact name (`<name>-lp`) — pick a sibling-distinct stem
  (`rl-ocfix-v81`, not `rl-opencode-keep1-lp`) so a later prefix-kill can't cross-hit it.
- **jobs-table columns are the `*_ms` names** (`submitted_at_ms`, `started_at_ms`,
  `finished_at_ms`, `error`, `exit_code`) — a bare `submitted_at`/`failure_count` errors `no such
  column`. If a `query` column errors, drop it and re-select `name, state` (don't abandon the query).
- **`analyze_iris_harbor_job.py` does NOT "just work" for cw-rno2a or pure-RL jobs:**
  - Region resolver passes `--config <cluster-name>` to iris, which wants a PATH → it errors `Path
    'cw-rno2a' does not exist`. Workaround: pass the config YAML **path** as `--cluster` (e.g.
    `--cluster /Users/benjaminfeuer/Documents/marin/lib/iris/config/cw-rno2a.yaml`) AND put the cw
    iris on PATH (`export PATH=/Users/benjaminfeuer/miniconda3/envs/otagent/bin:$PATH`).
  - Even then it **cannot resolve a pure-RL job's output dir** (no `--jobs-dir`/`--gcs-output-dir`
    in the baked command) → `LookupError`. It is built for jobs with a GCS trace/output dir. **For
    RL-job metrics, fetch the finelog directly** via `iris job logs --since-ms <submitted_at_ms>
    --no-tail` (bounded) instead.
- **`iris job logs` has NO server-side grep** (only `--since-ms`/`--since-seconds`/`--max-lines`/`--tail`).
  Never `--max-lines >~200 | grep` a long job into the Mac (OOM-crash risk). Bound every fetch with
  a tight `--since-ms` window + `--max-lines`.

**Per-trial `TimingInfo` duty-cycle read — CoreWeave in-pod access (recipe lives in the harbor
project doc):** The reusable recipe — the `result.json` `TimingInfo` field→phase map, the
duty-cycle fraction math, the lease-race / burst≠churn checks, and the aggregates-only discipline —
is a **harbor trial artifact**, so it lives in **`.claude/projects/harbor/ops.md` §"Per-trial
`TimingInfo` duty-cycle recipe"**. Only the cluster-specific *access* to the trials bucket is here:
- **Trials bucket is in-cluster-only from the Mac** (`marin-us-east-02a`, LOTA — see §Scheduling),
  so **aggregate IN-POD and transfer aggregates only** (never sync raw trials/logs to the Mac).
- **Trials path:**
  `s3://marin-us-east-02a/iris/<run-name>/trace_jobs/<trial>/{config.json,result.json,agent/,artifacts/,verifier/}`.
  The `<run-name>` dir is set from `trainer.run_name` (e.g. `rno2a-30b-coder-v0l`) — NOT the config's
  literal `terminal_bench.trials_dir` (the coordinator rewrites it). Discover it by listing `iris/`
  in the bucket and matching the job name.
- **In-pod S3 access:** `kubectl --context marin-rn02a_RNO2A -n iris exec <task-pod> -c task --
  python -c '...'` using boto3 with `endpoint_url="http://cwlota.com"` +
  `Config(s3={"addressing_style":"virtual"})` (LOTA rejects path-style, same as §Scheduling). List
  `result.json` keys with one paginated `list_objects_v2` over `.../trace_jobs/`, sort by
  `LastModified` desc, read the newest ~200 (steady-state duty cycle) / ~500 (error +
  re-provision tails); print only computed medians/percentiles.

Then apply the ops.md recipe to the fetched `result.json` fields.

### Binding gotchas

> **⚠ `--max-retries ≥1` for the transient HF weight-resolution flake.** At scale (e.g. 32 FSDP
> ranks each resolving sharded safetensors online) one rank can hit a transient HF Hub HTTP/EOF
> failure → transformers reports the generic `… does not appear to have a file named
> model.safetensors`; with `max_retries=0` that one rank SIGKILLs the whole gang. `--max-retries 1`
> re-brings-up the gang on that failure (time-only cost). A weight-resolution retry wrapper landed
> in MarinSkyRL (commit `0b2b05b`); keep `--max-retries ≥1` as belt-and-suspenders. (Durable
> alternative: pre-stage the model into the image's HF cache / a shared snapshot before the FSDP
> workers start, or raise `HF_HUB_DOWNLOAD_TIMEOUT`.)

> **⚠ `--memory` default is `1400GB`** (`launch_rl_iris.py:DEFAULT_MEMORY_PER_NODE`). `1800GB`
> sits so close to node-allocatable (~2014 GiB) that after daemonset + persistent-reservation
> overhead a leafgroup gang can't fit all its pods → Kueue
> `topology 'infiniband' allows to fit only K of N … excluded: resource "memory"` →
> **SchedulingGated stall**. **`1400GB` is validated** for the 8-node 131k EP8 run; drop to
> `1000–1200GB` for 2-node smokes. On an admission stall, LOWER `--memory` — **never raise a cap**.

- **Ray agent ports collide with `worker_ports` nondeterministically — pin them all.** `ray start`
  (head AND worker) lets Ray RANDOMIZE several system ports (`metrics_export`,
  `runtime_env_agent`, `dashboard_agent_grpc`, …) from the ephemeral zone that overlaps the default
  `worker_ports` range **10002–19999**. A random landing inside it aborts the node
  (`ValueError: Ray component worker_ports is trying to use a port number <N> that is used by other
  components`) — **nondeterministic** (passes or fails run-to-run; a likely cause of intermittent
  "long build then die" CoreWeave deaths). A head-only or single-port pin is INSUFFICIENT (the
  randomized port just moves to another agent). **Fix (committed `beda7a7f`,
  `cloud/iris/start_rl_iris_controller.py:_ray_port_flags`):** pin ALL of them outside the range on
  head+worker — `metrics_export=8090, runtime_env_agent=8092, dashboard_agent_grpc=8093,
  dashboard_agent_listen=8094, node_manager=8076, object_manager=8077`. Rides the `/app` upload (no
  rebuild).
- **Nodes have NODE-LOCAL storage (no shared GPFS) → stage the agentic task dataset on EVERY node.**
  CoreWeave's `/opt/openthoughts/tasks` is node-local, so a rank-0-only `parquet→tasks` extraction
  leaves the 7 workers with EMPTY task dirs → every rollout throws
  `FileNotFoundError: /opt/openthoughts/tasks/<dataset>/<instance>/task.toml` → reward 0. This is a
  SILENT data-starvation: compute path looks green (grouped-mm/R3 fine, no crash) but
  `avg_num_tokens≈1.0` and all rewards are 0. **Fix (committed `7c135780`):** the launcher forwards
  `--train-data` to the controller, which stages on every node before Ray via
  `resolve_rl_train_data`. Verify in bring-up: each rank logs
  `Staging train_data on this node (rank N/8)` → `[extract] … Done` before rollouts.
- **Transient self-healing on bring-up is NORMAL, not a fault to salvage:** a `ghcr.io` blob EOF →
  `ImagePullBackOff` self-heals (kubelet retries); `shm_broadcast: No available shared memory
  broadcast block found in 60s` is **benign** (engines idle-wait while the policy mesh loads weights).
- **k8s does NOT shell-expand `$VAR` in injected env.** Hardcode literal paths in a config's
  `extra_env:` (e.g. `LD_LIBRARY_PATH: /opt/openthoughts/envs/rl/lib`, **not** `$CONDA_PREFIX/lib`)
  — the launcher injects env as literal k8s values. (Config-authoring detail; full rules in the
  launch skill §4.)

### Daytona (orgs + sandbox lifecycle)

**Org routing — RL ALWAYS on the RL org.** Agentic RL (Daytona backend) runs on the dedicated
**Daytona RL org** (`DAYTONA_RL_API_KEY`, sha1 `f8f0296c1680`) — ~6000 fast sandboxes. Route it via
the committed **`--daytona-api-key-env DAYTONA_RL_API_KEY`** flag
(`cloud/iris/launch_rl_iris.py`), NOT a pre-launch `export DAYTONA_API_KEY=…` — the launcher
re-sources `secrets.env` after any shell export (`hpc/iris/env.py` "file overrides shell") and
CLOBBERS it. VERIFY in-pod: `kubectl exec … printenv DAYTONA_API_KEY | sha1sum` == the RL key hash.
- **DATA (`DAYTONA_DATA_API_KEY`) and main (`DAYTONA_API_KEY`) are interchangeable** general-purpose
  pools (evals use main, datagen uses DATA; either takes the other's overflow).
- **B org (`DAYTONA_B_KEY`) is USER-ONLY — NO agents / no automated jobs there, ever** (~250-sandbox
  cap). Never pass `DAYTONA_B_KEY` in an agent launch.

**Killed jobs ORPHAN their in-flight sandboxes → reap after every kill.** Harbor auto-destroys a
trial's sandbox only when the trial COMPLETES normally (the live RolloutCoordinator tears it down).
An `iris --cluster coreweave job stop` HARD-kills → the coordinator dies **before** destroying its
in-flight sandboxes → ~250–384 sandboxes ORPHAN and linger. **Fix — harbor `1143aba8`**
(marin-community/harbor `main`): `auto_stop_interval_mins` default 0→5 (now **30** on `main`), so
an idle orphan stops after 5 min → auto_delete removes it (the idle timer resets on every sandbox
exec, so active trials never trip it). **Takes effect on the NEXT image rebake** (harbor is baked
into the gpu-rl RL + eval images). **Until rebaked, manually reap after every kill:**
`python scripts/daytona/cleanup_stale_sandboxes.py --api-key-env DAYTONA_RL_API_KEY --threshold 120 --delete`
(`--threshold ≥120` so you never reap an active trial while OTHER jobs run — active agentic trials
idle 15–60 min, never >2h). Orphans take ~1h to cross the idle threshold, so a kill + immediate reap
MISSES them — reap ~1–2h later (or let the monitor/harvest cron catch them at its >1200-count trigger).

### Monitoring & debugging practices

- **A post-bring-up TRAINER wedge stays `state 3 RUNNING` and emits NO terminal log — RUNNING ≠ stepping.**
  A job can complete vLLM engine bring-up cleanly (weights loaded, KV cache sized, CUDA graphs
  captured) and then the trainer side (megatron/FSDP2 re-sync, first ppo_train step, mesh init)
  HANGS — the last real line is a benign heartbeat (e.g. an `fd-monitor` from `skyrl_entrypoint`),
  then silence for the whole wall-clock until the job goes terminal. Do NOT read "vLLM up + state 3"
  as healthy. Liveness for a trainer = **forward advancement** (a fresh training step / rising
  banked-gs / the run's `finished_at` horizon moving), never engine-bring-up completion.
- **While ANY debug thread is in flight, a single 30-minute cron re-verifies EVERY active debug job's
  authoritative state** (jobs-table `state` / `iris_ops.py --once`), independent of each job's
  own watcher — the cron backstops watchers that go silent on a clean kill/eviction/post-bring-up
  wedge. Per-job watchers are still armed, but the cron is the catch-all. Retire the cron when the
  debug roster drains. (Behavioral rule: memory `watcher-and-debug-monitoring`.)
- **Throughput / max-concurrency probes: fixed 60/120-min check-ins + DIRECT log reads — NEVER
  watcher-park a subagent.** Parked "bring-up/generation watcher" subagents re-invoke unreliably and
  once stalled a 35B max-conc probe **~8h** with no number. The throughput signal (vLLM scheduler
  `Running:/Waiting:/GPU KV cache usage:` lines + in-flight Daytona sandbox count) is readable
  DIRECTLY in one command (`iris job logs --since-ms … | grep 'loggers.py.*Running:'` + a Daytona
  `list()` count). Take the measurement at ~60 min (bring-up ~10–15 + sandbox ramp ~15–30 +
  steady-state), confirm at ~120 min, then move on; drive it with scheduled ticks, NOT an
  indefinite park.
- **Debug on the REAL failing config with the fix as the SOLE variable.** A/B: identical config, two
  runs differing only in the fix (env flag or `--skyrl-ref`) — fix-OFF must FAIL and fix-ON must
  PASS; one arm alone proves nothing. Do NOT build reduced/faster "canary" configs (smaller
  ctx/batch/steps) to debug a bug you have NOT proven they reproduce — a green result with no failing
  control is INCONCLUSIVE. Reduced configs are for SPEED of a *proven* repro only.
- **py-spy forensic on a wedged/hung job — capture BEFORE the kill.** A kill destroys the only live
  evidence of a hang (NCCL flight-recorder dumps are lost to pod GC), so on a suspected deadlock /
  collective-desync, py-spy the stuck ranks first, then recommend the kill with the stacks as
  evidence. VERIFIED working on CoreWeave despite `ptrace_scope=1` (the `task` container carries
  `CAP_SYS_PTRACE`). **py-spy is NOT on `$PATH` and NOT at `/opt/openthoughts/.venv/bin/` (stale — the
  RL venv is `/opt/openthoughts/envs/rl`, which has no `pip`/`py-spy`).** On the current gpu-rl image
  py-spy ships only in the **uv cache**; resolve it at call time (works across image rebuilds):
  ```bash
  # 1) find the rank pods (job name is TRUNCATED in the pod name — grep the user/model, not the -vN suffix):
  kubectl -n iris get pods -o wide | grep -aiE '<user>|<model-substr>'
  # 2) py-spy a stuck rank via the cached binary (no install; NO network needed):
  kubectl -n iris exec <pod> -c task -- bash -lc \
    'P=$(find /root/.cache/uv /uv/cache -name py-spy -type f 2>/dev/null | head -1); "$P" dump --pid <PID>'
  # (equivalently `uvx py-spy dump --pid <PID>` — /usr/local/bin/uvx exists — but the cache-binary form
  #  avoids a uvx resolve that can hit the network.)
  ```
  Find the target PID inside the pod with `ps -eo pid,args | grep -aiE 'skyrl_entrypoint|FSDPPolicy|EngineCore'`
  (the driver = `ray::skyrl_entrypoint`; policy rank = `ray::FSDPPolicyWorkerBase`; engine = the vLLM
  `EngineCore`/`RayWorkerProc`). The stuck Python frame is the **`(active)`** thread, not `MainThread`
  (which idles in Ray's `main_loop` awaiting the actor method). Dump several ranks to `agent_logs/`, and
  compare a LEADING vs LAGGING rank to pin which rank is stuck in which collective (the desync source).
  **A py-spy snapshot is NOT a wedge verdict:** ranks at `dist.barrier()` while others are mid-`forward`,
  plus a single NCCL `Watchdog caught collective … ran for N ms` LOG LINE, does not prove a terminal
  deadlock — a real tripped watchdog ABORTS the process (pod crash/restart). Require pod-restarts==0 +
  an actual abort/terminal state + stalled FRESH logs (all nodes) before calling wedge, and reconcile any
  cited timeout against the run's own timeline. On Leonardo py-spy is BLOCKED (`ptrace_scope=2`) — use
  another forensic.

### iris.oa.dev federated GPU submission (NEW 2026-07-09, operator — IN FLUX, bugs expected)
- **New path:** submit to `iris.oa.dev` requesting GPUs; an **H100 request auto-routes to a CW
  cluster** via a **meta-scheduler**. `--target-cluster <name>` pins a specific CW cluster. **Only
  OpenAthena accounts are authorized for CW** (we are `ben.feuer@openathena.ai`).
- **Auth:** needs `iris login` with the openathena.ai gmail (kludgy OAuth). **Iris REJECTS jobs
  submitted via the OLD-STYLE SSH TUNNEL** (to keep legacy non-OA users off CW).
- **Our EXISTING paths still work (validated 2026-07-09):** the current CW submission
  (MarinSkyRL `cloud/iris/launch_rl_iris.py` → `bundle.controller.tunnel()` +
  `KUBECONFIG=~/.kube/coreweave-iris-gpu`) and the Marin standalone agentic-eval
  submission (`experiments/agentic_evals`, `--cluster=marin`) are NOT the rejected
  "old-style SSH tunnel". `iris.oa.dev` is an EASIER path
  that becomes the default as bugs are fixed; NOT mandatory yet.
- ⚠ **Known-rough:** (1) NO queuing at the main server yet — an H100 request WITHOUT
  `--target-cluster` is RANDOMLY dispatched to a CW cluster → on the new path ALWAYS pass an
  explicit cluster; (2) log-forwarding slow (finelog push pending); (3) long tail of rollout bugs.
- **Our exposure:** we always pass explicit `--cluster=cw-us-east-02a` (RL) / `--cluster=marin`
  (TPU-eval), so the random-routing does NOT hit us; SLURM clusters don't touch iris. **⚠ OPEN:**
  watch the next submission for an auth/tunnel rejection → if so, migrate launchers to `iris login` +
  `iris.oa.dev`. (Serving ingress already uses `iris.oa.dev/proxy/t/*` — see §4.)

### Cross-reference

- **Launch procedure** (the flag set, config map + node-count derivation, config-authoring rules for
  `hpc/skyrl_yaml/iris/`, gang/rendezvous walkthrough, bring-up checklist, monitoring + completion)
  → the **`rl-agentic-launch-iris`** skill.
- **Code (canonical, MarinSkyRL `cloud/iris/`):** `cloud/iris/launch_rl_iris.py` (launcher, digest
  pin, `extra_env` forwarding, AWS_* warning), `cloud/iris/start_rl_iris_controller.py` (the
  per-node Ray rendezvous controller), `cloud/iris/run_rl.py` + `cloud/iris/rl_config_translation.py`
  (in-pod config parse → Hydra args), `cloud/iris/configs/*.yaml` (recipes). The former OT-Agent
  copies (`rl/cloud/launch_rl_iris.py`, `scripts/iris/start_rl_iris_controller.py`,
  `scripts/iris/tilelang_cache_sync.py`) have been removed — MarinSkyRL `cloud/iris/` is the sole home.
- **Standing constraints** (≤6 RUNNING RL/cluster, `enable_db_registration: false`, a3 CONCLUDED,
  Daytona snapshot caps HARD, never kill a RUNNING job / `iris cluster restart` without permission)
  — see `CLAUDE.md §Always` + the launch skill §7.

### Marin GitHub-issue monitoring + research (external skills)

CoreWeave/Iris is **Marin's** cluster — when monitoring/triaging/updating the Marin GitHub issues
this work touches (e.g. the RL-launch-on-Iris workflow issue, cluster/quota threads, upstream fixes
we depend on), use the **mumwelt skills** over the local offline Marin mirror. Which skill + how to
invoke = **`.claude/projects/mum/mum.md`** (prefer `marin-research`; `marin-context` is the
faster/less-accurate lookup).

These give the **context** to monitor + decide; the actual issue/PR **update** still goes through
the `gh` CLI.

---

## §3 Google TPU cluster (`marin`)

Operational lifecycle for OpenThoughts-Agent jobs (datagen/eval) on Marin's Iris-managed Google
TPU cloud. Canonical upstream ops: `marin:lib/iris/OPS.md`. Conventions in §0; IRIS for TPU =
`/Users/benjaminfeuer/Documents/marin/.venv/bin/iris` (or `conda activate marin && uv run iris`).

### Launch

> **ℹ NEW federated submission `iris.oa.dev` (2026-07-09) — see §2 iris.oa.dev.** Our existing
> paths (`--cluster=marin` TPU submission + the CW controller-tunnel launcher) still work and are
> NOT the rejected "old-style SSH tunnel". On the new path always pass an explicit cluster
> (no-target ⇒ random dispatch).

Two entrypoints submit `--no-wait`:
- **datagen / tracegen** → `data/cloud/launch_tracegen_iris.py`
- **agentic eval** → Marin `experiments/agentic_evals` (`python -m agentic_evals.launch`)

The standalone eval package owns its model-serving, durable-output, and
federated-endpoint behavior. See `eval-agentic-launch-iris` for the current
launch form; do not revive an OT-Agent Iris eval wrapper.

#### Before you submit — region, disk, node shape

**Region (cross-region egress is the #1 cost footgun).** Model **weight** buckets stay
multi-region: `gs://marin-models-us/...` and `gs://marin-models-eu/...` (durable inputs). Transient
**outputs** (trace dirs, eval outputs, `xla_cache`) route to a co-located **single-region** bucket
(`gs://marin-us-east5`, `gs://marin-eu-west4`, …) — ~half the multi-region cost, still read/write-local.
Keep model weight bucket + worker region in the same multi-region (all US or all EU); the launcher
handles output placement.
- Launcher auto-pins the job to the region with most capacity for the TPU type
  (`hpc/iris/regions.py:discover_region_for_tpu`) and routes output to that region's
  **single-region** bucket (`output_bucket_for_region`). Records the chosen output URI in the
  registry; readers resolve via `hpc.iris.job_output_resolver` (never hardcoded bucket).
  **Static default `DEFAULT_GCS_OUTPUT_ROOT` is `gs://marin-eu-west4/...`** (single-region EU) — the
  discovery-failed fallback; a US placement that lands here reads EU = egress, so let the pin run.
- `--gcs-output-dir gs://marin-models-us/ot-agent` **opts out of the region pin AND the
  single-region routing** (forces that multi-region bucket; places on first free worker in any US
  region — the fix for a collapsed single-region pool, stuck-PENDING below). Deliberate override;
  prefer leaving the pin on.

**Local disk (~100 GB/node ceiling).** Each TPU worker node has only ~100 GB.
- **Stream the model from GCS** (`--load-format runai_streamer`, gs:// URIs) — do NOT download a
  full checkpoint (122B-FP8 alone is ~122 GB).
- **Write `jobs_dir` to `gs://`, never local.**
- On memory-heavy/repo-based datasets, bound harbor RSS with `release_trial_payloads_in_memory: true`
  (`ctx32k_verified.yaml`) — else the orchestrator accumulates completed-trial payloads and OOMs the
  container (~256 GB host RAM, distinct from the 100 GB disk).
- `--disk` defaults to 5 GB ephemeral; raising it does not change the node ceiling.

**Node shape — get chip/host counts from the codebase, not arithmetic.** "Chips ÷ 4 = hosts" is
wrong (v5p counts *cores not chips*; v6e single-host packs up to 8 chips). Authoritative sources:
- **Host/process count:** `iris.cli.job.get_tpu_topology("<variant>").vm_count`. Known good:
  `v5p-8 → 1`, `v5p-16 → 2`, `v5p-32 → 4`, `v6e-8 → 1`, `v6e-16 → 4`. Launcher uses this to
  auto-set `--replicas`.
- **v5p naming is CORES, not chips:** `v5p-N` = N cores = **N/2 chips**. So `v5p-8` = 4 chips (1
  host), `v5p-32` = 16 chips (4 hosts). **Tensor-parallel degree must be ≤ chip count, not core
  count** — TP=8 won't fit v5p-8 (4 chips).
- **Live capacity + real chip counts:** query the cluster's `workers` table —
  ```bash
  $IRIS --cluster=marin query "SELECT device_variant, count(*) workers, sum(total_tpu_count) chips FROM workers WHERE device_type='tpu' GROUP BY device_variant ORDER BY device_variant" -f csv
  ```
- **Pools / variants / zones:** `marin:lib/iris/config/marin.yaml`. Per-chip HBM + slice totals in §1.
- 122B-FP8 fits **v5p** (95 GB HBM/chip) but **not v6e-8** (32 GB/chip, 256 GB/slice).

**Cold-compile budget:** 122B-FP8 first-serve compile can take ~60 min. Pass
`--health_max_attempts 600`; the default (~50 min) kills the job before it serves.

### Monitor

```bash
# Iris-side state (1=PENDING 2=starting 3=RUNNING 4=SUCCEEDED 5=FAILED 6=KILLED)
$IRIS --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC" -f csv

# Full-history analyzer (paginates the WHOLE log via time windows; do NOT eyeball --tail)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python scripts/iris/analyze_iris_harbor_job.py /benjaminfeuer/<job> --resync
#   sidecar JSON: total_runtime_s, iris_preemption_count, cycles[], serving_summary.{gen_tps,running}, non_empty_trials/total_trial_dirs, harbor_exception_stats

# Fetch daemon (mirrors GCS outputs → ~/.ot-agent/runs/<job>/ and captures .iris-job.log)
python -m hpc.iris_fetch_daemon status      # heartbeat should be ALIVE
$IRIS --cluster=marin job logs -f /benjaminfeuer/<job>    # live workload logs (controller tunnel)
```

> **⚠️ Never trust `iris job logs <job> --tail --max-lines N` for stats or debugging.** It truncates
> from the tail only — verbose Ray state-dumps crowd out the lines you care about, **under-sampling
> throughput by 10–100×**. **Use `analyze_iris_harbor_job.py`** — it paginates the entire log via fixed
> `--since-ms` + `--no-tail` windows and filters in python; even `--max-lines 200000` misses body
> lines the windowed walk recovers.

Outputs land in `~/.ot-agent/runs/<job>/` (daemon rsync) + `.iris-job.log`. Health signal =
productive-trial rate (`non_empty/total`) + `harbor_exception_stats`; gen tok/s varies by dataset
(short-task sets run lower). Jobs on `:tpu` image `ae085bc8`+ **auto-upload** their HF repo on
state-4 success — verify the repo exists before any manual rescue.

### Pitfalls & recovery

#### Preemption (`--preemptible` workers)
- Normal and frequent; a slice can take 10+ preempts in a few hours. Each preempt → fresh worker →
  **cold XLA recompile** (~60 min v5p-8 cold; ~13–20 min warm).
- **XLA persistent cache** makes warm restarts fast. Namespaced per CPU-microarch and per model
  under `OT_AGENT_XLA_CACHE_BASE` (region-matched bucket, auto-set on iris) — do not point two
  different host CPUs at the same cache subdir (cross-host poison → wrong execution).
- **harbor resumes from the gs:// `jobs_dir`** — completed trials persist across preempts; only
  recompile time is lost.
- `IRIS_TASK_ID` gains a `:N` suffix on retried/preempted attempts (e.g. `/user/job/0:2`) —
  rank-parsing must strip it (`.rsplit('/',1)[-1].split(':',1)[0]`) or it crashes on retry.
- **Stuck PENDING** = no capacity for that TPU type in the pinned region (preemptible pool can
  scale to zero). A finished job does NOT free its snapshot or instantly free capacity. For a fresh
  DATAGEN launch that won't place, relaunch **unpinned** with
  `--gcs-output-dir gs://marin-models-us/ot-agent` (iris places on any free US worker; note this
  override reverts outputs to the pricier multi-region bucket). Kill the stuck submission only with
  user permission.
- **⚑ PREEMPTIBLE JOBS STAY ON PREEMPTIBLE — a capacity-pending wait is NOT an escalation (operator,
  2026-07-09).** When a `--preemptible` job (datagen OR a training child respawned by its
  coordinator) sits PENDING for hours on "no workers match constraints" = pure preemptible-pool
  scarcity, that is the EXPECTED steady state. **Do NOT** propose/repin to a non-preemptible /
  on-demand slice, **do NOT** probe other zones to "rescue" it, and **do NOT** surface it to the
  user as a decision. Just report the pending status and let it self-place; a durable parent
  (`--max-retries`) or coordinator guarantees resume the moment a slice frees. (The unpinned-relaunch
  remedy above is still fine for a *fresh* datagen launch that never placed — that's a region-pin
  fix, not an on-demand upsell.)
- **⚑ DON'T GATE keep-N REFILLS ON A CAPACITY GUESS — let iris's scheduler place them (operator,
  2026-07-09).** When keep-N is below target, SUBMIT the refill(s) and let the iris queue manager
  decide placement; a submitted job that sits PENDING behind a full pool is fine (it places when a
  slice frees). **Do NOT withhold or defer a refill because you eyeballed "0 free TPUs" / a long
  pending queue.** Submit to keep-N every tick; only skip on a HARD blocker you can act on
  (Daytona snapshot cap with nothing reclaimable → note + move on). A pending refill is not an
  escalation (see the preemptible rule above).

#### How the pools work — monotonic tier ladder + crash-vs-preempt reservations (2026-07-11)

Two mechanics explain almost every long `v5p…`/`v6e…` PENDING. Both are **capacity behavior,
not config faults** — HOLD FAST; do NOT mis-escalate as quota/config blocks.

- **Monotonic tier ladder (per pool).** Each preemptible pool (e.g. `v5p-preemptible/us-east5-a`) has
  a **tier ladder by slice size** — `v5p-8` = tier 1, `v5p-16` = tier 2, `v5p-32` = tier 3, `v5p-64`
  = tier 4, … up to 2048 (static in `marin:lib/iris/config/marin.yaml` tpu_pools). **The autoscaler
  will NOT scale up a higher tier while a LOWER tier in the SAME pool has unsatisfied demand** — the
  pending reason is literally `Autoscaler: tier_blocked: quota-pool tier monotonicity`. So a `v5p-64`
  (tier-4) request can sit PENDING for hours *even when the raw chips exist*, purely because tier-1
  `v5p-8` demand (often OTHER users' jobs) is backlogged in that pool/zone. It self-resolves the
  instant the lower-tier backlog drains. Diagnose by enumerating same-pool lower-tier demand (the
  `workers` table + autoscaler snapshot: `peak_demand`, `slice_failed … no more capacity in zone`),
  NOT by touching your job.

- **A crash tears down the reservation; a preempt holds it.** While a training child is RUNNING it
  *holds* its slice, so the tier gate above is moot — which is why days of **preempt→resume** work
  fine. But a **hard crash (e.g. SIGSEGV exit 139) destroys the slice entirely**; the coordinator's
  `--max-retries` respawn must then **re-acquire the tier-N slice from scratch**, and *that* is when
  it hits the monotonicity gate at whatever the current contention is. So a crash-resume can be
  dramatically slower to place than a preempt-resume, even for the identical job — expected, not a wedge.

#### Wedged / stalled TRAINING run (coordinator + child) — checkpoint-resume
> **⚠ The Executor + `ExecutorStep` are RETIRED (marin PR #6649, 2026-07-09) → lazy `ArtifactStep`
> (`marin.execution.lazy`, `remote(fn,…)`, `name@version`). See
> `.claude/projects/marin-executor/`.** The coordinator/child + `.executor_info` model below still
> describes OLD executor-launched runs (Delphi midtrains) — read them the same way — but NEW Levanter
> training uses `ArtifactStep`. `StepRunner` + its per-step distributed lock DEADLOCK an SPMD
> (srun N-rank GPU) launch (issue #7080); workaround: call the Levanter entrypoint directly in every
> rank (bypass `StepRunner`). Full detail in the marin-executor project doc.

For executor-dispatched training (a CPU **coordinator** submits a v5p **child** training job, e.g. a
Levanter midtrain), recovery differs from datagen:
- **`--max-retries` auto-resumes ORGANIC failures only.** Child FAILED(5)/preempted → coordinator
  relaunches a fresh child resuming from the latest checkpoint. It does NOT catch a WEDGE (child
  `state 3` but frozen — never FAILED, no retry). It also **reuses the launch-time bundle**, so a
  CODE fix reaches the worker only on a FRESH relaunch.
- **Stopping the COORDINATOR is TERMINAL** — `iris job stop <coordinator>` kills children AND does
  NOT relaunch (it's the *abandon* path).
- **Liveness/wedge = ADVANCEMENT, not presence.** A training run can be `state 3 RUNNING` with
  `iris job logs` showing a recent step yet be dead — the CLI/IAP log window freezes at a stale
  timestamp. Judge by **SAVED-CHECKPOINT step AND its GCS mtime** advancing — but **use the RIGHT
  checkpoint path:**
  - **⚠ The PERMANENT path is COARSE (retains only sparse N×1500-step checkpoints) — do NOT use it
    for wedge detection.** For delphi/levanter it keeps only every-1500-steps (…3000, 4500, 6000,
    7500…), so at ~76 s/step its newest checkpoint can be **>1 DAY old on a perfectly healthy run** →
    a FALSE "frozen/wedged" reading.
    `gs://<bucket>/checkpoints/<run>/checkpoints/step-*/metadata.json`
  - **✅ Use the fine-grained TEMP (TTL) rolling-checkpoint path — that is the real liveness signal**
    (a fresh checkpoint every N steps, ~10-20 steps behind live):
    `gsutil ls -l 'gs://<bucket>/tmp/ttl=14d/checkpoints-temp/<bucket>/checkpoints/<run>/checkpoints/step-*/metadata.json' | sort -k2 | tail`
  - Cross-check with `iris job logs` live-step advancement too. Frozen TEMP-path checkpoint mtime for
    hours + frozen logs + healthy cgroup = progress WEDGE (not OOM/leak). NEVER declare a wedge off
    the permanent path alone.
- **Recover a confirmed wedge:**
  - **Option A (PRIMARY): stop the wedged CHILD only, NOT the coordinator.**
    `iris job stop <child_task_id>` forces the child terminal (KILLED) → the coordinator's executor
    respawn loop spawns a FRESH child that auto-resumes from the latest checkpoint on the pinned
    output dir. Preserves the healthy coordinator, `--max-retries` durability, and W&B run identity.
    Confirm the new child loads step-N (tqdm `Nk/29.9k`), not step 0.
  - **Option B (FALLBACK): stop-coordinator → relaunch-FRESH on the SAME output dir.** Use ONLY if
    the coordinator is dead/won't respawn, OR to deploy a CODE fix (child-only bounce reuses the
    old bundle). Verify the relaunch targets the SAME output-dir tag (step-0 start = wrong dir =
    lost progress).
- **Standing authority:** a CONFIRMED-wedged training run (frozen checkpoint mtime + frozen logs
  for hours, cgroup healthy) may be auto-bounced autonomously.

#### Daytona snapshot cap
Launches build a per-env Daytona snapshot; shared `cli` org caps at 60. On `SnapshotCapExceeded`,
delete **only `MISSING`-state `harbor__*` snapshots** (broken builds, safe). NEVER broad-prune
(`cleanup_unused_snapshots`) on the shared org — removes ACTIVE snapshots other jobs depend on.
Snippet in the `run-datagen-iris` skill.

#### Local-storage growth on the launch host
Daemon mirror `~/.ot-agent/runs/` (and `.iris-job.log`, 10s of MB) accumulates across jobs.
`python -m hpc.local_paths inventory` lists sizes; `... clean --older-than 30d --apply` purges old
runs. (Launch host; distinct from the 100 GB worker-node ceiling in §Launch.)

#### Empty GCS prefix after a "successful" job
Workload didn't route through UPath. Confirm `--harbor_extra_arg=--jobs-dir=<gcs>` is in the
submitted command (`iris job bug-report <id>`) and the harbor pin is the UPath-aware build.

#### Agentic eval result sinks
The standalone Marin `agentic_evals` package owns result publication. Configure
its durable-output/result-sink options at launch and inspect the recorded job
output prefix; do not assume OT-Agent's former GCS-only upload behavior applies.
  (+ `exception_stats`).
- **Traces to the Hub:** `gsutil rsync` the GCS job dir, then
  `scripts/harbor/make_and_upload_trace_dataset.py --episodes last --filter none --skip_register
  --chunk_size 300` (one row per trial; `--episodes all` explodes to per-step rows).

### Teardown

```bash
# Kill a job (ONLY with explicit user permission for a RUNNING/placed job)
$IRIS --cluster=marin job kill /benjaminfeuer/<job>
```
- **Rescue banked traces** before/after a kill if the repo didn't auto-create (resolve the recorded
  output prefix — never hardcode a bucket):
  `OUT=$(python -m hpc.iris.job_output_resolver <job> --cluster …/marin.yaml)` then
  `gsutil -m rsync -r "$OUT/<job>/" /tmp/<job>/` then
  `scripts/harbor/make_and_upload_trace_dataset.py --job_dir /tmp/<job> --repo_id penfever/<slug>-... --episodes last --filter none --skip_register`.
- **NEVER** `iris cluster restart` / stop / bounce the cluster without explicit user approval — it
  kills every running job. `job kill` is job-scoped (safe with permission); cluster ops are not.
  Killing the job frees its workers; iris reclaims preemptible workers.
- To stop a **CoreWeave** GPU job: `iris --cluster coreweave job stop /<user>/<job>` (full binary
  `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`; `which iris` fails); export
  `KUBECONFIG=~/.kube/coreweave-iris-gpu` first. A hard-kill ORPHANS in-flight Daytona sandboxes —
  reap them (see §2 Daytona).

### Authoritative references
- `marin:lib/iris/OPS.md` — cluster lifecycle, controller, SQL, GCP ops (read first).
- `marin:lib/iris/config/marin.yaml` — pools, variants, zones.
- `iris.cli.job.get_tpu_topology(variant)` — vm_count / chip topology (don't guess).
- `notes/marin/tech.md` — the OT-Agent↔iris fetch-daemon architecture + flag cheat sheet.

---

## §4 Ingress

How a Daytona sandbox reaches a co-located vLLM — for datagen / agentic-eval jobs where the agent
runs INSIDE a Daytona sandbox and must call the job's own co-located vLLM (RecordProxy → vLLM on
the iris worker). This is the durable topology + gotchas of the ingress plane; the launch flags
that select it live in the `datagen-launch-iris` skill.

### Current ingress = NATIVE `/proxy/t/*` capability-URL (pinggy retired 2026-07-06)

The Iris EndpointProxy fronts registered LINK endpoints. A Daytona sandbox must receive an
`https://iris.oa.dev/proxy/t/<JWT>/<endpoint>/v1` capability URL; its OpenAI key is only an inert
non-empty placeholder.

- **Marin-local serving (normal TPU datagen/eval):** `--ingress_mode controller --ingress_host https://iris.oa.dev`; the worker registers and mints on Marin, so the token issuer and public ingress match.
- **CoreWeave serving (RL or an external eval endpoint):** submit the serving job through the Marin meta-scheduler with `--target-cluster cw-us-east-02a` and forward the Marin login. Register on the peer, wait for FederationSync to mirror the endpoint on Marin, then mint at the Marin parent (`federated_capability_api_base`) and use `iris.oa.dev`. A CoreWeave peer token is rejected by the parent; its own `iris-cw-us-east-02a.oa.dev` host is IP-locked and unreachable from Daytona. Check the prerequisite with `iris --cluster=marin endpoints list <endpoint> --exact`: it must show a non-local peer before minting.
- **Never use a peer mint for a Daytona route:** `iris --cluster=cw-us-east-02a endpoints mint …` is only valid against that peer controller, not the Marin parent ingress.
- **Token TTL is re-minted per serve-spawn (24h).** ⚠️ The minted token has its own TTL, clamped
  server-side to `MAX_ENDPOINT_TOKEN_TTL_SECONDS`; the endpoint *registration* lease-renews but the
  **token does not**. If a job outlives the token TTL the sandbox's in-flight requests start
  returning 401 at the `/proxy` gate. Re-minting on each serve-spawn covers preempt-resumes; a single
  continuous serve longer than the TTL is the untested edge.
- **Route health check:**
  `curl -sk -w "%{http_code}" https://iris.oa.dev/proxy/t/badtoken/serve.nope/v1/models` → **401**
  (ingress up; bad token correctly rejected). A nonexistent endpoint with a VALID token returns
  `404 {"error":"No endpoint '<name>'"}` (`controller/endpoint_proxy.py`).
- **If a job fails specifically on `/proxy/t`** (401/403/base_url), do NOT redeploy pinggy — flag it.
  It has not recurred since cutover.

### ⛔ cw-rno2a has NO public endpoint-serving ingress → AGENTIC (opencode) workloads CANNOT reach the served model there (root-caused 2026-07-20)
**Non-agentic RL works on rno2a; agentic (opencode-in-Daytona) does NOT.** The opencode agent runs INSIDE a
Daytona sandbox (OUTSIDE the cluster VPC) and must reach the job's served vLLM via the public capability URL
`https://iris.oa.dev/proxy/t/<token>/otagent-<slug>/v1`. On **cw-rno2a** every agent turn silently hangs → the
`opencode.txt` transcript is **0 bytes** → `harbor.trial.errors.AgentTimeoutError` at the 1800 s ceiling → 0/N trials
finalize (no `result.json`), while the in-VPC RolloutCoordinator generates fine (it uses the internal head IP
`10.168.x:8010` directly, bypassing the proxy). **Root cause = a CLUSTER-PROVISIONING gap, not our launch flags:**
`marin/lib/iris/config/cw-us-east-02a.yaml` carries `provisioning.coreweave.ingress: {ingress_class: traefik,
acme_email, cluster_issuers: [letsencrypt-*]}` + `dashboard_url: https://iris-cw-us-east-02a.oa.dev` (added by the
Pulumi IaC `470ae1e7a4`), so east's controller serves `/proxy/t` capability URLs over a valid public HTTPS ingress
the parent `iris.oa.dev` forwards to. **`cw-rno2a.yaml` has NONE of that block** — it only got the IP-locked
*federation* ingress (`install_cw_network.py`, controller↔controller RPC; the per-cluster host
`iris-cw-rno2a.oa.dev` answers 403 = IP-locked to marin CIDRs, unreachable from a Daytona sandbox). So the parent has
no public rno2a endpoint-serving route to forward to. **Proof (A/B):** east keep-1 (`10.184.x`) + rno2a keep-6
(`10.168.x`) used the *identical* `iris.oa.dev/proxy/t` api_base — east opencode worked (`reward=1.0`,
`n_output_tokens=484`), rno2a produced empty transcripts. `iris.oa.dev/proxy/t/badtoken/...` → 401 (global ingress
UP), so the plane is fine; only rno2a's per-cluster serving ingress is missing.
- **Do NOT** try to fix this with launch flags, `--ingress-host`, a relaunch, or by pointing at `iris-cw-rno2a.oa.dev`
  (IP-locked). It is not our code/config — it is the rno2a controller's missing public serving ingress.
- **Unblock NOW:** run agentic RL/eval on **cw-us-east-02a** (proven GREEN). **Proper fix:** provision rno2a's
  controller with the `provisioning.coreweave.ingress` traefik/ACME block + `dashboard_url` (the Pulumi IaC east got)
  — a Marin cluster-admin change (file a marin issue; do NOT re-provision a live cluster unilaterally).

### Cutover provenance (marin #6847 / PR #6857, merged origin/main `b3df2573b`)
Native replacement for the pinggy sidecar: first-class per-endpoint auth-gated public ingress.
- **Endpoint access modes** — `EndpointAccess.ENDPOINT_ACCESS_{PRIVATE,PUBLIC,BEARER,…}`
  (`iris.cluster.types`); `PRIVATE=0` (unset ⇒ private). We register with **access=LINK**
  (capability-URL) — the token's `aud` is the endpoint name, `scope=proxy`, accepted ONLY at the
  `/proxy/<name>/…` gate; it carries **no RPC authority** and is minted under the owning user
  (surfaces in `iris key list`). Wiring: `hpc/ingress_utils.py`,
  `hpc/local_runner_utils.py::_serving_endpoint_meta`.

### Retired: pinggy sidecar (rollback lever only)
Before 2026-07-06 the public front was a standalone **non-preemptible CPU-only iris job**
`/benjaminfeuer/ingress-sidecar` (`hpc.ingress_sidecar` + a pinggy tunnel), in-VPC to the controller,
host `https://zksbejlvrn.a.pinggy.link`. Driver:
`scripts/inference/deploy_ingress_sidecar.py {deploy|ensure|status}` (idempotent: checks job state +
healthz, redeploys if down, re-emits `INGRESS_HOST=`). Retired via
`iris job stop /benjaminfeuer/ingress-sidecar`.
- **ROLLBACK (if native ever breaks):** redeploy the sidecar with `deploy_ingress_sidecar.py`, then
  flip the recipe back to `--ingress_host https://zksbejlvrn.a.pinggy.link`. No longer one-line.
- Env it used: `IRIS_INGRESS_API_KEY` (bearer; value in secrets.env),
  `CONTROLLER_PROXY_BASE=http://<controller-addr>:10000`, `IRIS_CONTROLLER_AUTH` unset (in-cluster
  controller view is unauthenticated).
- ⚠️ **pinggy DNS quirk (Mac-only):** from the launch Mac, plain `curl https://<id>.a.pinggy.link`
  returns 000 (ISP returns a bogus IP for `*.a.pinggy.link`). Health checks must resolve the real
  edge first:
  `EDGE=$(dig @1.1.1.1 +short <host> | tail -1); curl -sk --resolve "<host>:443:$EDGE" https://<host>/healthz`.
  Daytona sandboxes reach pinggy fine — quirk is Mac-only.

---

## §5 Tools

Inventory + when-to-reach-for-which index for **Iris** jobs (CoreWeave GPU cluster `cw-us-east-02a`
and Google TPU `marin` cluster). Preamble (env, kubeconfig, secrets) in §0.

**Canonical evidence layout.** `scripts/iris/iris_ops.py` validates job ids against Marin `main` and
maps every job to `/Users/benjaminfeuer/Documents/experiments/active/iris-job-bundles/jobs/<cluster>/<user>/<job>/`.
`watch_coreweave_rl.py` and `watch_iris_harbor.py` both refresh that bundle; completed analyzers
read it locally (or resync it), and the focused live tools write into the same path. Use
`coreweave_ops.py` for shared CoreWeave pod, Ray/vLLM, and object-store operations rather than
copying collection code into an analyzer.

### Tier 1 — everyday operational tools (monitoring + rollout inspection)

#### `iris_ops.py` — authoritative job-state watcher *(the liveness primitive)*
Polls the **authoritative iris job lifecycle state** (`iris job summary --json` → SQL `query`
fallback → `kubectl` pod cross-check), NOT log-string content — so it catches clean
kills/evictions/preemptions/early crashes that emit no terminal log line.
- **Use:** every liveness/terminal check; importable as the watch primitive (`get_job_state()`
  returns a `JobStateSnapshot`; `watch()` runs the loop, returns the terminal snapshot).
- **CLI:** `iris_ops.py <job_id> [--cluster cw-us-east-02a] [--interval 60] [--once]
  [--no-pods] [--max-polls N] [--json]`.
- **Exit codes:** `0` succeeded · `1` failed/killed/worker_failed/unschedulable · `2` absent from
  controller AND 0 pods (disappeared) · `3` watch error.
- **Parse gotcha:** with `--json --once` it prints a human `[HH:MM:SS] … state=running …` line
  *before* the JSON object — a naive `json.load(stdin)` chokes on both. Parse the human line
  (`grep -oE "state=[a-z]+"`) or strip to the JSON braces.
- **JobState enum:** `0` UNSPECIFIED `1` PENDING `2` BUILDING `3` RUNNING `4` SUCCEEDED `5` FAILED
  `6` KILLED `7` WORKER_FAILED `8` UNSCHEDULABLE.
- **Why this over a hand-rolled `SELECT state … WHERE state IN (4,5,6)` watcher:** the `jobs` table
  is PRUNED, so a completed job's `query` returns an EMPTY row and a naive terminal-state watcher
  polls forever. `iris_ops.py` treats **absence-after-existence + 0 pods** as terminal
  (`absent`, exit 2). Needs a correct `KUBECONFIG` to confirm 0 pods. Full rule + the artifact-watch
  pattern for HF-export/s3 jobs: §2 Observability "Empty job-state ≠ still-running".

#### `analyze_iris_harbor_job.py` — full-log pull + throughput/preemption stats *(the science tool)*
Paginates the ENTIRE job log by fixed time windows (`--since-ms` + `--no-tail`, the only way past
`--tail`'s line cap) and filters at the python level to retain just the signal (cycle boundaries,
vLLM throughput emissions), caching the filtered stream to `/tmp/iris_history_<job>.filtered.log`.
Emits a markdown report + JSON sidecar with **§1** preemption count + time-to-preempt, **§2** trace
progress per cycle (from harbor GCS output), **§3** serving throughput (full + warmup-excluded).
- **Use:** sel_rows / EPDIag / throughput **science only** — *never* for liveness/terminal detection
  (that's `iris_ops.py`). Also the way to recover a dead run's root cause from the full log.
- **CLI:** `analyze_iris_harbor_job.py <job_id> --output <report.md> [--resync] [--warmup-seconds 180]
  [--cluster …] [--iris-bin …] [--gsutil-sample …]`. Auto-resolves the cw-capable iris binary
  (`resolve_iris_bin()`: `$IRIS_BIN` → PATH → otagent env → marin `.venv` last).
- **⚠ CoreWeave (`--cluster cw-us-east-02a`) needs R2 archive creds.** The archive half reads
  `s3://marin-na/finelog/cw-us-east-02a` (R2) — the Mac lacks the creds, so the run crashes
  `FileNotFoundError: The specified bucket does not exist` unless you first source them from the
  `iris`-ns secret. **Full procedure: §6.** (The live half on cw uses a k8s tunnel, NOT IAP — so the
  `analyze-job-history-iris` skill's `marin-login` step is marin/TPU-only.)

#### `analyze_coreweave_rl_job_live.sh` — inspect / capture a running CoreWeave RL job's Harbor rollout artifacts
Reaches the **rank-0 pod** of a running agentic-RL job and reads its `trace_jobs` (per-trial
trajectory + prompts/responses + `verifier_output` + `result.json` reward). The jobs use a **remote
object-store `trials_dir`** (`s3://marin-us-east-02a/iris/<job>/trace_jobs`, durable) whose creds
live only in the pod (the launch-host Mac lacks cluster creds), so **all object-store ops run INSIDE
the pod via boto3**. (`finelog` archive `s3://marin-na/finelog/…` is a SEPARATE marin-controlled
location.) `result.json` is the COMPLETED-trial marker (carries the reward) → its count is the real
"how many trials finished".
- **Use:** "is the rollout buffer actually filling / what rewards are coming back / pull the full
  trace bundle for analysis".
- **Subcommands:** `<pod-substr>` (summary: started + completed + breakdown) · `ls [glob]` · `cat
  <trial-dir>` (dump a trial's json) · `grep <pattern>` · `cp <trial-dir> [dest]` · `pull [bundle-root]`
  (FULL CAPTURE → the canonical `<bundle-root>/jobs/<cluster>/<user>/<job>/` directory: finelog +
  per-rank pod logs + all `trace_jobs` synced from the CW object store `s3://marin-us-east-02a` +
  `manifest.json`).
- **`<substr>` matches the POD name** (`iris-benjaminfeuer-<name>-<rank>-<hash>-0`), which can differ
  from the iris job_id display name; no match → lists candidate RL pods.
- **Env:** `PEEK_KUBECONFIG` (default `~/.kube/coreweave-iris-gpu`), `NS`/`CONTAINER`, `PEEK_CLUSTER`,
  `IRIS_BIN`, `PEEK_OUT`, `PEEK_TRIALS_S3`, `PEEK_MAX_OBJECT_BYTES` (pull skip-size, default 20 MB;
  `0` = fetch everything). Forces the cw kubeconfig — ignores an inherited `$KUBECONFIG`.

### Tier 2 — RL runtime (load-bearing; you don't invoke it by hand)

#### `start_rl_iris_controller.py` — the per-node multi-node RL bootstrap
Canonical copy lives in **MarinSkyRL `cloud/iris/start_rl_iris_controller.py`** (invoked by
`python -m cloud.iris.launch_rl_iris`). iris runs **this same entrypoint on every node** of a gang
(injecting `IRIS_TASK_ID`/`IRIS_NUM_TASKS`/`IRIS_ADVERTISE_HOST` per task). It bootstraps ONE
cross-node Ray cluster: **rank 0** `ray start --head` → publishes head IP to the rendezvous file →
waits for all nodes to join → `exec`s the MarinSkyRL driver with `RAY_ADDRESS` set; **ranks 1..N-1**
read the head IP from the rendezvous, `ray start --address=…`, contribute their 8 H100s, and block
until rank 0 writes the `done` marker.
- **Rendezvous:** `ray_head.json` / `ray_head.done` under `--rendezvous-dir`
  (`OT_AGENT_IRIS_RENDEZVOUS_DIR`); opened via `fsspec` so `gs://` / `s3://` (CoreWeave CW object
  store `marin-us-east-02a`) / NFS all work. Pins ALL Ray agent ports outside the worker range (fixes
  the nondeterministic port-collision).
- **Invoked by** the RL launcher — you never type it directly; edit it in MarinSkyRL locally (rides
  the `/app` upload, no image rebuild). (The old OT-Agent copy `scripts/iris/start_rl_iris_controller.py`
  and its one-shot MoE/EP bring-up probes have been REMOVED — the MarinSkyRL port is the sole home;
  author new bring-up probes against `cloud/iris/` in MarinSkyRL.)

### Tier 3 — TPU-cluster data plumbing (the `marin` cluster, NOT CoreWeave)

Weight-mirroring helpers for the **Google TPU** Iris (`marin`) — staging model weights between HF,
GCS, and the LAION/Jülich S3 so vLLM's `runai_streamer` (needs real S3 + GCS HMAC keys it doesn't
have) can read them. Each is a `mirror_*` worker + a `launch_*` iris-job submitter.

| Script | Direction | Notes |
|---|---|---|
| `mirror_hf_to_gcs.py` | HF repo → `gs://marin-eu-west4/ot-agent/models/` | One shard at a time (download→upload→delete), so it doesn't need the full model on disk; idempotent (size-skip), resumable. |
| `launch_hf_mirror.py` | submits `mirror_hf_to_gcs.py` as an iris job | Marin has no CPU-only pool → runs on the smallest TPU slice (v6e-4), TPU idle; one-shot ~30–60 min, don't queue against a busy cluster. |
| `mirror_gcs_to_s3.py` | GCS prefix → S3 (e.g. LAION `mmlaion` @ Jülich) | Streaming gcsfs→boto3, one file at a time; endpoint from `$AWS_ENDPOINT_URL` or `--s3-endpoint`; idempotent. Workaround for missing GCS HMAC keys `runai_streamer` requires. |
| `launch_gcs_to_s3.py` | submits `mirror_gcs_to_s3.py` as an iris job | Companion to `launch_hf_mirror.py`, opposite direction. |

#### `patch_tpu_inference.py` — runtime patches to the TPU worker's `tpu-inference`
Invoked from the TPU launcher's bash bootstrap **after `uv sync`, before the workload**. Each patch
is idempotent + prints a one-line status. Currently: the `hbm_usage_bytes()` non-addressable-device
skip (guards `device.memory_stats()` on multi-host slices >v6e-8 where non-local chips raise
`INVALID_ARGUMENT`).

---

## §6 Credentials — R2 finelog archive (CoreWeave)

> **TL;DR.** `analyze_iris_harbor_job.py` (and anything reading the finelog **archive** half) needs **R2
> credentials** to list/read `s3://marin-na/finelog/cw-us-east-02a`. The Mac does **not** have them;
> they live only in the cluster's `iris`-namespace secret **`finelog-cw-use02a-env`** (`AWS_*` keys
> incl. `AWS_ENDPOINT_URL`). Source that secret into the env — **values never printed** — before
> running the analyzer against `--cluster cw-us-east-02a`. Without it the run crashes
> `FileNotFoundError: The specified bucket does not exist` (s3fs falls back to **real AWS S3**,
> where `marin-na` doesn't exist).
>
> The finelog **archive** is a SEPARATE marin-controlled location that stays on R2
> (`s3://marin-na/finelog/…`), read Mac-side with these R2 creds — distinct from the RL/eval
> **write-path** (`s3://marin-us-east-02a`). Don't "repoint" the finelog archive to CW.

### Why this bites (the failure mode)

The analyzer fetches each job's complete log as **live ∪ GCS-archive**, deduped on `seq`. The
archive half is where the cred gap is:

| cluster | finelog `client_url` | LIVE half | ARCHIVE half (`remote_log_dir`) | archive creds |
|---|---|---|---|---|
| `marin` (TPU) | set | IAP proxy (`marin-login login marin`) | `gs://…` (GCS) | the IAP/ADC session covers it |
| `cw-us-east-02a` (CoreWeave) | **None** | **k8s tunnel** (no IAP needed) | **`s3://marin-na/finelog/cw-us-east-02a`** (**R2**) | **R2 creds — NOT on the Mac** |

On CoreWeave the live half is *easier* (a tunnel, no IAP login), but the archive half needs R2
creds the Mac lacks. `fsspec.url_to_fs("s3://marin-na/…")` with no R2 endpoint/creds resolves to
**AWS S3**, where the bucket isn't there → the run aborts in `_list_namespace_segments → fs.ls`
**before** duckdb reads, so it is *not* caught by the script's compaction-race retry (that only
catches 404/NoSuchKey mid-read). Hard `FileNotFoundError`. Same Mac-lacks-marin-na-R2-creds fact
noted for `trials_dir` in §5; for finelog work around from the Mac by borrowing the pod's creds out
of the k8s secret.

### Where the creds are (and the var-name trap)

- **Secret:** `finelog-cw-use02a-env` in the **`iris`** namespace (the finelog deployment's env). The
  sibling `iris-task-env` carries the same `AWS_*` set (what every task pod gets) plus a `FSSPEC_S3`
  key — either works; prefer `finelog-cw-use02a-env`.
- **Keys (NAMES only):** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`,
  `AWS_REGION`, `AWS_DEFAULT_REGION`. The `AWS_ENDPOINT_URL` value is the CoreWeave R2 endpoint
  `https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com` (also non-secretly present as
  `object_storage_endpoint` in `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`).
- **⚠ Var-name trap.** The cluster-config header *comment* documents the creds as `R2_ACCESS_KEY_ID`
  / `R2_SECRET_ACCESS_KEY`, but **botocore/s3fs read `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL`** — which is exactly how the secret stores them. Use
  the `AWS_*` names; modern botocore picks up `AWS_ENDPOINT_URL` automatically, so **no** explicit
  `endpoint_url`/`FSSPEC_S3_ENDPOINT_URL`/`client_kwargs` is needed for the listing to work.
- **Do NOT** confuse these with the LAION `AWS_*` in the secrets env (`.claude/secret.md`; a
  *different* S3 store, `LAION_ENDPOINT`). Source the R2 creds in a shell where you have **not** also
  sourced the LAION secrets, or the LAION values clobber R2 and you're back to "bucket does not exist."

### The plumbing (values never printed)

Source the secret into the env via a base64-decode loop that pipes straight into `export` (credential
values stay out of stdout / scrollback / logs):

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu              # HARD prereq (cw kubeconfig)
# Borrow the pod's R2 creds out of the iris-ns secret — values are decoded inline, never echoed:
while IFS=$'\t' read -r k v; do export "$k=$(printf %s "$v" | base64 -d)"; done \
  < <(kubectl -n iris get secret finelog-cw-use02a-env \
        -o go-template='{{range $k,$v := .data}}{{$k}}{{"\t"}}{{$v}}{{"\n"}}{{end}}')
# sanity (prints only yes/no, never the value):
echo "R2 endpoint set? $([ -n "$AWS_ENDPOINT_URL" ] && echo yes || echo no)"
```

Run the analyzer under the **marin venv** (must import `finelog`/`rigging`/`duckdb`) with the
otagent-env iris binary (the marin `.venv` iris can't drive CoreWeave):

```bash
IRIS_BIN=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris \
/Users/benjaminfeuer/Documents/marin/.venv/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_iris_harbor_job.py \
  /benjaminfeuer/<job> --cluster cw-us-east-02a \
  --output /tmp/<job>_history.md --resync
```

Isolated check that the creds are wired (lists the archive root in seconds):

```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/python -c "
import fsspec; from finelog.deploy.config import load_finelog_config
cfg=load_finelog_config('cw-us-east-02a'); fs,_=fsspec.url_to_fs(cfg.remote_log_dir)
print('LS OK:', len(fs.ls(cfg.remote_log_dir, detail=False)), 'entries')"
# expect: LS OK: 4 entries  (iris.task_status, log, zephyr.stage, ...)
```

### Caveats / gotchas

- **Terminal vs running jobs.** For a **terminal** (old) job the live tunnel often has nothing left
  — archive (R2) is the real source, R2 creds mandatory. For a **running** job you need *both*
  (live tunnel for the recent L0 tail + R2 for compacted history); a live-half failure surfaces as
  a loud coverage gap, not a silent fragment.
- **GPU-RL jobs have no harbor trial sidecars**, so `analyze_iris_harbor_job.py` §2 is empty — for
  GPU-RL diagnosis use **rl-job-health-deep-dive**. The log-acquisition machinery here (live ∪
  R2-archive) is generic for terminal RL run's full finelog history: the default
  `FINELOG_CONTAINS_PATTERNS` filter is TPU/datagen-tuned, so to capture RL signals
  (`WORKER_FORWARD_ENTER`, `global_step`, `[weight-sync]`, mesh_fsdp watchdog) swap those
  `contains(data, …)` patterns when reusing `fetch_live`/`fetch_gcs`.
- **Secret hygiene.** Shared marin-infra R2 creds. Source by the loop above — never paste a value
  into a prompt/file/chat; a subagent that needs them gets *this procedure*, not the values. They
  don't belong in `secrets.env` — borrow from the live secret each time so a rotation can't leave
  a stale copy on disk.

### Cross-references
- **Skill:** `analyze-job-history-iris` (the analyzer how-to + sidecar parsing; this doc supplies its
  missing CoreWeave-archive cred step).
- **Cluster config:** `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`
  (`object_storage_endpoint`, `remote_state_dir`, the `R2_*` header comment).
- **GPU-RL diagnosis:** `rl-job-health-deep-dive`.
