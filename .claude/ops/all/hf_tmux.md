## HF Uploads + Long-Running Login-Node Commands

Two cross-cluster rules apply to any cleanup-step upload (RL, 8B SFT, 32B SFT, datagen, eval) and to anything else that detaches from your shell on a login node.

### Always `hf upload`, never `hf upload-large-folder`

`hf upload-large-folder` looks like the right tool on paper (parallel pre-upload workers, resumable cache, no-bars mode), but in practice it does not play nicely with our clusters' network paths or with HF Hub's LFS rate-limiting. Observed failure mode on Jupiter for a 131 GB 32B model:

- 42/42 files hashed locally in ~15 min ✓
- 0/28 pre-uploaded after 8h 17m of elapsed wall time
- Stuck in a `HTTP 429 → "Rate limited. Waiting 286.0s before retry [Retry 1/5]"` loop on `.git/info/lfs/objects/batch`. 28 pre-upload workers all hit 429 in parallel; the per-file retry budget cycles forever without ever committing a single LFS object.

`hf upload` (sequential, 3-arg form) does commit. Use it for every upload step:

```bash
# Folder-to-repo-root upload pattern (replaces `huggingface-cli upload-large-folder`)
hf upload <repo> <local-folder> . --repo-type=model

# Single-file pattern
hf upload <repo> <local-file> <path-in-repo> --repo-type=model
```

The old `huggingface-cli upload-large-folder` is now a deprecation stub on
recent huggingface_hub — it prints a hint and exits without uploading. Mentally
translate `huggingface-cli upload-large-folder REPO DIR --repo-type=model` →
`hf upload REPO DIR . --repo-type=model` everywhere in the checklists below.

### Always `tmux`, never `nohup` / `disown`

For any long-running command launched on a login node (HF uploads, eval listeners, datagen pipelines that run on the login node before sbatch submit, anything that needs to outlive an SSH session), wrap it in a detached `tmux` session — not `nohup ... &` / `disown`.

`tmux` advantages over `nohup`:
- Survives SSH disconnects more robustly (Leonardo's login-node killer takes down `nohup`/`disown` processes at ~100 s; tmux survives much longer).
- You can `tmux attach -t <session>` later to see live state.
- Output preserved in tmux scrollback even if you don't redirect to a log.
- Restartable from a single named anchor (`tmux kill-session -t <name>`; `tmux new-session -d -s <name> "<cmd>"`).

Pattern:

```bash
# Detached, named, output mirrored to a log via tee
tmux new-session -d -s <session_name> \
    "source ~/secrets.env && <command> 2>&1 | tee -a <log_path>"

# Inspect live:
ssh <cluster>
tmux attach -t <session_name>     # Ctrl-b d to detach
tmux ls | grep <session_name>     # liveness check

# Kill:
tmux kill-session -t <session_name>
```

---

## HF org / target / credentials

### Default to PUBLIC uploads; `laion/` is the canonical target
- **Default `--private false`** on all RL/SFT/trace uploads. The `laion/` org's **private** quota is consistently exhausted in practice → private uploads hit a quota error and need a manual flip-to-public retry. Public has headroom. (`hf upload` itself defaults to private repo creation, so pass `--private false` explicitly, or `create_repo(..., private=False)` first.) Note `--private` is a **no-value flag** — `--private false` is a CLI parse error in some `hf` versions; prefer creating the repo public first. Only upload private for genuinely sensitive weights/data.
- **`laion/` is the canonical org** for any non-trivial model upload (consolidated 32B SFT, RL exports, etc.) — it has the storage budget. Treat any `hub_model_id: mlfoundations-dev/...` leaking in from old `train_config` templates as an override to ignore.

### A persistent 429 on the LFS batch endpoint = org storage quota, NOT rate-limiting
When an upload fails with **HTTP 429 on `.../<repo>.git/info/lfs/objects/batch`** that never clears
(`hf upload-large-folder` spins "Rate limited. Waiting 286.0s before retry" forever), it's almost
certainly the target org being **out of storage** — HF masks the 403 quota error as 429 to
`upload-large-folder`. **Diagnose:** abort and run `hf upload <repo> <dir> . --repo-type=model` once — it
surfaces the real error within seconds (e.g. `403 Forbidden: You have exceeded your public storage space`).
Then switch destination to `laion/`. (Confirmed 2026-05-17 on a 32B SFT upload that spun 8h17m with zero
files committed.)

### `create_repo` 403 under `laion/` = org-membership role, NOT the token
A `403 "You don't have the rights to create a model under the namespace laion"` is an **org role** issue,
not a token/quota one — **reissuing a personal token does NOT fix it** (every penfever token inherits the
same `roleInOrg`). Only a laion admin bumping the account `read → write` grants create rights (then ALL
tokens work, no secrets.env change). **Pushes to PRE-EXISTING `laion/` repos still work** with `read`
(repo-level write) — only *creating* a new repo needs org write. Workarounds if it 403s on create: get the
role bumped / have a write-capable account pre-create the repos / publish to `penfever/` and re-home later.
Diagnostic (otagent python): `from huggingface_hub import whoami; [print(o['roleInOrg']) for o in whoami(token=…)['orgs'] if o['name']=='laion']`. (penfever was `read` 2026-06-11, **bumped to `write` same day** — keep this for the diagnostic + history.)

### secrets.env paths
- **Local Mac:** `/Users/benjaminfeuer/Documents/secrets.env` (NOT `~/secrets.env`). Load with `set -a; source /Users/benjaminfeuer/Documents/secrets.env; set +a` so a Python subprocess inherits the vars (used for local Supabase/HF/Daytona queries — faster than ssh+cluster python).
- **All clusters** (Jupiter, Perlmutter, Leonardo, NYU Torch): `~/secrets.env` — different convention from the Mac.