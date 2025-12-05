#!/usr/bin/env bash
set -euo pipefail

SIF_PATH="cuda-cudnn.sif"
CONTAINER_REF="docker://nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04"
OVERLAY_PATH="cuda-overlay.img"
OVERLAY_SIZE_MB=2048
FILESYSTEM_TYPE="ext3"
FORCE_PULL=false
FORCE_OVERLAY=false
ENTER_SHELL=false
USE_FAKEROOT=false
ENABLE_NV=true
APT_PACKAGES=""
PIP_PACKAGES=""
PYTHON_SETUP_PATH=""
CREATE_DIRS=()
BIND_PATHS=()

usage() {
  cat <<'EOF'
create-cuda-overlay.sh - Prepare a writable overlay for a CUDA Singularity/Apptainer image.

Usage:
  create-cuda-overlay.sh [options]

Options:
  -s, --size-mb <int>         Overlay size in MiB (default: 2048).
  -o, --overlay <path>        Overlay image path (default: cuda-overlay.img).
  -i, --sif <path>            Singularity image path (default: cuda-cudnn.sif).
  -r, --ref <uri>             Container URI to pull (default: docker://nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04).
      --fs <type>             Overlay filesystem type (default: ext3; skipped if unsupported).
      --create-dir <path>     Create directory inside overlay (repeatable).
      --bind <src:dst>        Bind path when running exec/shell (repeatable).
      --fakeroot              Use --fakeroot when executing inside the container.
      --no-nv                 Do not pass --nv to Singularity.
      --force-pull            Re-pull the SIF even if it already exists.
      --force-overlay         Recreate the overlay even if it already exists.
      --shell                 Drop into an interactive shell once setup completes.
      --apt "<pkgs>"          Install space-separated apt packages inside the overlay.
      --pip "<pkgs>"          Install space-separated pip packages inside the overlay.
      --python-setup <path>   Run python setup.py install in the specified project directory.
  -h, --help                  Show this message and exit.

Examples:
  create-cuda-overlay.sh --size-mb 4096 --apt "build-essential git" --pip "ninja cmake" --fakeroot
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_command() {
  local cmd=$1
  command -v "$cmd" >/dev/null 2>&1 || die "Required command '$cmd' not found in PATH."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--size-mb)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      OVERLAY_SIZE_MB=$2
      shift 2
      ;;
    -o|--overlay)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      OVERLAY_PATH=$2
      shift 2
      ;;
    -i|--sif)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      SIF_PATH=$2
      shift 2
      ;;
    -r|--ref)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      CONTAINER_REF=$2
      shift 2
      ;;
    --fs)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      FILESYSTEM_TYPE=$2
      shift 2
      ;;
    --create-dir)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      CREATE_DIRS+=("$2")
      shift 2
      ;;
    --bind)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      BIND_PATHS+=("$2")
      shift 2
      ;;
    --fakeroot)
      USE_FAKEROOT=true
      shift
      ;;
    --no-nv)
      ENABLE_NV=false
      shift
      ;;
    --force-pull)
      FORCE_PULL=true
      shift
      ;;
    --force-overlay)
      FORCE_OVERLAY=true
      shift
      ;;
    --shell)
      ENTER_SHELL=true
      shift
      ;;
    --apt)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      APT_PACKAGES=$2
      shift 2
      ;;
    --pip)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      PIP_PACKAGES=$2
      shift 2
      ;;
    --python-setup)
      [[ $# -lt 2 ]] && die "Missing argument for $1"
      PYTHON_SETUP_PATH=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ "$OVERLAY_SIZE_MB" =~ ^[0-9]+$ ]] || die "Overlay size must be an integer (MiB)."
(( OVERLAY_SIZE_MB > 0 )) || die "Overlay size must be greater than zero."

if [[ -n "$PYTHON_SETUP_PATH" && ! -d "$PYTHON_SETUP_PATH" ]]; then
  die "python-setup path '$PYTHON_SETUP_PATH' does not exist or is not a directory."
fi

if command -v singularity >/dev/null 2>&1; then
  SING_CLI=singularity
elif command -v apptainer >/dev/null 2>&1; then
  SING_CLI=apptainer
else
  die "Neither 'singularity' nor 'apptainer' was found in PATH."
fi

log "Using container CLI: $SING_CLI"
OVERLAY_CREATE_HELP=$("$SING_CLI" overlay create --help 2>&1 || true)

pull_image() {
  if [[ -f "$SIF_PATH" && $FORCE_PULL == false ]]; then
    log "SIF image $SIF_PATH already exists; skipping pull. Use --force-pull to re-download."
    return
  fi

  log "Pulling container: $CONTAINER_REF -> $SIF_PATH"
  "$SING_CLI" pull "$SIF_PATH" "$CONTAINER_REF"
}

ensure_fakeroot_ready() {
  $USE_FAKEROOT || return
  if ! "$SING_CLI" exec --fakeroot "$SIF_PATH" true >/dev/null 2>&1; then
    die "Fakeroot support appears unavailable on this system. Contact your HPC admins to enable fakeroot (subuid/subgid entries) or omit --fakeroot and skip apt installs."
  fi
}

create_overlay() {
  if [[ -f "$OVERLAY_PATH" && $FORCE_OVERLAY == false ]]; then
    log "Overlay $OVERLAY_PATH already exists; skipping creation. Use --force-overlay to recreate."
    return
  fi

  local overlay_opts=(--size "$OVERLAY_SIZE_MB")

  if [[ -n "$FILESYSTEM_TYPE" ]]; then
    if grep -q -- '--filesystem' <<<"$OVERLAY_CREATE_HELP"; then
      overlay_opts+=(--filesystem "$FILESYSTEM_TYPE")
    else
      log "Filesystem selection not supported; continuing with CLI default."
    fi
  fi

  for dir in "${CREATE_DIRS[@]}"; do
    overlay_opts+=(--create-dir "$dir")
  done
  if $USE_FAKEROOT && grep -q -- '--fakeroot' <<<"$OVERLAY_CREATE_HELP"; then
    overlay_opts+=(--fakeroot)
  fi

  log "Creating overlay $OVERLAY_PATH (${OVERLAY_SIZE_MB} MiB)"
  "$SING_CLI" overlay create "${overlay_opts[@]}" "$OVERLAY_PATH"
}

build_common_flags() {
  COMMON_FLAGS=()
  if $ENABLE_NV; then
    COMMON_FLAGS+=(--nv)
  fi
  if $USE_FAKEROOT; then
    COMMON_FLAGS+=(--fakeroot)
  fi
  for bind in "${BIND_PATHS[@]}"; do
    COMMON_FLAGS+=(--bind "$bind")
  done
  COMMON_FLAGS+=(--overlay "$OVERLAY_PATH")
}

run_apt() {
  [[ -z "$APT_PACKAGES" ]] && return
  $USE_FAKEROOT || die "Installing apt packages requires --fakeroot."

  build_common_flags

  log "Installing apt packages inside the overlay: $APT_PACKAGES"
  read -r -a apt_array <<<"$APT_PACKAGES"
  "$SING_CLI" exec "${COMMON_FLAGS[@]}" "$SIF_PATH" \
    bash -eu -c 'set -euo pipefail; apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"' _ \
    "${apt_array[@]}"
}

run_pip() {
  [[ -z "$PIP_PACKAGES" ]] && return

  build_common_flags

  log "Installing pip packages inside the overlay: $PIP_PACKAGES"
  read -r -a pip_array <<<"$PIP_PACKAGES"
  "$SING_CLI" exec "${COMMON_FLAGS[@]}" "$SIF_PATH" \
    bash -eu -c 'set -euo pipefail; pip install --upgrade pip; pip install "$@"' _ \
    "${pip_array[@]}"
}

run_python_setup() {
  [[ -z "$PYTHON_SETUP_PATH" ]] && return

  build_common_flags

  log "Running python setup.py install in $PYTHON_SETUP_PATH"
  "$SING_CLI" exec "${COMMON_FLAGS[@]}" "$SIF_PATH" \
    bash -eu -c 'set -euo pipefail; cd "$1"; python setup.py install' _ "$PYTHON_SETUP_PATH"
}

open_shell() {
  $ENTER_SHELL || return

  build_common_flags
  log "Opening interactive shell with overlay mounted."
  "$SING_CLI" shell "${COMMON_FLAGS[@]}" "$SIF_PATH"
}

pull_image
ensure_fakeroot_ready
create_overlay
run_apt
run_pip
run_python_setup
open_shell

log "Overlay setup complete. Next steps:"
if $ENABLE_NV; then
  NV_HINT="--nv "
else
  NV_HINT=""
fi
log "  - Edit overlay contents with: $SING_CLI exec ${NV_HINT}--overlay $OVERLAY_PATH $SIF_PATH <command>"
log "  - Drop into a shell anytime with: $SING_CLI shell ${NV_HINT}--overlay $OVERLAY_PATH $SIF_PATH"
