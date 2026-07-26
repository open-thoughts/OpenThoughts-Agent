#!/usr/bin/env bash
# build_gpu_glm52_kaniko.sh — in-cluster kaniko build of the :gpu-glm52 image.
#
# Cloned from build_tpu_kaniko.sh, changing only the Dockerfile + cache/destination repos.
# Dockerfile.gpu-glm52 is the gpu-8x datagen image with vLLM swapped to the marin-community
# fork (main) so it serves GLM-5.2 (glm_moe_dsa). Pushes ONLY the immutable :gpu-glm52-<gitsha>
# tag — no floating tag — so a botched build cannot touch the :gpu-8x campaign (#104).
#
# Runs INSIDE an iris job whose task-image is docker.io/library/ubuntu:22.04 (kaniko's executor
# is distroless / no bash). We crane-export the kaniko executor rootfs over / and run it.
#
# Required env (passed by the iris launch as -e):
#   DOCKER_USER_ID  ghcr user (penfever)
#   DOCKER_TOKEN    a GitHub PAT with write:packages (from `gh auth token`; NOT the Docker Hub
#                   dckr_pat_ in secrets.env)
#   GITSHA          OT-Agent commit sha for the immutable :gpu-glm52-<gitsha> tag
set -euxo pipefail

: "${DOCKER_USER_ID:?}"
: "${DOCKER_TOKEN:?}"
: "${GITSHA:?}"

# SINGLE_SNAPSHOT=0 (per-instruction layers) is the DEFAULT: it dodges the un-pullable ~16 GB
# single-blob layer problem (containerd restarts the single-blob GET from 0 over the ghcr egress
# and dies), giving small independently-retriable layers.
SINGLE_SNAPSHOT="${SINGLE_SNAPSHOT:-0}"
if [ "$SINGLE_SNAPSHOT" = "1" ]; then SNAPSHOT_FLAG="--single-snapshot"; else SNAPSHOT_FLAG=""; fi

DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.gpu-glm52}"
IMAGE_TAG_PREFIX="${IMAGE_TAG_PREFIX:-gpu-glm52}"
CACHE_REPO=ghcr.io/open-thoughts/openthoughts-agent/cache-glm52
DEST_PINNED="ghcr.io/open-thoughts/openthoughts-agent:${IMAGE_TAG_PREFIX}-${GITSHA}"

# --- 1. fetch crane (static binary) ---
apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl tar
cd /tmp
CRANE_VER=v0.20.2
curl -fsSL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane

# --- 2. crane-export the kaniko executor rootfs over / ---
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

# --- 3. write the ghcr auth config AFTER the overlay (kaniko clobbers /kaniko otherwise) ---
export DOCKER_CONFIG=/kaniko/.docker
mkdir -p "$DOCKER_CONFIG"
{ set +x; } 2>/dev/null
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$DOCKER_TOKEN" | base64 | tr -d '\n')
cat > "$DOCKER_CONFIG/config.json" <<EOF
{"auths":{"ghcr.io":{"auth":"${AUTH}"}}}
EOF
unset AUTH
set -x

# --- 4. run kaniko (pinned tag ONLY) ---
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile "${DOCKERFILE}" \
  --skip-unused-stages \
  $SNAPSHOT_FLAG \
  --compressed-caching=false \
  --cache=true \
  --cache-repo="${CACHE_REPO}" \
  --destination "${DEST_PINNED}"
