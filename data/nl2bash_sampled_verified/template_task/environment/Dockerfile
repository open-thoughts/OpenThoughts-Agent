FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        bzip2 \
        ca-certificates \
        coreutils \
        cron \
        curl \
        file \
        findutils \
        gawk \
        git \
        gnupg \
        grep \
        gzip \
        jq \
        less \
        passwd \
        procps \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        sed \
        tar \
        unzip \
        vim \
        wget \
        xz-utils && \
    rm -rf /var/lib/apt/lists/*

# Create /output directory for task outputs
RUN mkdir -p /output

COPY seeds/ /workspace/
