FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        unzip \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*

COPY tooling /workspace/tooling
RUN chmod -R a+rx /workspace/tooling
