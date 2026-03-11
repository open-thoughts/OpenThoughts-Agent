FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install openai --break-system-packages

COPY counterpart.py /app/counterpart.py
