#!/bin/bash

# Terminal-Bench Container and Process Cleanup Script
# Run this after interrupting evaluation runs to clean up orphaned containers and processes

echo "=== Cleaning up Terminal-Bench processes ==="

echo "Killing orphaned Terminal-Bench Python processes..."
# Kill main run_terminal_bench.py processes
pkill -f "run_terminal_bench.py" 2>/dev/null || true

# Kill any terminal-bench related Python processes
pkill -f "terminal_bench" 2>/dev/null || true
pkill -f "terminal-bench" 2>/dev/null || true

# Kill Ray processes if they exist
echo "Cleaning up Ray processes..."
pkill -f "ray::" 2>/dev/null || true
pkill -f "_raylet" 2>/dev/null || true
ray stop 2>/dev/null || true

# Kill any litellm related processes
echo "Cleaning up LiteLLM processes..."
pkill -f "litellm" 2>/dev/null || true

# Kill any multiprocessing worker processes
echo "Cleaning up multiprocessing workers..."
pkill -f "multiprocessing" 2>/dev/null || true

echo "=== Cleaning up Docker containers ==="

echo "Stopping running Terminal-Bench containers..."
docker stop $(docker ps -q --filter "label=com.docker.compose.project" | grep -v "^$") 2>/dev/null || true

echo "Removing all Terminal-Bench containers..."
docker rm $(docker ps -aq --filter "label=com.docker.compose.project" | grep -v "^$") 2>/dev/null || true

echo "Removing Terminal-Bench images (optional - uncomment if needed)..."
# docker rmi $(docker images --filter "label=com.docker.compose.project" -q | grep -v "^$") 2>/dev/null || true

echo "Cleaning up Docker networks..."
docker network prune -f

echo "Cleaning up Docker volumes..."
docker volume prune -f

echo "=== Cleanup complete! ==="

# Show remaining processes (for verification)
echo "Remaining terminal-bench processes (if any):"
ps aux | grep -i "terminal.bench\|run_terminal_bench\|ray::\|litellm" | grep -v grep || echo "None found"