# vLLM Gateway

A load-balancing gateway for multiple vLLM OpenAI-compatible servers running on SLURM cluster nodes.

## Overview

This project provides a FastAPI-based gateway that distributes requests across multiple vLLM servers using round-robin load balancing. It's designed to work with SLURM-managed GPU nodes in a cluster environment.

## Architecture

- **Gateway** (`gateway.py`): FastAPI application that proxies OpenAI-compatible API requests to backend vLLM servers
- **vLLM Server** (`vllm_serve.py`): Configurable vLLM server with OpenAI-compatible API endpoints

## Setup Instructions

### 1. Allocate SLURM Nodes

Start 2 or more nodes using SLURM:

```bash
srun --account=project_462000963 \
     --partition=dev-g \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=1 \
     --gres=gpu:mi250:1 \
     --time=0-02:00:00 \
     --mem=64G \
     --hint=nomultithread \
     --pty bash
```

### 2. Configure Environment

On each allocated node, set up the environment:

```bash
# Clear transformers cache and set HF home
unset TRANSFORMERS_CACHE
export HF_HOME=/path/to/your/hf/cache

# Enter singularity container
CONTAINER=/scratch/project_462000353/containers/vllm_v10.1.1.sif

export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd,/usr/lib64/libjansson.so.4

alias sing='singularity shell -B "$PWD" '"$CONTAINER"
sing
```

### 3. Start vLLM Servers

On each node, start the vLLM server:

```bash
python vllm_serve.py
```

Default configuration:
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Host: `0.0.0.0`
- Port: `18000`
- GPU memory utilization: `0.92`
- Max sequences: `512`
- Max model length: `2048`

Customize with CLI arguments:
```bash
python vllm_serve.py --model your-model --port 18001 --gpu-memory-utilization 0.8
```

### 4. Start Gateway

Set the backend servers and start the gateway:

```bash
export BACKENDS="http://nid007966:18000,http://nid007967:18000"
uvicorn gateway:app --host 0.0.0.0 --port 18080 --workers 1
```

## Usage

The gateway exposes OpenAI-compatible endpoints at `/v1/*` and forwards them to the backend servers.

### Health Check

```bash
curl http://gateway-host:18080/healthz
```

### Chat Completions

```bash
curl http://gateway-host:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Say hi"}]
  }'
```

## Configuration

### Gateway Configuration

- `BACKENDS`: Comma-separated list of backend server URLs
- Load balancing: Round-robin with automatic failover
- Timeout: 30s connect, 600s read, 30s write/pool
- Retry: Up to 3 attempts across different backends

### vLLM Server Configuration

Available CLI arguments:
- `--model`: Hugging Face model name
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 18000)
- `--gpu-memory-utilization`: GPU memory usage (default: 0.92)
- `--max-num-seqs`: Maximum concurrent sequences (default: 512)
- `--max-model-len`: Maximum model context length (default: 2048)
