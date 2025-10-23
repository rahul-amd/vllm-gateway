# TRL Gateway

A FastAPI-based load balancer and gateway for distributing requests across multiple vLLM backend instances. This gateway provides intelligent routing, health monitoring, and failover capabilities for TRL workloads.

## Features

- **Load Balancing**: Distributes requests across multiple backend instances using consistent hashing
- **Health Monitoring**: Continuous health checks with circuit breaker pattern
- **Sticky Sessions**: Routes requests to the same backend based on session ID or prompt content
- **Backpressure Control**: Limits concurrent requests per backend to prevent overload
- **Failover**: Automatic ejection and recovery of unhealthy backends
- **Admin Operations**: Broadcasts administrative commands to all backends

## Configuration

Set the following environment variables:

- `BACKENDS`: Comma-separated list of backend URLs (required)
- `REQUIRED_MODEL_VERSION`: Optional model version requirement
- `FAIL_EJECT_THRESHOLD`: Consecutive failures before ejection (default: 3)
- `COOLDOWN_SECONDS`: Ejection duration in seconds (default: 30)
- `HEALTH_INTERVAL`: Health check interval in seconds (default: 3.0)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 90.0)
- `MAX_BODY_MB`: Maximum request body size in MB (default: 20)
- `MAX_INFLIGHT_PER_BACKEND`: Max concurrent requests per backend (default: 8)
- `STICKY_HEADER`: Header name for sticky sessions (default: "x-session-id")
- `PROMPT_STICKY_PREFIX_CHARS`: Characters to use for prompt-based stickiness (default: 200)

## Endpoints

### Transparent Proxies
- `GET /health/` - Health check proxy
- `GET /get_world_size/` - World size proxy
- `POST /generate/` - Text generation proxy

### Admin Broadcasts
- `POST /init_communicator/` - Initialize communicator on all backends
- `POST /update_named_param/` - Update named parameters on all backends
- `POST /reset_prefix_cache/` - Reset prefix cache on all backends
- `POST /close_communicator/` - Close communicator on all backends

## Usage

Start TRL servers:
```
trl vllm-serve --model Qwen/Qwen2.5-7B --port 8000  # <-- asssume node1
trl vllm-serve --model Qwen/Qwen2.5-7B --port 8000  # <-- assume node2
```

Start gateway:
```
export BACKENDS="http://node1:8000,http://node2:8000"
uvicorn trl_gateway:app --host 0.0.0.0 --port 8080  # <-- assume gnode
```

Hit the APIs
```
curl http://gnode:8080/health/

curl -X POST http://gnode:8080/generate/ \
  -H 'content-type: application/json' \
  -H 'x-session-id: session-123' \
  -d '{"prompts":["Hello"],"max_tokens":8}'
```

