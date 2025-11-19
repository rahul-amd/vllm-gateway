# gateway.py
import asyncio
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

# ---------------------------
# Config
# ---------------------------
# Comma-separated list of backend base URLs, e.g.:
# BACKENDS="http://nid007974:8000,http://nid007975:8000"
BACKENDS = [b.strip() for b in os.getenv("BACKENDS", "").split(",") if b.strip()]
if not BACKENDS:
    raise RuntimeError("Set BACKENDS env var to a comma-separated list of backend URLs")

# Optional: ensure all backends run the same model version
REQUIRED_MODEL_VERSION = os.getenv("REQUIRED_MODEL_VERSION")  # e.g., "2025-10-23a"

# Circuit breaker / health tuning
FAIL_EJECT_THRESHOLD = int(os.getenv("FAIL_EJECT_THRESHOLD", 3))        # consecutive failures to eject
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", 30))               # seconds ejected after threshold
HEALTH_INTERVAL = float(os.getenv("HEALTH_INTERVAL", 30.0))             # seconds between active health checks
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 90.0))            # per-request timeout (no streaming)
MAX_BODY_MB = int(os.getenv("MAX_BODY_MB", 20))                         # body size limit (MB)

# Backpressure: max concurrent in-flight generate requests per backend
MAX_INFLIGHT_PER_BACKEND = int(os.getenv("MAX_INFLIGHT_PER_BACKEND", 8))

# Stickiness: prefer client-provided session id
STICKY_HEADER = os.getenv("STICKY_HEADER", "x-session-id")
# If absent, derive from prompts (normalized first N chars)
PROMPT_STICKY_PREFIX_CHARS = int(os.getenv("PROMPT_STICKY_PREFIX_CHARS", 200))

app = FastAPI()

# ---------------------------
# Backend registry/state
# ---------------------------
class Backend:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.consecutive_failures = 0
        self.ejected_until: float = 0.0
        self.healthy: bool = False
        self.model_version: Optional[str] = None
        self.world_size: Optional[int] = None
        self.inflight: int = 0
        self.last_latency_ms: float = 0.0

    def available(self) -> bool:
        now = time.time()
        version_ok = REQUIRED_MODEL_VERSION is None or self.model_version == REQUIRED_MODEL_VERSION
        return self.healthy and now >= self.ejected_until and version_ok

    def __repr__(self):
        return f"Backend(url={self.base_url}, healthy={self.healthy}, inflight={self.inflight}, ver={self.model_version})"

BACKEND_OBJS: List[Backend] = [Backend(b) for b in BACKENDS]

# ---------------------------
# Utilities
# ---------------------------
def get_sync_backend() -> Backend:
    if not BACKEND_OBJS:
        raise RuntimeError("No backends configured")
    return BACKEND_OBJS[0]

async def fetch_health(client: httpx.AsyncClient, be: Backend) -> None:
    try:
        r = await client.get(f"{be.base_url}/health/", timeout=5.0)
        r.raise_for_status()
        data = r.json() or {}
        be.healthy = (data.get("status") == "ok" or data.get("ok") is True)
        be.model_version = data.get("model_version")
    except Exception:
        be.healthy = False
        return

    if be.healthy and be.world_size is None:
        try:
            r2 = await client.get(f"{be.base_url}/get_world_size/", timeout=5.0)
            r2.raise_for_status()
            ws = (r2.json() or {}).get("world_size")
            be.world_size = int(ws) if ws is not None else None
        except Exception:
            be.world_size = None

async def health_loop() -> None:
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.gather(*[fetch_health(client, be) for be in BACKEND_OBJS], return_exceptions=True)
            await asyncio.sleep(HEALTH_INTERVAL)

@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(health_loop())

# Normalize prompt to derive sticky key when header absent
def derive_sticky_from_prompts(body: Dict[str, Any]) -> Optional[str]:
    try:
        prompts = body.get("prompts") or []
        if not prompts:
            return None
        p0 = str(prompts[0]).lower().strip()[:PROMPT_STICKY_PREFIX_CHARS]
        return p0 if p0 else None
    except Exception:
        return None

# Consistent hashing ring
class HashRing:
    def __init__(self, nodes: List[str], replicas: int = 100):
        ring: List[Tuple[int, str]] = []
        for n in nodes:
            for r in range(replicas):
                key = f"{n}:{r}".encode()
                ring.append((int(hashlib.md5(key).hexdigest(), 16), n))
        ring.sort(key=lambda x: x[0])
        self.ring = ring

    def get(self, key: str) -> str:
        if not self.ring:
            raise RuntimeError("Empty ring")
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        lo, hi = 0, len(self.ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ring[mid][0] < h:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(self.ring):
            lo = 0
        return self.ring[lo][1]

HASH_RING = HashRing([be.base_url for be in BACKEND_OBJS])

# Selection: sticky → available check → inflight threshold → least-inflight fallback
def choose_backend(sticky_key: Optional[str]) -> Backend:
    candidates = [be for be in BACKEND_OBJS if be.available()]
    if not candidates:
        cooled = [be for be in BACKEND_OBJS if time.time() >= be.ejected_until]
        if cooled:
            return random.choice(cooled)
        raise HTTPException(status_code=503, detail="No healthy backends")

    if sticky_key:
        target_url = HASH_RING.get(sticky_key)
        sticky = next((be for be in candidates if be.base_url == target_url), None)
        if sticky and sticky.inflight < MAX_INFLIGHT_PER_BACKEND:
            return sticky

    if len(candidates) >= 2:
        a, b = random.sample(candidates, 2)
        return a if a.inflight <= b.inflight else b
    return candidates[0]

# Failure/success book-keeping
def note_failure(be: Backend) -> None:
    be.consecutive_failures += 1
    if be.consecutive_failures >= FAIL_EJECT_THRESHOLD:
        be.ejected_until = time.time() + COOLDOWN_SECONDS
        be.consecutive_failures = 0
        be.healthy = False

def note_success(be: Backend, latency_ms: float) -> None:
    be.consecutive_failures = 0
    be.last_latency_ms = latency_ms

# Body size guard
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > MAX_BODY_MB * 1024 * 1024:
            return Response(status_code=413, content="Payload too large")
        # Re-inject body for downstream handlers
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        request = Request(request.scope, receive=receive)
    return await call_next(request)

# ---------------------------
# Transparent endpoints
# ---------------------------
@app.get("/health/")
async def health_proxy():
    be = choose_backend(sticky_key=None)
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{be.base_url}/health/")
        r.raise_for_status()
        data = r.json() or {}
        payload = {
            "status": "ok" if (data.get("ok") or data.get("status") == "ok") else "down",
            "model_version": data.get("model_version", None),
        }
        resp = JSONResponse(payload)
        resp.headers["X-Backend"] = be.base_url
        return resp
        

@app.get("/get_world_size/")
async def get_world_size_proxy():
    be = get_sync_backend()
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{be.base_url}/get_world_size/")
            r.raise_for_status()
            data = r.json() or {}
            ws = data.get("world_size")
            if ws is None:
                raise HTTPException(status_code=502, detail="Backend did not return world_size")
            return JSONResponse({"world_size": int(ws)})
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Error fetching world_size from {be.base_url}: {e}") from e

@app.post("/generate/")
async def generate_proxy(request: Request):
    """
    Proxy endpoint that directly calls the backend's /generate/ API from vllm_serve.py.
    This endpoint passes through the request body as-is to match the GenerateRequest format.
    """
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate required fields
    prompts = body.get("prompts") or []
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(status_code=400, detail="`prompts` must be a non-empty list of strings")

    # Stickiness: try header, else derive from first prompt
    sticky_key = request.headers.get(STICKY_HEADER) or derive_sticky_from_prompts(body)
    be = choose_backend(sticky_key)

    if be.inflight >= MAX_INFLIGHT_PER_BACKEND:
        exc = HTTPException(status_code=429, detail="Backend busy; please retry")
        exc.headers = {"Retry-After": "1"}
        raise exc

    url = f"{be.base_url}/generate/"
    t0 = time.perf_counter()
    be.inflight += 1
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            note_success(be, (time.perf_counter() - t0) * 1000)
            resp = JSONResponse(r.json())
            resp.headers["X-Backend"] = be.base_url
            return resp
    except httpx.HTTPStatusError as e:
        note_failure(be)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        note_failure(be)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        be.inflight -= 1

@app.post("/chat/")
async def chat_proxy(request: Request):
    """
    Proxy endpoint that directly calls the backend's /chat/ API from vllm_serve.py.
    This endpoint passes through the request body as-is to match the ChatRequest format.
    """
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate required fields
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list of message lists")

    # Stickiness: try header, else derive from first message content if available
    sticky_key = request.headers.get(STICKY_HEADER)
    if not sticky_key and messages:
        try:
            first_message_list = messages[0] if isinstance(messages[0], list) else []
            if first_message_list and isinstance(first_message_list[0], dict):
                content = str(first_message_list[0].get("content", ""))
                sticky_key = content.lower().strip()[:PROMPT_STICKY_PREFIX_CHARS] if content else None
        except Exception:
            pass

    be = choose_backend(sticky_key)

    if be.inflight >= MAX_INFLIGHT_PER_BACKEND:
        exc = HTTPException(status_code=429, detail="Backend busy; please retry")
        exc.headers = {"Retry-After": "1"}
        raise exc

    url = f"{be.base_url}/chat/"
    t0 = time.perf_counter()
    be.inflight += 1
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            note_success(be, (time.perf_counter() - t0) * 1000)
            resp = JSONResponse(r.json())
            resp.headers["X-Backend"] = be.base_url
            return resp
    except httpx.HTTPStatusError as e:
        note_failure(be)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        note_failure(be)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        be.inflight -= 1

class WeightSyncManager:
    """
    Simple manager to make /init_communicator and /close_communicator
    idempotent and single-backend.

    - We only talk to the "sync backend" (get_sync_backend()).
    - First successful /init_communicator/ call hits the backend and caches
      its response.
    - Subsequent /init_communicator/ calls just return the cached response.
    - /close_communicator/ is idempotent and resets state.
    """

    def __init__(self):
        self._initialized: bool = False
        self._last_init_response: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def init(self, body: Dict[str, Any]) -> Dict[str, Any]:
        be = get_sync_backend()
        async with self._lock:
            if self._initialized and self._last_init_response is not None:
                # Already initialized; be nice and just return cached response.
                return dict(self._last_init_response)

            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(f"{be.base_url}/init_communicator/", json=body)
                r.raise_for_status()
                data = r.json() or {}

            self._initialized = True
            self._last_init_response = data
            return dict(data)

    async def update_named_param(self, body: Dict[str, Any]) -> Dict[str, Any]:
        be = get_sync_backend()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{be.base_url}/update_named_param/", json=body)
            r.raise_for_status()
            return r.json() or {}

    async def reset_prefix_cache(self) -> Dict[str, Any]:
        be = get_sync_backend()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{be.base_url}/reset_prefix_cache/")
            r.raise_for_status()
            return r.json() or {}

    async def close(self) -> Dict[str, Any]:
        be = get_sync_backend()
        async with self._lock:
            if not self._initialized:
                # Already closed / never initialized.
                return {"status": "ok", "already_closed": True}

            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(f"{be.base_url}/close_communicator/")
                r.raise_for_status()
                data = r.json() or {}

            # Reset state
            self._initialized = False
            self._last_init_response = None
            return data

WEIGHT_SYNC_MANAGER = WeightSyncManager()

@app.post("/init_communicator/")
async def init_communicator_gateway(request: Request):
    """
    Idempotent /init_communicator/:
      - Only the first caller actually triggers backend init.
      - Later callers get the cached backend response.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        data = await WEIGHT_SYNC_MANAGER.init(body)
        return JSONResponse(data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Backend init_communicator failed: {e}")

@app.post("/update_named_param/")
async def update_named_param_gateway(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        data = await WEIGHT_SYNC_MANAGER.update_named_param(body)
        return JSONResponse(data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Backend update_named_param failed: {e}")

@app.post("/reset_prefix_cache/")
async def reset_prefix_cache_gateway():
    try:
        data = await WEIGHT_SYNC_MANAGER.reset_prefix_cache()
        return JSONResponse(data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Backend reset_prefix_cache failed: {e}")

@app.post("/close_communicator/")
async def close_communicator_gateway():
    try:
        data = await WEIGHT_SYNC_MANAGER.close()
        return JSONResponse(data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Backend close_communicator failed: {e}")
