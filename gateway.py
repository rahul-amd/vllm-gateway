from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import httpx, asyncio, os, itertools

# Comma-separated backend URLs, e.g. "http://n1:18000,http://n2:18000"
BACKENDS = [u.strip() for u in os.getenv("BACKENDS","").split(",") if u.strip()]
assert BACKENDS, "Set BACKENDS env var to a comma-separated list of http://host:port"

_rr = itertools.cycle(BACKENDS)
pick_backend = lambda: next(_rr)
app = FastAPI()

@app.get("/healthz")
async def healthz():
    return {"ok": True, "backends": BACKENDS}

def _forwardable_headers(hdrs: dict) -> dict:
    # Strip hop-by-hop headers for safety
    hop = {"connection","keep-alive","transfer-encoding","upgrade","proxy-authenticate",
           "proxy-authorization","te","trailers"}
    return {k: v for k, v in hdrs.items() if k.lower() not in hop}

async def _proxy(req: Request, path: str) -> Response:
    method = req.method
    body = await req.body()
    headers = _forwardable_headers(dict(req.headers))

    timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)

    attempts = min(len(BACKENDS), 3)
    last_err = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(attempts):
            backend = pick_backend()
            url = f"{backend}{path}"
            try:
                r = await client.request(method, url, headers=headers, content=body)
                # Return full buffered body (no streaming)
                resp_headers = _forwardable_headers(r.headers)
                return Response(content=r.content, status_code=r.status_code, headers=resp_headers)
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

    return JSONResponse({"error": "All backends failed", "detail": last_err}, status_code=502)

# Proxy all OpenAI-compatible endpoints
@app.api_route("/v1/{full_path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def v1_proxy(full_path: str, request: Request):
    return await _proxy(request, f"/v1/{full_path}")


# run with:
# export BACKENDS="http://nid007966:18000,http://nid007967:18000"
# uvicorn gateway:app --host 0.0.0.0 --port 18080 --workers 1
