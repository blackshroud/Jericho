from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
import logging
import json
import asyncio
import time
import uuid
import re
from typing import Optional, Any

# Use explicit IPv4 loopback by default to avoid ::1 resolution issues on some systems
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

app = FastAPI(title="autogpt-web-ollama")
# Configure root logger if not already configured
class ColorFormatter(logging.Formatter):
    GREEN = "\x1b[32m"; YELLOW = "\x1b[33m"; RED = "\x1b[31m"; CYAN = "\x1b[36m"; RESET = "\x1b[0m"
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            prefix = f"{self.RED}ERROR:{self.RESET}"
        elif record.levelno >= logging.WARNING:
            prefix = f"{self.YELLOW}WARNING:{self.RESET}"
        elif record.levelno == logging.INFO:
            prefix = f"{self.GREEN}INFO:{self.RESET}"
        else:
            prefix = f"{self.CYAN}DEBUG:{self.RESET}"
        return f"{prefix} {record.getMessage()}"

root = logging.getLogger()
if not root.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    root.addHandler(handler)
root.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

logger = logging.getLogger("autogpt-web-ollama")


def _make_timeout(sec: Optional[int], default_read: int, write: int = 30) -> httpx.Timeout:
    """Create an httpx.Timeout, allowing infinite read when sec <= 0.
    - sec: user-provided seconds; if <= 0, wait indefinitely for read.
    - default_read: fallback when sec is None.
    - write: write timeout seconds (defaults differ for some providers).
    """
    read = None if (sec is not None and sec <= 0) else (sec or default_read)
    return httpx.Timeout(connect=10, read=read, write=write, pool=10)


class ChatRequest(BaseModel):
    messages: list[dict]
    model: Optional[str] = None
    stream: Optional[bool] = False
    # Provider selection and settings
    provider: Optional[str] = "ollama"  # one of: ollama, openai, anthropic, gpt4all
    api_base: Optional[str] = None       # for openai-compatible or custom base
    api_key: Optional[str] = None        # for third-party providers
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    anthropic_version: Optional[str] = None
    timeout_sec: Optional[int] = None


ASSISTANT_PREFIX_RE = re.compile(r"^\s*(assistant|Assistant)\s*:\s*")

def _clean_content(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return ASSISTANT_PREFIX_RE.sub("", s, count=1)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    provider = (req.provider or "ollama").lower()
    rid = uuid.uuid4().hex[:8]
    logger.info(f"[{rid}] /api/chat start provider={provider} model={req.model} stream={req.stream} base={req.api_base} msgs={len(req.messages)} timeout={req.timeout_sec}")
    timeout = httpx.Timeout(connect=10, read=300, write=30, pool=10)

    if provider == "ollama":
        # Allow overriding host per-request via api_base
        host = (req.api_base or OLLAMA_HOST).rstrip("/")
        payload = {
            "model": req.model or DEFAULT_MODEL,
            "messages": req.messages,
            "stream": False,
        }
        # Map optional tuning to Ollama options
        options: dict[str, Any] = {}
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.max_tokens is not None:
            options["num_predict"] = req.max_tokens
        if options:
            payload["options"] = options

        url = f"{host}/api/chat"
        try:
            logger.info(f"[{rid}] ollama -> {url} model={payload.get('model')} msgs={len(payload.get('messages', []))}")
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300, write=30)) as client:
                try:
                    r = await client.post(url, json=payload)
                except httpx.ReadTimeout:
                    logger.warning(f"[{rid}] ollama read timeout; retrying once")
                    r = await client.post(url, json=payload)
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"[{rid}] ollama <- {r.status_code} {dt:.1f}ms")
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = r.text
                    logger.error("Ollama HTTP %s: %s", r.status_code, body)
                    return JSONResponse(
                        status_code=502,
                        content={
                            "ok": False,
                            "error": "HTTPStatusError",
                            "status": r.status_code,
                            "detail": str(e),
                            "body": body,
                        },
                    )
                data = r.json()
                content = None
                try:
                    content = data.get("message", {}).get("content")
                except Exception:
                    content = None
                content = _clean_content(content)
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception(f"[{rid}] ollama network error url={url}")
            raise HTTPException(status_code=502, detail=f"Network error talking to Ollama: {e}")

    elif provider == "openai":
        api_base = req.api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        api_key = req.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=400, content={"ok": False, "error": "Missing OPENAI_API_KEY or api_key"})
        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": req.model or "gpt-4o-mini",
            "messages": req.messages,
            "stream": False,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        try:
            logger.info(f"[{rid}] openai -> {url} model={payload.get('model')}")
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300, write=30)) as client:
                r = await client.post(url, headers=headers, json=payload)
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"[{rid}] openai <- {r.status_code} {dt:.1f}ms")
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = r.text
                    logger.error("OpenAI HTTP %s: %s", r.status_code, body[:500])
                    return JSONResponse(status_code=502, content={"ok": False, "status": r.status_code, "detail": str(e), "body": body})
                data = r.json()
                content = None
                try:
                    content = data.get("choices", [{}])[0].get("message", {}).get("content")
                except Exception:
                    content = None
                content = _clean_content(content)
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception(f"[{rid}] openai network error url={url}")
            raise HTTPException(status_code=502, detail=f"Network error talking to OpenAI-compatible: {e}")

    elif provider == "anthropic":
        api_base = req.api_base or os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1")
        api_key = req.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return JSONResponse(status_code=400, content={"ok": False, "error": "Missing ANTHROPIC_API_KEY or api_key"})
        url = f"{api_base.rstrip('/')}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": req.anthropic_version or os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        }
        # Extract optional system prompt
        system_msg = next((m.get("content") for m in req.messages if m.get("role") == "system"), None)
        conv = [m for m in req.messages if m.get("role") in ("user", "assistant")]
        payload: dict[str, Any] = {
            "model": req.model or "claude-3-5-sonnet-latest",
            "max_tokens": req.max_tokens or 512,
            "messages": [{"role": m.get("role"), "content": m.get("content", "")} for m in conv],
        }
        if system_msg:
            payload["system"] = system_msg
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        try:
            logger.info(f"[{rid}] anthropic -> {url} model={payload.get('model')}")
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300, write=30)) as client:
                r = await client.post(url, headers=headers, json=payload)
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"[{rid}] anthropic <- {r.status_code} {dt:.1f}ms")
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = r.text
                    logger.error("Anthropic HTTP %s: %s", r.status_code, body[:500])
                    return JSONResponse(status_code=502, content={"ok": False, "status": r.status_code, "detail": str(e), "body": body})
                data = r.json()
                content = None
                try:
                    # messages API returns { content: [{type:'text', text:'...'}], ... }
                    parts = data.get("content", [])
                    content = "".join(p.get("text", "") for p in parts if p.get("type") == "text") or None
                except Exception:
                    content = None
                content = _clean_content(content)
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception(f"[{rid}] anthropic network error url={url}")
            raise HTTPException(status_code=502, detail=f"Network error talking to Anthropic: {e}")

    elif provider == "gpt4all":
        api_base = req.api_base or os.getenv("GPT4ALL_API_BASE", "http://127.0.0.1:4891/v1")
        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "messages": req.messages,
            "stream": False,
        }
        # Include model only if provided; otherwise let server pick default
        if req.model:
            payload["model"] = req.model
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        else:
            payload["max_tokens"] = 512

        read_timeout = req.timeout_sec or 600
        try:
            logger.info(f"[{rid}] gpt4all -> {url} model={payload.get('model')} base={api_base}")
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, read_timeout, write=60)) as client:
                async def attempt():
                    _r = await client.post(url, headers=headers, json=payload)
                    _used = url
                    if _r.status_code == 404 and "/v1" not in api_base:
                        url_alt = f"{api_base.rstrip('/')}/v1/chat/completions"
                        logger.debug("GPT4All fallback -> %s", url_alt)
                        _r = await client.post(url_alt, headers=headers, json=payload)
                        _used = url_alt
                    if _r.status_code == 404:
                        prompt = "\n\n".join(f"{m.get('role')}: {m.get('content','')}" for m in req.messages)
                        payload2: dict[str, Any] = {
                            "prompt": prompt,
                            "stream": False,
                        }
                        if "model" in payload:
                            payload2["model"] = payload["model"]
                        if req.temperature is not None:
                            payload2["temperature"] = req.temperature
                        payload2["max_tokens"] = req.max_tokens if req.max_tokens is not None else 512
                        url2 = f"{api_base.rstrip('/')}/completions"
                        logger.debug("GPT4All fallback -> %s", url2)
                        _r = await client.post(url2, headers=headers, json=payload2)
                        _used = url2
                        if _r.status_code == 404 and "/v1" not in api_base:
                            url3 = f"{api_base.rstrip('/')}/v1/completions"
                            logger.debug("GPT4All fallback -> %s", url3)
                            _r = await client.post(url3, headers=headers, json=payload2)
                            _used = url3
                    return _r, _used

                try:
                    r, used_url = await attempt()
                    logger.info(f"[{rid}] gpt4all <- {r.status_code} via {used_url}")
                except httpx.ReadTimeout:
                    logger.warning("GPT4All read timeout, retrying once...")
                    r, used_url = await attempt()
                except httpx.RequestError:
                    logger.warning("GPT4All request error, retrying once...")
                    r, used_url = await attempt()

                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = r.text
                    logger.error("GPT4All HTTP %s (%s): %s", r.status_code, used_url, body[:500])
                    return JSONResponse(status_code=502, content={"ok": False, "status": r.status_code, "detail": str(e), "body": body, "url": used_url})
                data = r.json()
                content = None
                try:
                    choice0 = (data.get("choices") or [{}])[0]
                    msg = choice0.get("message") or {}
                    content = msg.get("content") or choice0.get("text")
                except Exception:
                    content = None
                content = _clean_content(content)
                # Capture the actual model used by the server (may differ from requested)
                used_model = data.get("model") or ((data.get("choices") or [{}])[0].get("model")) or payload.get("model")
                return JSONResponse({"ok": True, "data": data, "content": content, "used_model": used_model})
        except httpx.RequestError as e:
            logger.exception(f"[{rid}] gpt4all network error url={url}")
            raise HTTPException(status_code=502, detail=f"Network error talking to GPT4All: {e}")

    else:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Unsupported provider: {provider}"})

# Mount static files last so API routes take precedence

@app.get("/api/gpt4all/models")
async def gpt4all_models(api_base: str, timeout_sec: Optional[int] = None):
    base = (api_base or "").rstrip("/")
    if not base:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Missing api_base"})
    urls = [f"{base}/models"]
    if "/v1" not in base:
        urls.append(f"{base}/v1/models")
    # For listing, force a finite timeout even if timeout_sec<=0
    sec_eff = timeout_sec if (timeout_sec and timeout_sec > 0) else 30
    t = _make_timeout(sec_eff, 30, write=15)
    try:
        async with httpx.AsyncClient(timeout=t) as client:
            last_err = None
            for u in urls:
                try:
                    r = await client.get(u)
                    if r.status_code == 404:
                        last_err = f"404 at {u}"
                        continue
                    r.raise_for_status()
                    data = r.json()
                    models: list[str] = []
                    # OpenAI-style: { data: [ { id: "..." } ] }
                    if isinstance(data, dict) and isinstance(data.get("data"), list):
                        for it in data["data"]:
                            mid = it.get("id") or it.get("model") or it.get("name")
                            if isinstance(mid, str):
                                models.append(mid)
                    # GPT4All variants: { models: [ { name: "..." } ] } or { models: ["..."] }
                    if not models and isinstance(data.get("models"), list):
                        for it in data["models"]:
                            if isinstance(it, str):
                                models.append(it)
                            elif isinstance(it, dict):
                                mid = it.get("name") or it.get("id") or it.get("model")
                                if isinstance(mid, str):
                                    models.append(mid)
                    return JSONResponse({"ok": True, "models": sorted(set(models))})
                except httpx.HTTPError as e:
                    last_err = str(e)
                    continue
            return JSONResponse(status_code=502, content={"ok": False, "error": last_err or "Failed to fetch models"})
    except httpx.RequestError as e:
        return JSONResponse(status_code=502, content={"ok": False, "error": str(e)})


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    provider = (req.provider or "ollama").lower()
    rid = uuid.uuid4().hex[:8]
    logger.info(f"[{rid}] /api/chat/stream start provider={provider} model={req.model} base={req.api_base} msgs={len(req.messages)} timeout={req.timeout_sec}")

    async def gen():
        try:
            if provider == "ollama":
                host = (req.api_base or OLLAMA_HOST).rstrip("/")
                url = f"{host}/api/chat"
                payload: dict[str, Any] = {
                    "messages": req.messages,
                    "stream": True,
                }
                if req.model:
                    payload["model"] = req.model
                options: dict[str, Any] = {}
                if req.temperature is not None:
                    options["temperature"] = req.temperature
                if req.max_tokens is not None:
                    options["num_predict"] = req.max_tokens
                if options:
                    payload["options"] = options
                async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300)) as client:
                    async with client.stream("POST", url, json=payload) as r:
                        logger.info(f"[{rid}] ollama stream -> {url}")
                        r.raise_for_status()
                        chunk_n = 0
                        async for line in r.aiter_lines():
                            chunk_n += 1
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            # Ollama streams message chunks; stop on done
                            msg = obj.get("message", {})
                            delta = msg.get("content") or ""
                            if delta:
                                yield delta
                            if obj.get("done"):
                                logger.info(f"[{rid}] ollama stream done chunks={chunk_n}")
                                break
            elif provider in ("gpt4all", "openai"):
                base = (req.api_base or "http://127.0.0.1:4891/v1").rstrip("/")
                url = f"{base}/chat/completions"
                headers = {
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Content-Type": "application/json",
                }
                payload: dict[str, Any] = {
                    "messages": req.messages,
                    "stream": True,
                }
                # Ensure model when possible
                if req.model:
                    payload["model"] = req.model
                if req.temperature is not None:
                    payload["temperature"] = req.temperature
                if req.max_tokens is not None:
                    payload["max_tokens"] = req.max_tokens
                async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 600, write=60)) as client:
                    # Try chat/completions first; on 400/404, fallback to completions
                    try:
                        async with client.stream("POST", url, headers=headers, json=payload) as r:
                            logger.info(f"[{rid}] gpt4all stream -> {url}")
                            if r.status_code in (400, 404):
                                # Read error body for diagnostics
                                try:
                                    err_body = await r.aread()
                                    logger.warning(f"[{rid}] gpt4all stream not supported (chat/completions): {err_body.decode(errors='ignore')}")
                                except Exception:
                                    logger.warning(f"[{rid}] gpt4all stream not supported (chat/completions): no body")
                                raise RuntimeError("fallback_to_completions")
                            r.raise_for_status()
                            chunk_n = 0
                            async for raw in r.aiter_lines():
                                chunk_n += 1
                                if not raw:
                                    continue
                                if raw.startswith("data:"):
                                    data = raw[5:].strip()
                                else:
                                    data = raw.strip()
                                if data == "[DONE]":
                                    logger.info(f"[{rid}] gpt4all stream done chunks={chunk_n}")
                                    break
                                try:
                                    obj = json.loads(data)
                                except Exception:
                                    continue
                                choice0 = (obj.get("choices") or [{}])[0]
                                delta = (choice0.get("delta") or {}).get("content") or (choice0.get("message") or {}).get("content") or choice0.get("text") or obj.get("text") or obj.get("response")
                                if delta:
                                    yield delta
                    except RuntimeError:
                        # Fallback to /completions with a prompt
                        url2 = f"{base}/completions"
                        prompt = "\n\n".join(f"{m.get('role')}: {m.get('content','')}" for m in req.messages)
                        payload2: dict[str, Any] = {"prompt": prompt, "stream": True}
                        if req.model:
                            payload2["model"] = req.model
                        if req.temperature is not None:
                            payload2["temperature"] = req.temperature
                        if req.max_tokens is not None:
                            payload2["max_tokens"] = req.max_tokens
                        # Try streaming on /completions
                        async with client.stream("POST", url2, headers=headers, json=payload2) as r2:
                            logger.info(f"[{rid}] gpt4all stream fallback -> {url2}")
                            if r2.status_code in (400, 404):
                                # If streaming is not supported, pseudo-stream a normal response
                                try:
                                    err_body = await r2.aread()
                                    logger.warning(f"[{rid}] gpt4all stream not supported (completions): {err_body.decode(errors='ignore')}")
                                except Exception:
                                    logger.warning(f"[{rid}] gpt4all stream not supported (completions): no body")
                                # Non-streaming fallback
                                payload2_ns = dict(payload2)
                                payload2_ns["stream"] = False
                                resp_ns = await client.post(url2, headers={"Content-Type": "application/json"}, json=payload2_ns)
                                resp_ns.raise_for_status()
                                data = resp_ns.json()
                                # Extract text content
                                text = None
                                try:
                                    choice0 = (data.get("choices") or [{}])[0]
                                    text = choice0.get("text") or (choice0.get("message") or {}).get("content") or data.get("text") or data.get("response")
                                except Exception:
                                    text = None
                                text = _clean_content(text)
                                if not text:
                                    yield json.dumps(data)
                                else:
                                    # Pseudo-stream in chunks
                                    s = str(text)
                                    chunk_size = 120
                                    for i in range(0, len(s), chunk_size):
                                        yield s[i:i+chunk_size]
                                        await asyncio.sleep(0)
                                return
                            # If streaming is supported, forward SSE lines
                            r2.raise_for_status()
                            chunk_n2 = 0
                            async for raw in r2.aiter_lines():
                                chunk_n2 += 1
                                if not raw:
                                    continue
                                if raw.startswith("data:"):
                                    data = raw[5:].strip()
                                else:
                                    data = raw.strip()
                                if data == "[DONE]":
                                    logger.info(f"[{rid}] gpt4all stream fallback done chunks={chunk_n2}")
                                    break
                                try:
                                    obj = json.loads(data)
                                except Exception:
                                    continue
                                choice0 = (obj.get("choices") or [{}])[0]
                                delta = choice0.get("text") or (choice0.get("delta") or {}).get("content") or (choice0.get("message") or {}).get("content") or obj.get("text") or obj.get("response")
                                if delta:
                                    yield delta
            else:
                yield "[Unsupported provider]"
        except httpx.HTTPError as e:
            logger.exception(f"[{rid}] stream http error")
            yield f"\n[stream error: {str(e)}]"
        except Exception as e:
            logger.exception(f"[{rid}] stream error")
            yield f"\n[stream error: {str(e)}]"
        await asyncio.sleep(0)

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

app.mount("/", StaticFiles(directory="web", html=True), name="static")
