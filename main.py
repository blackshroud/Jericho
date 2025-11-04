from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
import logging
from typing import Optional, Any

# Use explicit IPv4 loopback by default to avoid ::1 resolution issues on some systems
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

app = FastAPI(title="autogpt-web-ollama")
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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    provider = (req.provider or "ollama").lower()
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
            logger.debug("/api/chat(ollama) -> %s model=%s messages=%s", url, payload.get("model"), len(payload.get("messages", [])))
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300, write=30)) as client:
                try:
                    r = await client.post(url, json=payload)
                except httpx.ReadTimeout:
                    logger.warning("Read timeout from Ollama, retrying once...")
                    r = await client.post(url, json=payload)
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
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception("Network error talking to Ollama at %s", url)
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
            logger.debug("/api/chat(openai) -> %s model=%s", url, payload.get("model"))
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300, write=30)) as client:
                r = await client.post(url, headers=headers, json=payload)
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
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception("Network error talking to OpenAI-compatible at %s", url)
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
            logger.debug("/api/chat(anthropic) -> %s model=%s", url, payload.get("model"))
            async with httpx.AsyncClient(timeout=_make_timeout(req.timeout_sec, 300, write=30)) as client:
                r = await client.post(url, headers=headers, json=payload)
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
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception("Network error talking to Anthropic at %s", url)
            raise HTTPException(status_code=502, detail=f"Network error talking to Anthropic: {e}")

    elif provider == "gpt4all":
        api_base = req.api_base or os.getenv("GPT4ALL_API_BASE", "http://127.0.0.1:4891/v1")
        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": req.model or "gpt4all",
            "messages": req.messages,
            "stream": False,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        else:
            payload["max_tokens"] = 512

        read_timeout = req.timeout_sec or 600
        try:
            logger.debug("/api/chat(gpt4all) -> %s model=%s", url, payload.get("model"))
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
                            "model": payload["model"],
                            "prompt": prompt,
                            "stream": False,
                        }
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
                return JSONResponse({"ok": True, "data": data, "content": content})
        except httpx.RequestError as e:
            logger.exception("Network error talking to GPT4All at %s", url)
            raise HTTPException(status_code=502, detail=f"Network error talking to GPT4All: {e}")

    else:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Unsupported provider: {provider}"})

# Mount static files last so API routes take precedence
app.mount("/", StaticFiles(directory="web", html=True), name="static")
