"""Drive a model over the catalog tools via the OpenAI Responses API.

Deliberately not built on the consumer ChatGPT UI: custom tool support there is
gated by plan and the catalogue would have to be exposed to reach it. Through
the API the tools live inside this process, the database never leaves it, and
the model's entire view of the catalogue is the three functions in tools.py.

No SDK — one HTTPS endpoint and `requests`, which the project already depends on.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from .tools import TOOL_SCHEMAS, CatalogService, dispatch

log = logging.getLogger(__name__)

ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5-mini"

INSTRUCTIONS = """You are an interior designer working from a fixed catalogue of \
products available in India, priced in INR.

Rules you must not break:
- Every product you propose must come from search_catalog or get_product. Never \
name a product, SKU or price the tools did not return.
- Never state a price yourself. Prices come from price_design.
- Search per object slot (bed, wardrobe, lighting, ...) rather than once for the \
whole room, and respect the room's dimensions using the max_*_mm filters.
- For materials such as paint or cement, pass placeable_only=false: they are \
bought for a room but never placed in it as objects.
- If nothing suitable exists, say so plainly. An honest gap is better than a \
substitute the customer did not ask for.

Finish by calling price_design with everything you placed, then summarise the \
room in a few sentences and give the total the tool returned."""


class AgentError(RuntimeError):
    pass


def _post(payload: dict[str, Any], api_key: str, timeout: int = 180) -> dict[str, Any]:
    for attempt in range(1, 4):
        r = requests.post(
            ENDPOINT, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get("error", {}).get("message", r.text[:300])
        except ValueError:
            detail = r.text[:300]

        # A 429 means either "slow down" or "you have no money". Only the first
        # is worth waiting out; backing off on a billing failure just adds six
        # seconds before the same error.
        exhausted = any(s in detail.lower() for s in
                        ("no credits", "quota", "billing", "insufficient_quota"))
        if r.status_code == 429 and attempt < 3 and not exhausted:
            wait = float(r.headers.get("Retry-After", 2 * attempt))
            log.warning("rate limited, sleeping %.0fs", wait)
            time.sleep(wait)
            continue
        raise AgentError(f"HTTP {r.status_code}: {detail}")
    raise AgentError("gave up after repeated rate limiting")


def run(
    prompt: str,
    service: CatalogService,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_turns: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one design request to completion. Returns the reply and the priced design."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise AgentError("OPENAI_API_KEY is not set")

    conversation: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    calls: list[dict[str, Any]] = []
    bom: dict[str, Any] | None = None
    text = ""
    tokens = {"input": 0, "output": 0}

    for turn in range(1, max_turns + 1):
        data = _post({
            "model": model,
            "instructions": INSTRUCTIONS,
            "input": conversation,
            "tools": TOOL_SCHEMAS,
        }, api_key)

        usage = data.get("usage") or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)

        output = data.get("output") or []
        pending = [o for o in output if o.get("type") == "function_call"]

        for item in output:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        text += part.get("text", "")

        if not pending:
            break

        # Echo the model's own call items back before answering them; the API
        # matches each result to its call by call_id.
        conversation.extend(pending)
        for call in pending:
            try:
                args = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"arguments were not valid JSON: {exc}"}
                args = {}
            else:
                try:
                    result = dispatch(service, call["name"], args)
                except TypeError as exc:
                    # A wrong argument name is the model's mistake to correct,
                    # not a crash: hand it back as a tool result.
                    result = {"error": f"bad arguments for {call['name']}: {exc}"}
                except Exception as exc:
                    log.exception("tool %s failed", call.get("name"))
                    result = {"error": f"{call['name']} failed: {exc.__class__.__name__}"}

            if call["name"] == "price_design" and "lines" in result:
                bom = result
            calls.append({"turn": turn, "tool": call["name"], "arguments": args,
                          "result_size": len(json.dumps(result, default=str))})
            if verbose:
                brief = {k: v for k, v in args.items() if v not in (None, "")}
                print(f"  → {call['name']}({json.dumps(brief, ensure_ascii=False)[:110]})")

            conversation.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result, ensure_ascii=False, default=str),
            })
    else:
        log.warning("hit the %d-turn limit without a final answer", max_turns)

    return {
        "model": model,
        "reply": text.strip(),
        "priced_design": bom,
        "tool_calls": calls,
        "turns": len({c["turn"] for c in calls}) + 1,
        "tokens": tokens,
        "products_offered": len(service.issued),
    }
