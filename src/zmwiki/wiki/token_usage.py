"""Small, content-free accounting for chat completion responses."""

import json


def response_usage(body: bytes) -> dict[str, int] | None:
    try:
        usage = json.loads(body).get("usage")
    except (ValueError, AttributeError, TypeError):
        return None
    if not isinstance(usage, dict):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in keys):
        return None
    return {key: usage[key] for key in keys}


def usage_report(requests: list[dict[str, int] | None]) -> dict:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    known = all(item is not None for item in requests)
    return {
        "request_count": len(requests),
        "requests": [item.copy() if item is not None else None for item in requests],
        "usage_status": "known" if known else "unknown",
        **{key: sum(item[key] for item in requests if item is not None) if known else None
           for key in keys},
    }
