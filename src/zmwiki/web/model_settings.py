"""Server-side model configuration for the local workbench.

Three responsibilities, all standard library:

* ``ModelConfig`` -- the validated Base URL / model name / API key triple.
  The key is excluded from ``repr`` so it cannot leak into logs or tracebacks.
* ``SettingsStore`` -- load/save that triple as a user-only JSON file
  (``~/.zmwiki/settings.json`` by default, mode 0600 on POSIX).
* ``test_model_connection`` -- one minimal chat-completions request whose
  result is mapped to a fixed, actionable message. Response bodies and
  exception text are never echoed: an upstream that reflects request headers
  must not be able to leak the key back to the browser.

The key is never returned by any function here; callers only learn whether
one is present.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from zmwiki.llm_generators import (
    HttpRequest,
    HttpResponse,
    Transport,
    urllib_transport,
)

__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "ConfigError",
    "ConnectionTestResult",
    "ModelConfig",
    "SettingsStore",
    "make_config",
    "parse_config_payload",
    "test_model_connection",
]

DEFAULT_SETTINGS_PATH = Path.home() / ".zmwiki" / "settings.json"

MAX_BASE_URL_CHARS = 500
MAX_MODEL_CHARS = 200
MAX_API_KEY_CHARS = 4096
TEST_TIMEOUT_SECONDS = 10.0


class ConfigError(ValueError):
    """The supplied model configuration is missing or malformed."""


@dataclass(frozen=True)
class ModelConfig:
    """A validated model endpoint configuration.

    ``api_key`` is omitted from ``repr`` on purpose: this object is held by the
    server and may be inspected in a debugger or log line.
    """

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ConnectionTestResult:
    ok: bool
    code: str
    message: str


def _clean_base_url(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "Base URL is required (for example https://api.deepseek.com/v1)"
        )
    url = value.strip()
    if not url:
        raise ConfigError(
            "Base URL is required (for example https://api.deepseek.com/v1)"
        )
    if len(url) > MAX_BASE_URL_CHARS:
        raise ConfigError(f"Base URL must be at most {MAX_BASE_URL_CHARS} characters")
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise ConfigError("Base URL must not contain spaces or control characters")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise ConfigError("Base URL is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ConfigError(
            "Base URL must be an absolute http(s) URL, "
            "for example https://api.deepseek.com/v1"
        )
    if parts.username is not None or parts.password is not None:
        raise ConfigError(
            "Base URL must not embed credentials; use the API key field"
        )
    if parts.query or parts.fragment:
        raise ConfigError("Base URL must not contain a query string or fragment")
    return url.rstrip("/")


def _clean_model(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError("Model name is required (for example deepseek-chat)")
    name = value.strip()
    if not name:
        raise ConfigError("Model name is required (for example deepseek-chat)")
    if len(name) > MAX_MODEL_CHARS:
        raise ConfigError(f"Model name must be at most {MAX_MODEL_CHARS} characters")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        raise ConfigError("Model name must not contain control characters")
    return name


def _clean_api_key(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError("API key must be a string")
    key = value.strip()
    if not key:
        raise ConfigError("API key must not be empty (leave it blank to keep the current key)")
    if len(key) > MAX_API_KEY_CHARS:
        raise ConfigError(f"API key must be at most {MAX_API_KEY_CHARS} characters")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in key):
        raise ConfigError("API key must not contain control characters")
    return key


def parse_config_payload(
    payload: dict, *, existing: ModelConfig | None = None
) -> ModelConfig:
    """Validate a settings payload.

    ``api_key`` omitted or blank keeps ``existing.api_key`` so the browser can
    submit the form without ever receiving the stored key. ``clear_api_key``
    explicitly removes it (useful for anonymous local endpoints).
    """
    if not isinstance(payload, dict):
        raise ConfigError("model settings must be a JSON object")
    base_url = _clean_base_url(payload.get("base_url"))
    model = _clean_model(payload.get("model"))

    clear = payload.get("clear_api_key", False)
    if not isinstance(clear, bool):
        raise ConfigError("clear_api_key must be true or false")
    raw_key = payload.get("api_key")
    if clear:
        api_key = None
    elif raw_key is None or (isinstance(raw_key, str) and not raw_key.strip()):
        api_key = existing.api_key if existing is not None else None
    else:
        api_key = _clean_api_key(raw_key)

    return ModelConfig(base_url=base_url, model=model, api_key=api_key)


def make_config(
    base_url: object, model: object, api_key: object = None
) -> ModelConfig:
    """Build a config from CLI values, applying the same validation as the UI."""
    return parse_config_payload(
        {"base_url": base_url, "model": model, "api_key": api_key}
    )


class SettingsStore:
    """One JSON file holding the model configuration for this machine.

    ``path=None`` disables the store (the caller then keeps config in memory).
    A missing or invalid file simply means "not configured" -- loading never
    raises, so a hand-edited or truncated file cannot stop the server.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH

    @property
    def persisted(self) -> bool:
        return self.path.is_file()

    def load(self) -> ModelConfig | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            return None
        if not isinstance(payload, dict):
            return None
        version = payload.get("version")
        if isinstance(version, bool) or version != 1:
            return None
        try:
            return parse_config_payload(payload)
        except (ConfigError, TypeError):
            return None

    def save(self, config: ModelConfig) -> None:
        """Persist ``config`` atomically with user-only permissions."""
        payload: dict = {
            "version": 1,
            "base_url": config.base_url,
            "model": config.model,
        }
        if config.api_key is not None:
            payload["api_key"] = config.api_key
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            raw = json.dumps(payload, ensure_ascii=False, indent=2)
            fd, tmp_name = tempfile.mkstemp(
                dir=self.path.parent, prefix=".settings-", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(raw)
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    pass  # Windows: user-profile ACLs already scope the file
                os.replace(tmp_path, self.path)
                tmp_path = None
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
        except OSError:
            raise ConfigError(
                "model settings cannot be written; check permissions for the "
                "settings file"
            ) from None


def test_model_connection(
    config: ModelConfig,
    *,
    transport: Transport | None = None,
    timeout: float = TEST_TIMEOUT_SECONDS,
) -> ConnectionTestResult:
    """Send one minimal completion request and map the outcome safely.

    Diagnostics never include the response body or transport exception text:
    a hostile or broken endpoint could reflect the Authorization header into
    either, which would expose the key to the browser.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError(
            f"config must be a ModelConfig, got {type(config).__name__}"
        )
    body = json.dumps({
        "model": config.model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "temperature": 0.0,
        "max_tokens": 1,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if config.api_key is not None:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = HttpRequest(
        url=f"{config.base_url}/chat/completions",
        method="POST",
        headers=headers,
        body=body,
        timeout=timeout,
    )
    try:
        response = (transport or urllib_transport)(request)
    except Exception:
        return ConnectionTestResult(
            False,
            "connection_failed",
            "Could not reach the model endpoint. Check the Base URL, network "
            "connection, and proxy settings.",
        )

    if not isinstance(response, HttpResponse):
        return ConnectionTestResult(
            False, "invalid_response", "The model transport returned an unexpected response."
        )

    if 200 <= response.status < 300:
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices:
            return ConnectionTestResult(
                True, "ok", "Connected. The endpoint returned a chat completion."
            )
        return ConnectionTestResult(
            False,
            "invalid_response",
            "The endpoint responded but not with an OpenAI-compatible chat "
            "completion. Check that the Base URL points at the API root "
            "(it usually ends with /v1).",
        )

    status = response.status
    if status in (401, 403):
        return ConnectionTestResult(
            False,
            "auth_failed",
            "Authentication failed. Check the API key (and that it is allowed "
            "to use this model).",
        )
    if status == 404:
        return ConnectionTestResult(
            False,
            "endpoint_not_found",
            "Endpoint not found. Check the Base URL; providers usually expose "
            "the API under a /v1 path.",
        )
    if status == 400:
        return ConnectionTestResult(
            False,
            "bad_request",
            "The endpoint rejected the request. Check the model name; some "
            "providers require an exact deployment name.",
        )
    if status == 429:
        return ConnectionTestResult(
            False,
            "rate_limited",
            "The endpoint is rate limiting requests. Wait and retry, or check "
            "the account quota.",
        )
    if status >= 500:
        return ConnectionTestResult(
            False,
            "service_error",
            f"The model service returned HTTP {status}. Retry later or check "
            "the provider status page.",
        )
    return ConnectionTestResult(
        False,
        "failed",
        f"The endpoint returned HTTP {status}. Check the Base URL and model name.",
    )
