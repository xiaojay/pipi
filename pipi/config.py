from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "pi"
CONFIG_DIR_NAME = ".pi"
ENV_AGENT_DIR = "PI_CODING_AGENT_DIR"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    env_vars: tuple[str, ...]
    default_headers: dict[str, str] = field(default_factory=dict)


OPENAI_COMPAT_PROVIDERS: dict[str, ProviderDefaults] = {
    "openai": ProviderDefaults("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "openrouter": ProviderDefaults(
        "https://openrouter.ai/api/v1",
        ("OPENROUTER_API_KEY",),
        {"HTTP-Referer": "https://github.com/badlogic/pi-mono", "X-Title": "pipi"},
    ),
    "groq": ProviderDefaults("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "xai": ProviderDefaults("https://api.x.ai/v1", ("XAI_API_KEY",)),
    "cerebras": ProviderDefaults("https://api.cerebras.ai/v1", ("CEREBRAS_API_KEY",)),
    "mistral": ProviderDefaults("https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "local": ProviderDefaults("http://localhost:11434/v1", ("OPENAI_API_KEY",)),
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    thinking_level: str = "medium"
    headers: dict[str, str] = field(default_factory=dict)


def get_agent_dir() -> Path:
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / CONFIG_DIR_NAME / "agent"


def get_sessions_dir() -> Path:
    return get_agent_dir() / "sessions"


def get_default_session_dir(cwd: str | Path) -> Path:
    cwd_str = str(Path(cwd).resolve())
    safe_path = "--" + cwd_str.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-") + "--"
    return get_sessions_dir() / safe_path


def _resolve_provider_defaults(provider: str) -> ProviderDefaults:
    return OPENAI_COMPAT_PROVIDERS.get(provider, OPENAI_COMPAT_PROVIDERS["openai"])


def resolve_api_key(provider: str, explicit_api_key: str | None = None, base_url: str | None = None) -> str:
    if explicit_api_key:
        return explicit_api_key
    defaults = _resolve_provider_defaults(provider)
    for env_var in defaults.env_vars:
        value = os.environ.get(env_var)
        if value:
            return value
    if base_url and "localhost" in base_url:
        return "dummy"
    if provider == "local":
        return "dummy"
    raise ValueError(
        f"No API key found for provider '{provider}'. "
        f"Set one of {', '.join(defaults.env_vars)} or pass --api-key."
    )


def resolve_model_config(
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    thinking_level: str,
) -> ModelConfig:
    resolved_provider = provider or os.environ.get("PI_PY_PROVIDER", "openai")
    defaults = _resolve_provider_defaults(resolved_provider)
    resolved_model = model or os.environ.get("PI_PY_MODEL", DEFAULT_MODEL)
    resolved_base_url = base_url or os.environ.get("PI_PY_BASE_URL") or defaults.base_url
    resolved_api_key = resolve_api_key(resolved_provider, api_key, resolved_base_url)
    return ModelConfig(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url.rstrip("/"),
        api_key=resolved_api_key,
        thinking_level=thinking_level,
        headers=dict(defaults.default_headers),
    )
