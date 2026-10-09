"""Settings read from environment variables (.env). Every parameter has a sensible default."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


_DEFAULTS = {
    "openai": dict(llm="gpt-4o-mini", judge="gpt-4.1-mini", rpm=300, p_in=0.15, p_out=0.60, embed="text-embedding-3-small", dim=1536),
    "fake": dict(llm="fake", judge="fake", rpm=6000, p_in=0.0, p_out=0.0, embed="hash", dim=512),
    "hash": dict(llm="fake", judge="fake", rpm=6000, p_in=0.0, p_out=0.0, embed="hash", dim=512),
}


def _env(name: str, default, cast=str):
    value = os.getenv(name)
    return cast(default if value in (None, "") else value)


@dataclass
class Settings:
    # data
    kb_dir: Path = ROOT / "data" / "kb"
    index_dir: Path = ROOT / "data" / "index"
    eval_file: Path = ROOT / "data" / "eval" / "tickets.jsonl"
    cache_dir: Path = ROOT / ".cache"
    reports_dir: Path = ROOT / "reports"
    chunk_chars: int = field(default_factory=lambda: _env("CHUNK_CHARS", 300, int))

    # LLM: openai (default) | fake (offline, tests only)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", _DEFAULTS[_env("LLM_PROVIDER", "openai")]["llm"]))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", _DEFAULTS[_env("LLM_PROVIDER", "openai")]["judge"]))
    llm_rpm: int = field(default_factory=lambda: _env("LLM_RPM", _DEFAULTS[_env("LLM_PROVIDER", "openai")]["rpm"], int))
    # USD per 1M tokens, used to estimate cost per ticket (check the provider price list)
    price_in_per_m: float = field(default_factory=lambda: _env("PRICE_INPUT_PER_M", _DEFAULTS[_env("LLM_PROVIDER", "openai")]["p_in"], float))
    price_out_per_m: float = field(default_factory=lambda: _env("PRICE_OUTPUT_PER_M", _DEFAULTS[_env("LLM_PROVIDER", "openai")]["p_out"], float))

    # embedding: openai (default) | hash (offline, tests only)
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "openai"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", _DEFAULTS[_env("EMBED_PROVIDER", "openai")]["embed"]))
    embed_dim: int = field(default_factory=lambda: _env("EMBED_DIM", _DEFAULTS[_env("EMBED_PROVIDER", "openai")]["dim"], int))

    # reranker: bge (cross-encoder BAAI/bge-reranker-v2-m3) | llm | none
    reranker: str = field(default_factory=lambda: _env("RERANKER", "bge"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))

    # retrieval
    top_k_retrieve: int = field(default_factory=lambda: _env("TOP_K_RETRIEVE", 20, int))
    top_k_context: int = field(default_factory=lambda: _env("TOP_K_CONTEXT", 6, int))
    rrf_k: int = 60
    max_rounds: int = field(default_factory=lambda: _env("CRAG_MAX_ROUNDS", 3, int))


settings = Settings()
