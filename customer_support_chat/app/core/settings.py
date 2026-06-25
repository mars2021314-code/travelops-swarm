from os import environ
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".dev.env", override=False)

environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
environ.setdefault("no_proxy", "localhost,127.0.0.1")
if not environ.get("LANGCHAIN_API_KEY", "").strip():
    environ["LANGCHAIN_TRACING_V2"] = "false"

class Config:
    # DeepSeek API (primary LLM)
    DEEPSEEK_API_KEY: str = environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    
    # Optional OpenAI API key. The app can run without it.
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = environ.get(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    
    DATA_PATH: str = "./customer_support_chat/data"
    LOG_LEVEL: str = environ.get("LOG_LEVEL", "DEBUG")
    SQLITE_DB_PATH: str = environ.get(
        "SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite"
    )
    DATABASE_URL: str = environ.get("DATABASE_URL", "")
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_PATH: str = environ.get("QDRANT_PATH", "./customer_support_chat/data/qdrant_local")
    REQUIRE_QDRANT_SERVER: bool = environ.get("REQUIRE_QDRANT_SERVER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    REQUIRE_PERSISTENT_CHECKPOINT: bool = environ.get(
        "REQUIRE_PERSISTENT_CHECKPOINT", "false"
    ).lower() in {"1", "true", "yes", "on"}
    CHECKPOINT_DATABASE_URL: str = environ.get(
        "CHECKPOINT_DATABASE_URL", environ.get("DATABASE_URL", "")
    )
    REDIS_URL: str = environ.get("REDIS_URL", "redis://localhost:6379")
    CHAT_CONCURRENCY_LIMIT: int = int(environ.get("CHAT_CONCURRENCY_LIMIT", "20"))
    CHAT_DISTRIBUTED_CONCURRENCY_LIMIT: int = int(
        environ.get(
            "CHAT_DISTRIBUTED_CONCURRENCY_LIMIT",
            environ.get("CHAT_CONCURRENCY_LIMIT", "20"),
        )
    )
    CHAT_REQUEST_TIMEOUT_SECONDS: int = int(
        environ.get("CHAT_REQUEST_TIMEOUT_SECONDS", "180")
    )
    CHAT_RATE_LIMIT_PER_MINUTE: int = int(environ.get("CHAT_RATE_LIMIT_PER_MINUTE", "60"))
    CHAT_USER_RATE_LIMIT_PER_MINUTE: int = int(
        environ.get("CHAT_USER_RATE_LIMIT_PER_MINUTE", environ.get("CHAT_RATE_LIMIT_PER_MINUTE", "60"))
    )
    CHAT_IP_RATE_LIMIT_PER_MINUTE: int = int(environ.get("CHAT_IP_RATE_LIMIT_PER_MINUTE", "120"))
    CHAT_GLOBAL_RATE_LIMIT_PER_MINUTE: int = int(environ.get("CHAT_GLOBAL_RATE_LIMIT_PER_MINUTE", "600"))
    CHAT_RATE_LIMIT_BURST: int = int(environ.get("CHAT_RATE_LIMIT_BURST", "20"))
    CHAT_CACHE_TTL_SECONDS: int = int(environ.get("CHAT_CACHE_TTL_SECONDS", "3600"))
    CHAT_IDEMPOTENCY_TTL_SECONDS: int = int(environ.get("CHAT_IDEMPOTENCY_TTL_SECONDS", "3600"))
    CHAT_IDEMPOTENCY_PENDING_TTL_SECONDS: int = int(
        environ.get("CHAT_IDEMPOTENCY_PENDING_TTL_SECONDS", "300")
    )
    REDIS_QUERY_CACHE_TTL_SECONDS: int = int(environ.get("REDIS_QUERY_CACHE_TTL_SECONDS", "300"))
    REDIS_WRITE_LOCK_TTL_SECONDS: int = int(environ.get("REDIS_WRITE_LOCK_TTL_SECONDS", "30"))
    REDIS_LOCK_WAIT_TIMEOUT_SECONDS: float = float(
        environ.get("REDIS_LOCK_WAIT_TIMEOUT_SECONDS", "5")
    )
    LLM_TIMEOUT_SECONDS: int = int(environ.get("LLM_TIMEOUT_SECONDS", "60"))
    LLM_MAX_RETRIES: int = int(environ.get("LLM_MAX_RETRIES", "2"))
    SQLITE_BUSY_TIMEOUT_MS: int = int(environ.get("SQLITE_BUSY_TIMEOUT_MS", "5000"))
    RECREATE_COLLECTIONS: bool = environ.get("RECREATE_COLLECTIONS", "False")
    LIMIT_ROWS: int = environ.get("LIMIT_ROWS", "100")
    MEMORY_TOP_K: int = int(environ.get("MEMORY_TOP_K", "8"))
    MEMORY_HANDOFF_TTL_HOURS: int = int(environ.get("MEMORY_HANDOFF_TTL_HOURS", "24"))
    MEMORY_TOOL_OBSERVATION_TTL_HOURS: int = int(environ.get("MEMORY_TOOL_OBSERVATION_TTL_HOURS", "24"))
    MEMORY_CANDIDATE_TTL_HOURS: int = int(environ.get("MEMORY_CANDIDATE_TTL_HOURS", "12"))
    MEMORY_POLICY_TTL_HOURS: int = int(environ.get("MEMORY_POLICY_TTL_HOURS", "24"))
    MEMORY_OPEN_LOOP_TTL_HOURS: int = int(environ.get("MEMORY_OPEN_LOOP_TTL_HOURS", "72"))
    MEMORY_SESSION_SUMMARY_TTL_HOURS: int = int(environ.get("MEMORY_SESSION_SUMMARY_TTL_HOURS", "168"))
    MEMORY_AUDIT_LOG_ENABLED: bool = environ.get(
        "MEMORY_AUDIT_LOG_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}

def get_settings():
    return Config()
