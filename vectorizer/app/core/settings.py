from os import environ
from dotenv import load_dotenv


load_dotenv()
load_dotenv(".dev.env", override=False)

environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
environ.setdefault("no_proxy", "localhost,127.0.0.1")
if not environ.get("LANGCHAIN_API_KEY", "").strip():
    environ["LANGCHAIN_TRACING_V2"] = "false"

class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY", "")
    SQLITE_DB_PATH: str = environ.get("SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite")
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_PATH: str = environ.get("QDRANT_PATH", "./customer_support_chat/data/qdrant_local")

def get_settings():
    return Config()
