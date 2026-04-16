# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
KAFU_API_TOKEN: str = os.getenv("KAFU_API_TOKEN", "dev-token")

QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./knowledge_base/qdrant_data")
MANUALS_DIR: Path = Path(os.getenv("MANUALS_DIR", "./手册"))
IMAGES_DIR: Path = Path(os.getenv("IMAGES_DIR", "./手册/插图"))
DOCSTORE_PATH: Path = Path(os.getenv("DOCSTORE_PATH", "./knowledge_base/docstore.json"))
POLICY_DIR: Path = Path("./knowledge_base/policy")

MANUALS_COLLECTION = "manuals"
POLICY_COLLECTION = "policy"

RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
RETRIEVAL_TOP_K: int = 5
MAX_IMAGES_PER_ANSWER: int = int(os.getenv("MAX_IMAGES_PER_ANSWER", "5"))
MAX_VALIDATOR_RETRIES: int = 2
CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
