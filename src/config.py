import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"
CHUNKS_PATH = CHUNKS_DIR / "all_chunks.json"

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
EMBEDDING_LOCATION = os.getenv("EMBEDDING_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

MODEL_FLASH = "heuristic"
MODEL_FLASH_LITE = "heuristic"
MODEL_EMBEDDING = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

RANDOM_SEED = 42
TOP_K = 5
