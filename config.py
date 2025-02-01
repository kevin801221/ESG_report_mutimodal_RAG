# config.py
import os
from pathlib import Path

# 基礎路徑配置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# 確保必要目錄存在
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型配置
GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-4o-mini"

# PDF處理配置
PDF_PROCESSING = {
    "max_characters": 10000,
    "combine_text_under_n_chars": 2000,
    "new_after_n_chars": 6000
}

# 向量存儲配置
VECTOR_STORE = {
    "collection_name": "multi_modal_rag",
    "persist_directory": str(OUTPUT_DIR / "chroma_db")
}