from typing import Dict, Any
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# API Keys 配置
API_KEYS = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
}

# LangChain 配置
LANGCHAIN_CONFIG = {
    "TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
    "ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    "PROJECT": os.getenv("LANGCHAIN_PROJECT", "default-project")
}

# 模型配置
MODEL_CONFIG = {
    "GROQ": {
        "DEFAULT_MODEL": os.getenv("DEFAULT_GROQ_MODEL", "llama-3.1-8b-instant"),
        "TEMPERATURE": 0.5,
        "MAX_TOKENS": 1024
    },
    "OPENAI": {
        "DEFAULT_MODEL": os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o"),
        "TEMPERATURE": 0.7,
        "MAX_TOKENS": 2048
    }
}

# PDF 處理配置
PDF_PROCESSING = {
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1000)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 200)),
    "STRATEGY": "fast",
    "CHUNKING_STRATEGY": "basic"
}

# NLTK 資源配置
NLTK_RESOURCES = [
    'punkt',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'punkt_tab',
    'popular'
]

# 向量存儲配置
VECTORSTORE_CONFIG = {
    "COLLECTION_NAME": "multi_modal_rag",
    "PERSIST_DIRECTORY": "data/vectorstore"
}

def get_api_key(key_name: str) -> str:
    """
    獲取指定的 API key
    
    Args:
        key_name: API key 的名稱
        
    Returns:
        str: API key 的值
        
    Raises:
        ValueError: 如果 API key 未設置
    """
    api_key = API_KEYS.get(key_name)
    if not api_key:
        raise ValueError(f"{key_name} not found in environment variables")
    return api_key

def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    獲取指定模型的配置
    
    Args:
        model_type: 模型類型 ("GROQ" 或 "OPENAI")
        
    Returns:
        Dict[str, Any]: 模型配置
        
    Raises:
        ValueError: 如果模型類型不存在
    """
    config = MODEL_CONFIG.get(model_type.upper())
    if not config:
        raise ValueError(f"Unknown model type: {model_type}")
    return config

def verify_api_keys() -> bool:
    """
    驗證所有必要的 API keys 是否都已設置
    
    Returns:
        bool: 是否所有必要的 API keys 都已設置
    """
    return all(API_KEYS.values())