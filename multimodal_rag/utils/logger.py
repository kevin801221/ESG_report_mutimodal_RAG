# multimodal_rag/multimodal_rag/utils/logger.py
import logging
from typing import Optional

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    獲取 logger 實例
    
    Args:
        name: logger 名稱
        
    Returns:
        logging.Logger: logger 實例
    """
    # 創建 logger
    logger = logging.getLogger(name or __name__)
    
    # 如果 logger 已經有處理器，直接返回
    if logger.handlers:
        return logger
        
    # 設置日誌級別
    logger.setLevel(logging.INFO)
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 設置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加處理器
    logger.addHandler(console_handler)
    
    return logger