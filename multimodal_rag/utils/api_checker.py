from typing import Dict, Optional, Tuple
import os
from groq import Groq
from langsmith import Client
from openai import OpenAI
# from ..config.settings import get_api_key, MODEL_CONFIG
from multimodal_rag.config.settings import get_api_key, MODEL_CONFIG
class APIChecker:
    """API 連接狀態檢查器"""
    
    @staticmethod
    def check_groq_api() -> Tuple[bool, str]:
        """
        檢查 Groq API 的連接狀態
        
        Returns:
            Tuple[bool, str]: (是否連接成功, 狀態消息)
        """
        try:
            api_key = get_api_key("GROQ_API_KEY")
            client = Groq(api_key=api_key)
            
            # 測試簡單的完成請求
            completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": "Test connection"
                }],
                model=MODEL_CONFIG["GROQ"]["DEFAULT_MODEL"]
            )
            
            return True, "Groq API 連接成功"
            
        except Exception as e:
            return False, f"Groq API 連接失敗: {str(e)}"
    
    @staticmethod
    def check_langchain_api() -> Tuple[bool, str]:
        """
        檢查 LangChain API 的連接狀態
        
        Returns:
            Tuple[bool, str]: (是否連接成功, 狀態消息)
        """
        try:
            client = Client()
            # 嘗試獲取項目列表來測試連接
            projects = client.list_projects()
            return True, "LangChain API 連接成功"
            
        except Exception as e:
            return False, f"LangChain API 連接失敗: {str(e)}"
    
    @staticmethod
    def check_openai_api() -> Tuple[bool, str]:
        """
        檢查 OpenAI API 的連接狀態
        
        Returns:
            Tuple[bool, str]: (是否連接成功, 狀態消息)
        """
        try:
            api_key = get_api_key("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            
            # 測試簡單的完成請求
            response = client.chat.completions.create(
                model=MODEL_CONFIG["OPENAI"]["DEFAULT_MODEL"],
                messages=[{
                    "role": "user",
                    "content": "Test connection"
                }]
            )
            
            return True, "OpenAI API 連接成功"
            
        except Exception as e:
            return False, f"OpenAI API 連接失敗: {str(e)}"
    
    @classmethod
    def check_all_apis(cls) -> Dict[str, Dict[str, any]]:
        """
        檢查所有 API 的連接狀態
        
        Returns:
            Dict[str, Dict[str, any]]: 各 API 的狀態信息
        """
        results = {}
        
        # 檢查 Groq API
        groq_status, groq_message = cls.check_groq_api()
        results["Groq"] = {
            "status": groq_status,
            "message": groq_message
        }
        
        # 檢查 LangChain API
        langchain_status, langchain_message = cls.check_langchain_api()
        results["LangChain"] = {
            "status": langchain_status,
            "message": langchain_message
        }
        
        # 檢查 OpenAI API
        openai_status, openai_message = cls.check_openai_api()
        results["OpenAI"] = {
            "status": openai_status,
            "message": openai_message
        }
        
        return results
    
    @staticmethod
    def format_status_report(results: Dict[str, Dict[str, any]]) -> str:
        """
        格式化 API 狀態報告
        
        Args:
            results: API 檢查結果
            
        Returns:
            str: 格式化的狀態報告
        """
        report = ["API 狀態報告:", "=" * 50]
        
        for api_name, status in results.items():
            status_symbol = "✅" if status["status"] else "❌"
            report.append(f"{status_symbol} {api_name}: {status['message']}")
        
        return "\n".join(report)

def verify_api_connections() -> bool:
    """
    驗證所有 API 連接並輸出狀態報告
    
    Returns:
        bool: 是否所有 API 都連接成功
    """
    checker = APIChecker()
    results = checker.check_all_apis()
    
    # 輸出狀態報告
    print(checker.format_status_report(results))
    
    # 檢查是否所有 API 都連接成功
    return all(status["status"] for status in results.values())

if __name__ == "__main__":
    verify_api_connections()