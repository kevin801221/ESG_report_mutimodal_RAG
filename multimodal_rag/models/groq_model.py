from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.chains import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..config.settings import get_api_key, get_model_config

class GroqModel:
    """Groq 模型處理器，負責處理文本摘要和生成任務"""
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        初始化 Groq 模型
        
        Args:
            model_name: 模型名稱，如果為 None 則使用配置中的默認值
            temperature: 溫度參數，如果為 None 則使用配置中的默認值
        """
        config = get_model_config("GROQ")
        self.model_name = model_name or config["DEFAULT_MODEL"]
        self.temperature = temperature or config.get("TEMPERATURE", 0.5)
        self.api_key = get_api_key("GROQ_API_KEY")
        
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        
        # 初始化摘要鏈
        self.summarize_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            verbose=True
        )
    
    def create_text_summary_chain(self, prompt_template: str = None) -> Any:
        """
        創建文本摘要處理鏈
        
        Args:
            prompt_template: 自定義提示模板
            
        Returns:
            摘要處理鏈
        """
        if prompt_template is None:
            prompt_template = """
            請提供以下內容的簡潔摘要。
            回答時直接給出摘要，不要加入任何額外的說明。
            不要以「以下是摘要」或類似的句子開頭。

            內容：{element}
            """
            
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = {"element": lambda x: x} | prompt | self.llm | StrOutputParser()
        return chain
    
    def batch_process_texts(self, texts: List[str], prompt_template: str = None, 
                          max_concurrency: int = 3) -> List[str]:
        """
        批量處理文本生成摘要
        
        Args:
            texts: 要處理的文本列表
            prompt_template: 自定義提示模板
            max_concurrency: 最大並發數
            
        Returns:
            摘要列表
        """
        chain = self.create_text_summary_chain(prompt_template)
        return chain.batch(texts, {"max_concurrency": max_concurrency})
    
    def process_documents(self, documents: List[Any], batch_size: int = 10) -> List[str]:
        """
        處理文檔對象生成摘要
        
        Args:
            documents: 文檔對象列表
            batch_size: 批次大小
            
        Returns:
            摘要列表
        """
        summaries = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                summary = self.summarize_chain.invoke({
                    "input_documents": batch
                })
                summaries.append(summary.get('output_text', ''))
            except Exception as e:
                print(f"處理批次 {i//batch_size + 1} 時發生錯誤: {str(e)}")
                summaries.append("")
        
        return summaries
    
    def query_with_context(self, query: str, context: str) -> str:
        """
        基於上下文進行查詢
        
        Args:
            query: 查詢文本
            context: 上下文信息
            
        Returns:
            生成的回答
        """
        prompt_template = """
        基於以下上下文回答問題。如果上下文中沒有足夠的信息，請說明無法回答。

        上下文：{context}
        
        問題：{query}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "context": context,
            "query": query
        })
    
    @staticmethod
    def create_system_prompt(task_description: str) -> str:
        """
        創建系統提示
        
        Args:
            task_description: 任務描述
            
        Returns:
            格式化的系統提示
        """
        return f"""你是一個專業的文本分析助手。
        
        你的任務是：{task_description}
        
        請簡潔、準確地完成任務，避免添加任何不必要的解釋或評論。"""