from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..config.settings import get_api_key, get_model_config

class OpenAIModel:
    """OpenAI 模型處理器，負責處理多模態（文本+圖像）任務"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        初始化 OpenAI 模型
        
        Args:
            model_name: 模型名稱，如果為 None 則使用配置中的默認值
            temperature: 溫度參數，如果為 None 則使用配置中的默認值
            max_tokens: 最大生成token數，如果為 None 則使用配置中的默認值
        """
        config = get_model_config("OPENAI")
        self.model_name = model_name or config["DEFAULT_MODEL"]
        self.temperature = temperature or config.get("TEMPERATURE", 0.7)
        self.max_tokens = max_tokens or config.get("MAX_TOKENS", 2048)
        self.api_key = get_api_key("OPENAI_API_KEY")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )
    
    def analyze_image(
        self,
        image_data: str,
        prompt: str = "請詳細描述這張圖片的內容。",
        output_format: str = "text"
    ) -> str:
        """
        分析圖片內容
        
        Args:
            image_data: base64 編碼的圖片數據
            prompt: 分析提示
            output_format: 輸出格式 ("text" 或 "json")
            
        Returns:
            分析結果
        """
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )
        ]
        
        if output_format == "json":
            prompt += "\n請以 JSON 格式回答，包含主要元素、主題和描述等字段。"
            
        return self.llm.invoke(messages).content
    
    def batch_analyze_images(
        self,
        images_data: List[str],
        prompt: str = None,
        max_concurrency: int = 3
    ) -> List[str]:
        """
        批量分析圖片
        
        Args:
            images_data: base64 編碼的圖片數據列表
            prompt: 分析提示
            max_concurrency: 最大並發數
            
        Returns:
            分析結果列表
        """
        if prompt is None:
            prompt = """請描述這張圖片的內容。如果是圖表或數據可視化，
            請特別說明圖表類型和主要數據趨勢。"""
            
        template = ChatPromptTemplate.from_messages([
            ("user", [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"}
                }
            ])
        ])
        
        chain = template | self.llm | StrOutputParser()
        return chain.batch(images_data, {"max_concurrency": max_concurrency})
    
    def multimodal_query(
        self,
        query: str,
        context_text: str = "",
        context_images: List[str] = None
    ) -> str:
        """
        執行多模態查詢（結合文本和圖像）
        
        Args:
            query: 查詢問題
            context_text: 上下文文本
            context_images: base64 編碼的圖片列表
            
        Returns:
            回答結果
        """
        prompt_content = [{
            "type": "text",
            "text": f"""請基於以下提供的上下文（包括文本和圖片）來回答問題。
            
            上下文文本：{context_text}
            
            問題：{query}"""
        }]
        
        if context_images:
            for image in context_images:
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                })
        
        messages = [HumanMessage(content=prompt_content)]
        return self.llm.invoke(messages).content
    
    def create_chain_with_image_support(
        self,
        system_prompt: str = None
    ) -> Any:
        """
        創建支持圖像的處理鏈
        
        Args:
            system_prompt: 系統提示
            
        Returns:
            處理鏈
        """
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt or "你是一個專業的多模態助手，能夠理解和分析文本與圖像。"),
            ("user", [
                {"type": "text", "text": "{text_input}"},
                {"type": "image_url", "image_url": {"url": "{image_url}"}}
            ])
        ])
        
        return template | self.llm | StrOutputParser()
    
    def extract_text_from_image(self, image_data: str) -> str:
        """
        從圖片中提取文本（OCR）
        
        Args:
            image_data: base64 編碼的圖片數據
            
        Returns:
            提取的文本
        """
        prompt = """請執行以下任務：
        1. 仔細查看圖片中的所有文本內容
        2. 按照閱讀順序提取所有可見的文字
        3. 保持原始格式和段落結構
        4. 如果是表格，請保持表格結構
        只需返回提取的文本，不需要任何解釋或描述。"""
        
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )
        ]
        
        return self.llm.invoke(messages).content
    
    def analyze_chart(self, image_data: str) -> Dict[str, Any]:
        """
        分析圖表數據
        
        Args:
            image_data: base64 編碼的圖表圖片數據
            
        Returns:
            圖表分析結果
        """
        prompt = """請分析這個圖表並提供以下信息（以JSON格式回答）：
        1. 圖表類型
        2. 主要趨勢或發現
        3. 最大和最小值（如果可見）
        4. 數據範圍
        5. 軸標籤信息
        6. 圖例內容（如果有）"""
        
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )
        ]
        
        response = self.llm.invoke(messages).content
        # 注意：這裡假設返回的是合法的 JSON 字符串
        # 實際使用時可能需要添加錯誤處理
        import json
        return json.loads(response)