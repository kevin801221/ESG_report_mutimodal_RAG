from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
import json

class MultiModalRetriever:
    """多模態檢索器，整合文本、圖像和表格的檢索功能"""
    
    def __init__(self, vectorstore_manager, openai_model, groq_model=None):
        """
        初始化多模態檢索器
        
        Args:
            vectorstore_manager: 向量存儲管理器實例
            openai_model: OpenAI 模型實例（用於處理多模態查詢）
            groq_model: Groq 模型實例（可選，用於文本處理）
        """
        self.vectorstore = vectorstore_manager
        self.openai_model = openai_model
        self.groq_model = groq_model
    
    def parse_retrieval_results(self, docs: List[Any]) -> Dict[str, List]:
        """
        解析檢索結果，將不同類型的文檔分類
        
        Args:
            docs: 檢索到的文檔列表
            
        Returns:
            按類型分類的文檔字典
        """
        try:
            return {
                "images": [doc for doc in docs if doc.metadata.get("type") == "image"],
                "texts": [doc for doc in docs if doc.metadata.get("type") in (None, "text")],
                "tables": [doc for doc in docs if doc.metadata.get("type") == "table"]
            }
        except Exception as e:
            print(f"解析檢索結果時發生錯誤: {str(e)}")
            return {"images": [], "texts": [], "tables": []}
    
    def build_prompt(self, question: str, context: Dict[str, List]) -> List[Dict]:
        """
        構建多模態提示
        
        Args:
            question: 用戶問題
            context: 檢索到的上下文
            
        Returns:
            格式化的提示消息列表
        """
        prompt_content = []
        
        # 添加文本上下文
        context_text = ""
        if context["texts"]:
            context_text += "\n文本上下文:\n"
            for doc in context["texts"]:
                context_text += f"{doc.page_content}\n"
        
        if context["tables"]:
            context_text += "\n表格上下文:\n"
            for doc in context["tables"]:
                context_text += f"{doc.page_content}\n"
                
        prompt_content.append({
            "type": "text",
            "text": f"""基於提供的上下文（包括文本、表格和圖片）來回答問題。
            
            {context_text}
            
            問題: {question}"""
        })
        
        # 添加圖片上下文
        for img_doc in context["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_doc.page_content}"
                }
            })
            
        return prompt_content
    
    def retrieve_and_process(
        self,
        query: str,
        filter_metadata: Optional[dict] = None,
        k: int = 4,
        rerank: bool = True
    ) -> Tuple[str, Dict[str, List]]:
        """
        檢索並處理查詢
        
        Args:
            query: 查詢文本
            filter_metadata: 過濾條件
            k: 檢索文檔數量
            rerank: 是否重新排序結果
            
        Returns:
            處理後的回答和原始檢索結果
        """
        # 執行檢索
        raw_docs = self.vectorstore.retrieve(query, filter_metadata, k)
        
        # 解析結果
        context = self.parse_retrieval_results(raw_docs)
        
        # 構建提示
        prompt_content = self.build_prompt(query, context)
        
        # 使用 OpenAI 模型生成回答
        messages = [HumanMessage(content=prompt_content)]
        response = self.openai_model.llm.invoke(messages)
        
        return response.content, raw_docs
    
    def create_qa_chain(self):
        """
        創建問答鏈
        
        Returns:
            可執行的問答鏈
        """
        def _combine_documents(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        retrieve_and_transform = RunnableLambda(
            lambda x: {
                "context": _combine_documents(
                    self.vectorstore.retrieve(x["question"])
                ),
                "question": x["question"]
            }
        )
        
        prompt_template = """基於以下上下文回答問題。
        如果上下文中沒有足夠的信息，請說明無法回答。
        保持回答準確和簡潔。

        上下文: {context}
        
        問題: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        model = self.groq_model.llm if self.groq_model else self.openai_model.llm
        
        chain = (
            retrieve_and_transform 
            | prompt 
            | model 
            | StrOutputParser()
        )
        
        return chain
    
    def semantic_search(
        self,
        query: str,
        threshold: float = 0.7,
        filter_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        執行語義搜索
        
        Args:
            query: 查詢文本
            threshold: 相似度閾值
            filter_metadata: 過濾條件
            
        Returns:
            相關文檔列表
        """
        # 使用向量存儲的相似度搜索
        results = self.vectorstore.vectorstore.similarity_search_with_score(
            query,
            filter=filter_metadata
        )
        
        # 過濾低於閾值的結果
        filtered_results = [
            doc for doc, score in results
            if score >= threshold
        ]
        
        return filtered_results
    
    def analyze_retrieval_quality(
        self,
        query: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        分析檢索質量
        
        Args:
            query: 查詢文本
            retrieved_docs: 檢索到的文檔
            
        Returns:
            質量分析結果
        """
        # 使用 Groq 模型評估相關性
        if self.groq_model:
            eval_prompt = f"""分析以下查詢與檢索結果的相關性：
            
            查詢：{query}
            
            檢索結果：
            {[doc.page_content for doc in retrieved_docs]}
            
            請提供以下分析（以JSON格式）：
            1. 相關性評分（0-1）
            2. 檢索結果的多樣性評分（0-1）
            3. 建議的改進方向
            """
            
            response = self.groq_model.llm.invoke(eval_prompt)
            try:
                return json.loads(response.content)
            except:
                return {
                    "error": "無法解析評估結果",
                    "raw_response": response.content
                }
        
        return {
            "error": "未提供 Groq 模型，無法執行質量分析"
        }