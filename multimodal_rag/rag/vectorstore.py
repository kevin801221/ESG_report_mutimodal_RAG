from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from ..config.settings import VECTORSTORE_CONFIG, get_api_key

class VectorStoreManager:
    """向量存儲管理器，用於管理文檔的向量化存儲和檢索"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        初始化向量存儲管理器
        
        Args:
            collection_name: 集合名稱
            persist_directory: 持久化目錄路徑
        """
        self.collection_name = collection_name or VECTORSTORE_CONFIG["COLLECTION_NAME"]
        self.persist_directory = persist_directory or VECTORSTORE_CONFIG["PERSIST_DIRECTORY"]
        
        # 確保持久化目錄存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化 OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=get_api_key("OPENAI_API_KEY")
        )
        
        # 初始化向量存儲
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # 初始化文檔存儲
        self.docstore = InMemoryStore()
        
        # 初始化多向量檢索器
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key="doc_id"
        )
        
    def add_texts(
        self,
        texts: List[str],
        summaries: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """
        添加文本及其摘要到向量存儲
        
        Args:
            texts: 原始文本列表
            summaries: 對應的摘要列表
            metadatas: 元數據列表
            
        Returns:
            文檔ID列表
        """
        # 生成唯一ID
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        
        # 創建摘要文檔
        summary_docs = [
            Document(
                page_content=summary,
                metadata={"doc_id": doc_id, **(meta or {})}
            )
            for summary, doc_id, meta in zip(
                summaries,
                doc_ids,
                metadatas or [{}] * len(texts)
            )
        ]
        
        # 添加摘要到向量存儲
        self.retriever.vectorstore.add_documents(summary_docs)
        
        # 添加原始文本到文檔存儲
        self.retriever.docstore.mset(list(zip(doc_ids, texts)))
        
        return doc_ids
    
    def add_images(
        self,
        images: List[str],  # base64 編碼的圖片
        summaries: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """
        添加圖片及其描述到向量存儲
        
        Args:
            images: base64 編碼的圖片列表
            summaries: 圖片描述/摘要列表
            metadatas: 元數據列表
            
        Returns:
            文檔ID列表
        """
        # 生成唯一ID
        img_ids = [str(uuid.uuid4()) for _ in images]
        
        # 創建圖片描述文檔
        summary_docs = [
            Document(
                page_content=summary,
                metadata={
                    "doc_id": img_id,
                    "type": "image",
                    **(meta or {})
                }
            )
            for summary, img_id, meta in zip(
                summaries,
                img_ids,
                metadatas or [{}] * len(images)
            )
        ]
        
        # 添加描述到向量存儲
        self.retriever.vectorstore.add_documents(summary_docs)
        
        # 添加原始圖片數據到文檔存儲
        self.retriever.docstore.mset(list(zip(img_ids, images)))
        
        return img_ids
    
    def add_tables(
        self,
        tables: List[str],  # HTML 格式的表格
        summaries: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """
        添加表格及其描述到向量存儲
        
        Args:
            tables: HTML 格式的表格列表
            summaries: 表格描述/摘要列表
            metadatas: 元數據列表
            
        Returns:
            文檔ID列表
        """
        # 生成唯一ID
        table_ids = [str(uuid.uuid4()) for _ in tables]
        
        # 創建表格描述文檔
        summary_docs = [
            Document(
                page_content=summary,
                metadata={
                    "doc_id": table_id,
                    "type": "table",
                    **(meta or {})
                }
            )
            for summary, table_id, meta in zip(
                summaries,
                table_ids,
                metadatas or [{}] * len(tables)
            )
        ]
        
        # 添加描述到向量存儲
        self.retriever.vectorstore.add_documents(summary_docs)
        
        # 添加原始表格數據到文檔存儲
        self.retriever.docstore.mset(list(zip(table_ids, tables)))
        
        return table_ids
    
    def retrieve(
        self,
        query: str,
        filter_metadata: Optional[dict] = None,
        k: int = 4
    ) -> Dict[str, List]:
        """
        檢索相關文檔
        
        Args:
            query: 查詢文本
            filter_metadata: 過濾元數據
            k: 返回的文檔數量
            
        Returns:
            分類後的檢索結果
        """
        # 執行檢索
        docs = self.retriever.get_relevant_documents(
            query,
            filter=filter_metadata,
            k=k
        )
        
        # 按類型分類結果
        results = {
            "texts": [],
            "images": [],
            "tables": []
        }
        
        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            doc_id = doc.metadata.get("doc_id")
            
            if doc_id:
                original_content = self.retriever.docstore.get(doc_id)
                if doc_type == "image":
                    results["images"].append(original_content)
                elif doc_type == "table":
                    results["tables"].append(original_content)
                else:
                    results["texts"].append(original_content)
        
        return results
    
    def save(self):
        """保存向量存儲到磁盤"""
        self.vectorstore.persist()
    
    def load(self):
        """從磁盤加載向量存儲"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )