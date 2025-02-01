# utils/rag_pipeline.py
import uuid
from typing import List, Any
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from base64 import b64decode

class RAGPipeline:
    def __init__(self, vectorstore, docstore, llm):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.llm = llm
        self.id_key = "doc_id"

    def load_summaries(
        self,
        texts: List[Any],
        text_summaries: List[str],
        tables: List[Any],
        table_summaries: List[str],
        images: List[str],
        image_summaries: List[str]
    ) -> None:
        """加載所有摘要到向量存储"""
        # 加載文本摘要
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        self.vectorstore.add_documents(summary_texts)
        self.docstore.mset(list(zip(doc_ids, texts)))

        # 加載表格摘要
        if tables and table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]})
                for i, summary in enumerate(table_summaries)
            ]
            self.vectorstore.add_documents(summary_tables)
            self.docstore.mset(list(zip(table_ids, tables)))

        # 加載圖像摘要
        if images and image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in images]
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]})
                for i, summary in enumerate(image_summaries)
            ]
            self.vectorstore.add_documents(summary_img)
            self.docstore.mset(list(zip(img_ids, images)))

    def query(self, question: str) -> str:
        """處理查詢並返回回答"""
        docs = self.vectorstore.similarity_search(question)
        
        context_text = "\n".join([doc.page_content for doc in docs])
        
        prompt_template = f"""
        Answer the question based only on the following context:
        Context: {context_text}
        Question: {question}
        """
        
        messages = [HumanMessage(content=prompt_template)]
        response = self.llm.invoke(messages)
        
        return response.content