# # utils/summarizer.py
# from typing import List, Any
# from langchain.schema import Document
# from langchain.chains import load_summarize_chain

# class Summarizer:
#     def __init__(self, groq_llm, openai_llm):
#         self.groq_llm = groq_llm
#         self.openai_llm = openai_llm
        
#     def summarize_texts(self, texts: List[Any]) -> List[str]:
#         """為文本生成摘要"""
#         summarize_chain = load_summarize_chain(
#             self.groq_llm,
#             chain_type="map_reduce",
#             verbose=True
#         )
        
#         summaries = []
#         for text in texts:
#             doc = Document(page_content=str(text))
#             summary = summarize_chain.invoke({"input_documents": [doc]})
#             summaries.append(summary.get('output_text', ''))
        
#         return summaries
    
#     def summarize_tables(self, tables: List[Any]) -> List[str]:
#         """為表格生成摘要"""
#         # 實現表格摘要邏輯
#         pass
    
#     def summarize_images(self, images: List[str]) -> List[str]:
#         """為圖像生成摘要"""
#         prompt_template = """Describe the image in detail. For context,
#                           the image is part of a research paper explaining the transformers
#                           architecture. Be specific about graphs, such as bar plots."""
                          
#         messages = [
#             (
#                 "user",
#                 [
#                     {"type": "text", "text": prompt_template},
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": "data:image/jpeg;base64,{image}"},
#                     },
#                 ],
#             )
#         ]
        
#         prompt = ChatPromptTemplate.from_messages(messages)
#         chain = prompt | self.openai_llm | StrOutputParser()
        
#         return chain.batch(images)
# utils/summarizer.py
from typing import List, Any
from langchain.schema import Document
from langchain.chains import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate  # 添加這個導入
from langchain_core.output_parsers import StrOutputParser  # 添加這個導入

class Summarizer:
    def __init__(self, groq_llm, openai_llm):
        self.groq_llm = groq_llm
        self.openai_llm = openai_llm
        
    def summarize_texts(self, texts: List[Any]) -> List[str]:
        """為文本生成摘要"""
        summarize_chain = load_summarize_chain(
            self.groq_llm,
            chain_type="map_reduce",
            verbose=True
        )
        
        summaries = []
        for text in texts:
            doc = Document(page_content=str(text))
            summary = summarize_chain.invoke({"input_documents": [doc]})
            summaries.append(summary.get('output_text', ''))
        
        return summaries
    
    def summarize_tables(self, tables: List[Any]) -> List[str]:
        """為表格生成摘要"""
        if not tables:
            return []
            
        summaries = []
        for table in tables:
            summary = f"Table with {len(table.rows)} rows and {len(table.columns)} columns"
            summaries.append(summary)
        
        return summaries
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """為圖像生成摘要"""
        if not images:
            return []
            
        prompt_template = """Describe the image in detail. For context,
                          the image is part of a research paper explaining the transformers
                          architecture. Be specific about graphs, such as bar plots."""
                          
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.openai_llm | StrOutputParser()
        
        return chain.batch(images)