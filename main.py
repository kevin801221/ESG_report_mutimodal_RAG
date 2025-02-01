# # main.py
# import os
# import logging
# from dotenv import load_dotenv
# from pathlib import Path
# from typing import List, Dict, Any

# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.storage import InMemoryStore
# from langchain.chains import load_summarize_chain
# from langchain.schema import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# from utils.pdf_processor import PDFProcessor
# from utils.summarizer import Summarizer
# from utils.rag_pipeline import RAGPipeline

# # 設置日誌
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MultiModalRAG:
#     def __init__(self):
#         load_dotenv()  # 載入環境變量
        
#         # 初始化必要的API客户端
#         self.groq_llm = ChatGroq(
#             temperature=0,
#             model_name="llama-3.1-8b-instant"
#         )
        
#         self.openai_llm = ChatOpenAI(model="gpt-4o-mini")
        
#         # 初始化向量存储
#         self.vectorstore = Chroma(
#             collection_name="multi_modal_rag",
#             embedding_function=OpenAIEmbeddings()
#         )
        
#         # 初始化文檔存储
#         self.docstore = InMemoryStore()
        
#         # 初始化處理器
#         self.pdf_processor = PDFProcessor()
#         self.summarizer = Summarizer(self.groq_llm, self.openai_llm)
#         self.rag_pipeline = RAGPipeline(
#             self.vectorstore,
#             self.docstore,
#             self.openai_llm
#         )

#     def process_document(self, file_path: str) -> None:
#         """處理輸入的PDF文檔"""
#         try:
#             # 提取文檔內容
#             texts, tables, images = self.pdf_processor.process_pdf(file_path)
            
#             # 生成摘要
#             text_summaries = self.summarizer.summarize_texts(texts)
#             table_summaries = self.summarizer.summarize_tables(tables)
#             image_summaries = self.summarizer.summarize_images(images)
            
#             # 加載到向量存储
#             self.rag_pipeline.load_summaries(
#                 texts, text_summaries,
#                 tables, table_summaries,
#                 images, image_summaries
#             )
            
#             logger.info("Document processing completed successfully")
            
#         except Exception as e:
#             logger.error(f"Error processing document: {str(e)}")
#             raise

#     def query(self, question: str) -> str:
#         """查詢處理過的文檔"""
#         try:
#             return self.rag_pipeline.query(question)
#         except Exception as e:
#             logger.error(f"Error during query: {str(e)}")
#             raise

# def main():
#     # 創建MultiModalRAG實例
#     rag_system = MultiModalRAG()
    
#     # 處理PDF文檔
#     pdf_path = "/Users/kevinluo/Documents/multimodal_rag_test/data/attention.pdf"
#     rag_system.process_document(pdf_path)
    
#     # 示例查詢
#     question = "What is the main topic of the document?"
#     response = rag_system.query(question)
#     print(f"Question: {question}")
#     print(f"Answer: {response}")

# if __name__ == "__main__":
#     main()
# import os
# import uuid
# import base64
# import nltk
# from typing import List
# from IPython.display import Image, display
# from unstructured.partition.pdf import partition_pdf
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from groq import Groq
# from langsmith import Client
# from langchain.chains import load_summarize_chain
# from langchain.schema import Document
# from langchain_openai import ChatOpenAI
# from langchain.vectorstores import Chroma
# from langchain.storage import InMemoryStore
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.retrievers.multi_vector import MultiVectorRetriever

from dotenv import load_dotenv

load_dotenv()

# def setup_nltk():
#     """Download required NLTK packages"""
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('averaged_perceptron_tagger_eng')
#     nltk.download('punkt_tab')
#     nltk.download('popular')

# def check_file_exists(file_path: str) -> bool:
#     """Check if the PDF file exists"""
#     if os.path.exists(file_path):
#         print(f"File exists at: {file_path}")
#         return True
#     else:
#         print(f"File not found at: {file_path}")
#         raise FileNotFoundError(f"File not found at: {file_path}")

# def process_pdf(file_path: str):
#     """Process PDF and extract chunks"""
#     chunks = partition_pdf(
#         filename=file_path,
#         infer_table_structure=True,
#         strategy="hi_res",
#         extract_image_block_types=["Image"],
#         extract_image_block_to_payload=True,
#         chunking_strategy="by_title",
#         max_characters=10000,
#         combine_text_under_n_chars=2000,
#         new_after_n_chars=6000,
#     )
#     return chunks

# def separate_content(chunks):
#     """Separate tables, texts and images from chunks"""
#     tables = []
#     texts = []
#     for chunk in chunks:
#         if "Table" in str(type(chunk)):
#             tables.append(chunk)
#         if "CompositeElement" in str(type((chunk))):
#             texts.append(chunk)
#     return tables, texts

# def get_images_base64(chunks):
#     """Extract base64 encoded images from chunks"""
#     images_b64 = []
#     for chunk in chunks:
#         if "CompositeElement" in str(type(chunk)):
#             chunk_els = chunk.metadata.orig_elements
#             for el in chunk_els:
#                 if "Image" in str(type(el)):
#                     images_b64.append(el.metadata.image_base64)
#     return images_b64

# def display_base64_image(base64_code):
#     """Display a base64 encoded image"""
#     image_data = base64.b64decode(base64_code)
#     display(Image(data=image_data))

# def setup_api_connections():
#     """Setup and test API connections"""
#     # Check GROQ API
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         print("GROQ_API_KEY not found in environment variables")
#     else:
#         print("GROQ_API_KEY found:", api_key[:6] + "..." + api_key[-4:])
    
#     try:
#         client = Groq(api_key=api_key)
#         completion = client.chat.completions.create(
#             messages=[{"role": "user", "content": "Say 'Hello, this is a test!'"}],
#             model="mixtral-8x7b-32768",
#         )
#         print("\nAPI Test Result:")
#         print("Status: Success!")
#         print("Response:", completion.choices[0].message.content)
#     except Exception as e:
#         print("\nAPI Test Result:")
#         print("Status: Failed")
#         print("Error:", str(e))

#     # Check LangSmith API
#     langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
#     if not langchain_api_key:
#         print("LANGCHAIN_API_KEY not found in environment variables")
#     else:
#         print("LANGCHAIN_API_KEY found:", langchain_api_key[:6] + "..." + langchain_api_key[-4:])

# def convert_to_documents(chunks) -> List[Document]:
#     """Convert unstructured chunks to LangChain documents"""
#     documents = []
#     for chunk in chunks:
#         if hasattr(chunk, 'text'):
#             text = chunk.text
#         else:
#             text = str(chunk)
#         doc = Document(page_content=text)
#         documents.append(doc)
#     return documents

# def process_documents(text_documents, table_documents=None):
#     """Process and summarize documents"""
#     llm = ChatGroq(
#         model_name="llama-3.1-8b-instant",
#         temperature=0,
#         max_tokens=1024
#     )
    
#     summarize_chain = load_summarize_chain(
#         llm=llm,
#         chain_type="map_reduce",
#         verbose=True
#     )

#     # Process text documents
#     text_summaries = []
#     print(f"Processing {len(text_documents)} text documents")
#     for doc in text_documents:
#         try:
#             summary = summarize_chain.invoke({"input_documents": [doc]})
#             summary_text = summary.get('output_text', '')
#             text_summaries.append(summary_text)
#             print(f"Successfully summarized document: {summary_text[:100]}...")
#         except Exception as e:
#             print(f"Error processing document: {str(e)}")
#             text_summaries.append("")

#     # Process table documents if provided
#     table_summaries = []
#     if table_documents:
#         print(f"\nProcessing {len(table_documents)} table documents")
#         for doc in table_documents:
#             try:
#                 summary = summarize_chain.invoke({"input_documents": [doc]})
#                 summary_text = summary.get('output_text', '')
#                 table_summaries.append(summary_text)
#                 print(f"Successfully summarized table: {summary_text[:100]}...")
#             except Exception as e:
#                 print(f"Error processing table: {str(e)}")
#                 table_summaries.append("")

#     return text_summaries, table_summaries

# def process_images(images):
#     """Process and describe images"""
#     prompt_template = """Describe the image in detail. For context,
#     the image is part of a research paper explaining the transformers
#     architecture. Be specific about graphs, such as bar plots."""
    
#     messages = [
#         (
#             "user",
#             [
#                 {"type": "text", "text": prompt_template},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": "data:image/jpeg;base64,{image}"},
#                 },
#             ],
#         )
#     ]
    
#     prompt = ChatPromptTemplate.from_messages(messages)
#     chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
#     return chain.batch(images)

# def setup_retriever(texts, text_summaries, images, image_summaries):
#     """Setup and configure the retriever"""
#     vectorstore = Chroma(
#         collection_name="multi_modal_rag",
#         embedding_function=OpenAIEmbeddings()
#     )
#     store = InMemoryStore()
#     id_key = "doc_id"
    
#     retriever = MultiVectorRetriever(
#         vectorstore=vectorstore,
#         docstore=store,
#         id_key=id_key,
#     )

#     # Add texts
#     doc_ids = [str(uuid.uuid4()) for _ in texts]
#     summary_texts = [
#         Document(page_content=summary, metadata={id_key: doc_ids[i]})
#         for i, summary in enumerate(text_summaries)
#     ]
#     retriever.vectorstore.add_documents(summary_texts)
#     retriever.docstore.mset(list(zip(doc_ids, texts)))

#     # Add image summaries
#     img_ids = [str(uuid.uuid4()) for _ in images]
#     summary_img = [
#         Document(page_content=summary, metadata={id_key: img_ids[i]})
#         for i, summary in enumerate(image_summaries)
#     ]
#     retriever.vectorstore.add_documents(summary_img)
#     retriever.docstore.mset(list(zip(img_ids, images)))

#     return retriever

# def main():
#     # 設置檔案路徑
#     file_path = "/Users/kevinluo/Documents/multimodal_rag_test/data/attention.pdf"
    
#     # 初始設置
#     setup_nltk()
#     check_file_exists(file_path)
#     setup_api_connections()
    
#     # 處理 PDF
#     chunks = process_pdf(file_path)
#     print(f"Successfully extracted {len(chunks)} chunks")
    
#     # 分離內容
#     tables, texts = separate_content(chunks)
#     images = get_images_base64(chunks)
    
#     # 處理文檔
#     text_documents = convert_to_documents(texts)
#     table_documents = convert_to_documents(tables) if tables else None
#     text_summaries, table_summaries = process_documents(text_documents, table_documents)
    
#     # 處理圖片
#     image_summaries = process_images(images)
    
#     # 設置檢索器
#     retriever = setup_retriever(texts, text_summaries, images, image_summaries)
    
#     # 測試檢索
#     query = "who are the authors of the paper?"
#     docs = retriever.invoke(query)
    
#     print("\nRetrieval Results:")
#     for doc in docs:
#         print(str(doc) + "\n\n" + "-" * 80)

# if __name__ == "__main__":
#     main()

import os
import uuid
import base64
import nltk
import json
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from IPython.display import Image, display
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
from langsmith import Client
from langchain.chains import load_summarize_chain
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

class PDFAnalysisCache:
    """Cache system for PDF analysis results"""
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_path(self, file_path: str, analysis_type: str) -> Path:
        """Get cache file path based on file hash and analysis type"""
        file_hash = self._get_file_hash(file_path)
        return self.cache_dir / f"{file_hash}_{analysis_type}.pkl"
    
    def get(self, file_path: str, analysis_type: str) -> Optional[Any]:
        """Get cached analysis result"""
        cache_path = self._get_cache_path(file_path, analysis_type)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save(self, file_path: str, analysis_type: str, data: Any):
        """Save analysis result to cache"""
        cache_path = self._get_cache_path(file_path, analysis_type)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

class ReportGenerator:
    """Generate and manage analysis reports"""
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.report_sections = {
            "text": [],
            "image": [],
            "table": []
        }
        self.current_report_path = None
        self._initialize_report()
    
    def _initialize_report(self):
        """Initialize a new report file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_report_path = self.output_dir / f"analysis_report_{timestamp}.md"
        self._write_header()
    
    def _write_header(self):
        """Write report header"""
        header = f"""# PDF Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        with open(self.current_report_path, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def add_section(self, section_type: str, content: str, page_number: Optional[int] = None):
        """Add content to a section with real-time file update"""
        section_header = f"\n## Information from {section_type.title()}\n"
        if page_number is not None:
            section_header += f"(Page {page_number})\n"
        
        content_with_header = f"{section_header}{content}\n"
        self.report_sections[section_type].append(content_with_header)
        
        # Real-time update to file
        with open(self.current_report_path, 'a', encoding='utf-8') as f:
            f.write(content_with_header)
        
        # Print to console
        print(f"\n{'-' * 80}\n{content_with_header}\n{'-' * 80}")

def setup_nltk():
    """Download required NLTK packages"""
    nltk_packages = ['punkt', 'averaged_perceptron_tagger', 
                    'averaged_perceptron_tagger_eng', 'punkt_tab', 'popular']
    for package in tqdm(nltk_packages, desc="Setting up NLTK"):
        nltk.download(package, quiet=True)

def check_file_exists(file_path: str) -> bool:
    """Check if the PDF file exists"""
    if os.path.exists(file_path):
        print(f"File exists at: {file_path}")
        return True
    else:
        raise FileNotFoundError(f"File not found at: {file_path}")

def process_pdf(file_path: str, cache: PDFAnalysisCache):
    """Process PDF and extract chunks with caching"""
    cached_chunks = cache.get(file_path, "pdf_chunks")
    if cached_chunks:
        print("Using cached PDF chunks")
        return cached_chunks
    
    print("Processing PDF file...")
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    
    cache.save(file_path, "pdf_chunks", chunks)
    return chunks

def extract_page_number(chunk: Any) -> Optional[int]:
    """Safely extract page number from chunk metadata"""
    try:
        if hasattr(chunk, 'metadata'):
            metadata = chunk.metadata
            if hasattr(metadata, 'page_number'):
                return metadata.page_number
    except AttributeError:
        pass
    return None

def process_text_chunk(chunk: Any, llm: Any, summarize_chain: Any, 
                      report_generator: ReportGenerator) -> str:
    """Process a single text chunk and update report"""
    try:
        # Safely extract page number
        page_num = extract_page_number(chunk)
        
        # Safely get text content
        text_content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        # Create document and get summary
        doc = Document(page_content=text_content)
        summary = summarize_chain.invoke({"input_documents": [doc]})
        summary_text = summary.get('output_text', '')
        
        # Add to report
        report_generator.add_section("text", summary_text, page_num)
        return summary_text
    
    except Exception as e:
        print(f"Error processing text chunk: {str(e)}")
        return ""

def process_table_chunk(table: Any, llm: Any, summarize_chain: Any, 
                       report_generator: ReportGenerator) -> str:
    """Process a single table and update report"""
    try:
        # Safely extract page number
        page_num = extract_page_number(table)
        
        # Get table content
        table_content = table.metadata.text_as_html if hasattr(table.metadata, 'text_as_html') else str(table)
        
        # Create document and get summary
        doc = Document(page_content=table_content)
        summary = summarize_chain.invoke({"input_documents": [doc]})
        summary_text = summary.get('output_text', '')
        
        # Add to report
        report_generator.add_section("table", summary_text, page_num)
        return summary_text
    
    except Exception as e:
        print(f"Error processing table: {str(e)}")
        return ""

def process_image_chunk(image: str, chain: Any, report_generator: ReportGenerator, 
                       page_num: Optional[int] = None) -> str:
    """Process a single image and update report"""
    try:
        description = chain.invoke(image)
        report_generator.add_section("image", description, page_num)
        return description
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return ""

def setup_image_chain() -> Any:
    """Setup the image processing chain"""
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
    
    return (ChatPromptTemplate.from_messages(messages) | 
            ChatOpenAI(model="gpt-4o-mini") | 
            StrOutputParser())

def main():
    # Initialize cache and report generator
    cache = PDFAnalysisCache()
    report_generator = ReportGenerator()
    
    # Set file path
    file_path = "/Users/kevinluo/Documents/multimodal_rag_test/data/attention.pdf"
    
    # Setup and initialization
    print("Initializing...")
    setup_nltk()
    check_file_exists(file_path)
    
    # Process PDF
    chunks = process_pdf(file_path, cache)
    print(f"\nExtracted {len(chunks)} chunks from PDF")
    
    # Initialize models
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024
    )
    
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=False
    )
    
    image_chain = setup_image_chain()
    
    # Extract and process images first
    print("\nProcessing images...")
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
    
    print(f"Found {len(images)} images")
    
    # Process images with progress bar
    for i, img in enumerate(tqdm(images, desc="Processing images")):
        try:
            process_image_chunk(img, image_chain, report_generator, f"Image {i+1}")
        except Exception as e:
            print(f"Error processing image {i+1}: {str(e)}")
    
    # Process other chunks with progress bars
    for chunk in tqdm(chunks, desc="Processing text and tables"):
        chunk_type = type(chunk).__name__
        
        if "Table" in chunk_type:
            process_table_chunk(chunk, llm, summarize_chain, report_generator)
        elif "CompositeElement" in str(type(chunk)) or "Text" in chunk_type:
            process_text_chunk(chunk, llm, summarize_chain, report_generator)

    print(f"\nAnalysis complete! Report saved to: {report_generator.current_report_path}")

if __name__ == "__main__":
    main()