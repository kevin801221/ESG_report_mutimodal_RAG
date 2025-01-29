import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# from config.settings import verify_api_keys
# from utils.api_checker import APIChecker
# from utils.display import DisplayManager
# from processors.pdf_processor import PDFProcessor
# from processors.image_processor import ImageProcessor
# from processors.text_processor import TextProcessor
# from models.groq_model import GroqModel
# from models.openai_model import OpenAIModel
# from rag.vectorstore import VectorStoreManager
# from rag.retriever import MultiModalRetriever
from multimodal_rag.utils.api_checker import APIChecker
from multimodal_rag.config.settings import verify_api_keys
from multimodal_rag.utils.display import DisplayManager
from multimodal_rag.processors.pdf_processor import PDFProcessor
from multimodal_rag.processors.image_processor import ImageProcessor
from multimodal_rag.processors.text_processor import TextProcessor
from multimodal_rag.models.groq_model import GroqModel
from multimodal_rag.models.openai_model import OpenAIModel
from multimodal_rag.rag.vectorstore import VectorStoreManager
from multimodal_rag.rag.retriever import MultiModalRetriever

class MultiModalRAGSystem:
    """多模態 RAG 系統的主要類"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 RAG 系統
        
        Args:
            config_path: 配置文件路徑
        """
        # 載入環境變數
        load_dotenv(config_path)
        
        # 驗證 API keys
        if not verify_api_keys():
            raise ValueError("API keys 未正確配置")
            
        # 初始化顯示管理器
        self.display = DisplayManager(use_rich=True)
        
        # 初始化處理器
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        
        # 初始化模型
        self.groq_model = GroqModel()
        self.openai_model = OpenAIModel()
        
        # 初始化向量存儲和檢索器
        self.vectorstore = VectorStoreManager()
        self.retriever = MultiModalRetriever(
            self.vectorstore,
            self.openai_model,
            self.groq_model
        )
        
        # 創建必要的目錄
        self._create_directories()
        
    def _create_directories(self):
        """創建必要的目錄結構"""
        directories = [
            "data/raw",
            "data/processed",
            "data/processed/images",
            "data/processed/texts",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        處理單個文檔
        
        Args:
            file_path: 文檔路徑
            
        Returns:
            處理結果
        """
        self.display.show_progress(0, 5, "開始處理文檔")
        
        try:
            # 1. 提取 PDF 內容
            chunks = self.pdf_processor.process_file(file_path)
            self.display.show_progress(1, 5, "PDF 內容提取完成")
            
            # 2. 處理文本內容
            texts = []
            for chunk in chunks:
                if hasattr(chunk, 'text'):
                    cleaned_text = self.text_processor.clean_text(chunk.text)
                    texts.append(cleaned_text)
            self.display.show_progress(2, 5, "文本處理完成")
            
            # 3. 提取和處理圖片
            images = self.pdf_processor.get_images_base64(chunks)
            processed_images = []
            for image in images:
                img = self.image_processor.base64_to_image(image)
                if img is not None:
                    optimized = self.image_processor.optimize_image(img)
                    if optimized:
                        processed_images.append(optimized)
            self.display.show_progress(3, 5, "圖片處理完成")
            
            # 4. 生成摘要
            text_summaries = self.groq_model.batch_process_texts(texts)
            image_summaries = self.openai_model.batch_analyze_images(processed_images)
            self.display.show_progress(4, 5, "摘要生成完成")
            
            # 5. 添加到向量存儲
            text_ids = self.vectorstore.add_texts(
                texts=texts,
                summaries=text_summaries
            )
            
            image_ids = self.vectorstore.add_images(
                images=processed_images,
                summaries=image_summaries
            )
            
            self.display.show_progress(5, 5, "向量存儲完成")
            
            return {
                "text_ids": text_ids,
                "image_ids": image_ids,
                "text_count": len(texts),
                "image_count": len(processed_images)
            }
            
        except Exception as e:
            self.display.console.print(f"[red]處理文檔時發生錯誤: {str(e)}[/red]")
            return {"error": str(e)}
    
    def query_document(self, query: str, filter_metadata: Optional[dict] = None) -> Dict[str, Any]:
        """
        查詢文檔
        
        Args:
            query: 查詢文本
            filter_metadata: 過濾條件
            
        Returns:
            查詢結果
        """
        try:
            # 執行檢索
            response, docs = self.retriever.retrieve_and_process(
                query=query,
                filter_metadata=filter_metadata
            )
            
            # 分析檢索質量
            quality_analysis = self.retriever.analyze_retrieval_quality(
                query=query,
                retrieved_docs=docs
            )
            
            # 顯示結果
            self.display.display_retrieval_results({
                "texts": [doc.page_content for doc in docs if doc.metadata.get("type") in (None, "text")],
                "images": [doc.page_content for doc in docs if doc.metadata.get("type") == "image"],
                "tables": [doc.page_content for doc in docs if doc.metadata.get("type") == "table"]
            })
            
            return {
                "response": response,
                "quality_analysis": quality_analysis,
                "raw_docs": docs
            }
            
        except Exception as e:
            self.display.console.print(f"[red]查詢時發生錯誤: {str(e)}[/red]")
            return {"error": str(e)}
    
    def generate_report(self, results: Dict[str, Any], title: str = "查詢報告") -> str:
        """
        生成報告
        
        Args:
            results: 查詢結果
            title: 報告標題
            
        Returns:
            HTML 格式的報告
        """
        report_content = {
            "查詢回應": results.get("response", "無回應"),
            "質量分析": results.get("quality_analysis", "無分析結果"),
            "檢索文檔數": len(results.get("raw_docs", []))
        }
        
        return self.display.create_html_report(title, report_content)

def main():
    """主函數"""
    try:
        # 檢查 API 連接
        api_checker = APIChecker()
        api_status = api_checker.check_all_apis()
        
        # 顯示每個 API 的具體狀態
        for api_name, status in api_status.items():
            print(f"{api_name} API 狀態: {'成功' if status['status'] else '失敗'}")
            if not status['status']:
                print(f"錯誤信息: {status['message']}")
                
        if not all(status["status"] for status in api_status.values()):
            raise RuntimeError("部分 API 連接失敗，請檢查設置")
        
        # 初始化系統
        system = MultiModalRAGSystem()
        
        # 示例用法
        # 1. 處理文檔
        result = system.process_document("data/raw/sample.pdf")
        
        if "error" not in result:
            print(f"成功處理文檔，提取了 {result['text_count']} 個文本片段和 {result['image_count']} 張圖片")
            
            # 2. 執行查詢
            query = "文檔的主要主題是什麼？"
            query_result = system.query_document(query)
            
            if "error" not in query_result:
                # 3. 生成報告
                report_html = system.generate_report(query_result)
                
                # 保存報告
                with open("data/processed/report.html", "w", encoding="utf-8") as f:
                    f.write(report_html)
                    
                print("報告已生成並保存到 data/processed/report.html")
            
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()