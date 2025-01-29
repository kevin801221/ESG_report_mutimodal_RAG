from typing import List, Dict, Optional, Any
import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from ..config.settings import PDF_PROCESSING
from ..utils.logger import get_logger  # 我們稍後會實現這個

logger = get_logger(__name__)

class PDFProcessor:
    """PDF 文件處理器，負責提取和處理 PDF 中的文本、圖片和表格"""
    
    def __init__(
        self,
        chunk_size: int = PDF_PROCESSING["CHUNK_SIZE"],
        chunk_overlap: int = PDF_PROCESSING["CHUNK_OVERLAP"],
        strategy: str = PDF_PROCESSING["STRATEGY"],
        chunking_strategy: str = PDF_PROCESSING["CHUNKING_STRATEGY"]
    ):
        """
        初始化 PDF 處理器
        
        Args:
            chunk_size: 文本分塊大小
            chunk_overlap: 分塊重疊大小
            strategy: PDF 處理策略 ("fast" 或 "accurate")
            chunking_strategy: 分塊策略 ("basic" 或 "advanced")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.chunking_strategy = chunking_strategy
        
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        處理單個 PDF 文件
        
        Args:
            file_path: PDF 文件路徑
            
        Returns:
            Dict[str, Any]: 包含處理結果的字典，
                           包括文本塊、圖片和表格
                           
        Raises:
            FileNotFoundError: 如果文件不存在
            Exception: 處理過程中的其他錯誤
        """
        logger.info(f"開始處理 PDF 文件: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            # 提取所有元素
            elements = partition_pdf(
                filename=file_path,
                strategy=self.strategy,
                chunking_strategy=self.chunking_strategy,
                extract_images_in_pdf=True,
            )
            
            # 解析結果
            result = self._parse_elements(elements)
            logger.info(f"PDF 處理完成: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"PDF 處理錯誤: {str(e)}")
            raise
    
    def process_directory(self, directory_path: str) -> Dict[str, Dict[str, Any]]:
        """
        處理目錄中的所有 PDF 文件
        
        Args:
            directory_path: PDF 文件目錄路徑
            
        Returns:
            Dict[str, Dict[str, Any]]: 文件名到處理結果的映射
        """
        results = {}
        directory = Path(directory_path)
        
        logger.info(f"開始處理目錄: {directory_path}")
        
        for pdf_file in directory.glob("*.pdf"):
            try:
                results[pdf_file.name] = self.process_file(str(pdf_file))
            except Exception as e:
                logger.error(f"處理文件 {pdf_file.name} 時發生錯誤: {str(e)}")
                results[pdf_file.name] = {"error": str(e)}
        
        return results
    
    def _parse_elements(self, elements: List[Any]) -> Dict[str, Any]:
        """
        解析提取的元素
        
        Args:
            elements: 從 PDF 提取的元素列表
            
        Returns:
            Dict[str, Any]: 包含分類後元素的字典
        """
        texts = []
        images = []
        tables = []
        
        for element in elements:
            element_type = str(type(element))
            
            if "Text" in element_type:
                texts.append(element)
            elif "Image" in element_type:
                # 提取圖片的 base64 編碼
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    images.append(element.metadata.image_base64)
            elif "Table" in element_type:
                if hasattr(element, "metadata") and hasattr(element.metadata, "text_as_html"):
                    tables.append(element.metadata.text_as_html)
                else:
                    tables.append(str(element))
        
        return {
            "texts": texts,
            "images": images,
            "tables": tables,
            "metadata": {
                "total_elements": len(elements),
                "text_count": len(texts),
                "image_count": len(images),
                "table_count": len(tables)
            }
        }
    
    @staticmethod
    def get_images_base64(chunks: List[Any]) -> List[str]:
        """
        從 chunks 中提取所有圖片的 base64 編碼
        
        Args:
            chunks: PDF chunks 列表
            
        Returns:
            List[str]: base64 編碼的圖片列表
        """
        images_b64 = []
        
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)) and hasattr(el.metadata, "image_base64"):
                        images_b64.append(el.metadata.image_base64)
        
        return images_b64