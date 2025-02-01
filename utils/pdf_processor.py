# utils/pdf_processor.py
from unstructured.partition.pdf import partition_pdf
from typing import Tuple, List, Any

class PDFProcessor:
    def __init__(self):
        pass

    def process_pdf(self, file_path: str) -> Tuple[List[Any], List[Any], List[Any]]:
        """提取PDF中的文本、表格和圖像"""
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
        
        # 分離不同類型的內容
        texts = []
        tables = []
        images = []
        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                
        # 提取圖像
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images.append(el.metadata.image_base64)
        
        return texts, tables, images