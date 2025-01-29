from typing import List, Dict, Any, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """文本處理器，負責文本的預處理、分割和分析"""
    
    def __init__(self, language: str = "english"):
        """
        初始化文本處理器
        
        Args:
            language: 文本語言（用於停用詞和分詞）
        """
        self.language = language
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # 確保必要的 NLTK 數據已下載
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def clean_text(self, text: str) -> str:
        """
        清理和標準化文本
        
        Args:
            text: 輸入文本
            
        Returns:
            清理後的文本
        """
        if not text:
            return ""
            
        # 移除多餘的空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 統一換行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 移除控制字符
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text
    
    def split_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        將文本分割成小段
        
        Args:
            text: 輸入文本
            chunk_size: 每段的最大字符數
            chunk_overlap: 段落間的重疊字符數
            
        Returns:
            文本段落列表
        """
        if chunk_size and chunk_overlap:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        else:
            splitter = self.text_splitter
            
        return splitter.split_text(text)
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """
        提取關鍵短語
        
        Args:
            text: 輸入文本
            top_k: 返回的關鍵短語數量
            
        Returns:
            關鍵短語列表
        """
        # 分詞
        words = word_tokenize(text.lower())
        
        # 移除停用詞
        stop_words = set(stopwords.words(self.language))
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # 計算詞頻
        freq_dist = nltk.FreqDist(words)
        
        # 返回最常見的短語
        return [word for word, _ in freq_dist.most_common(top_k)]
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        分析文本結構
        
        Args:
            text: 輸入文本
            
        Returns:
            文本結構分析結果
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        提取文本中的章節
        
        Args:
            text: 輸入文本
            
        Returns:
            章節標題和內容的映射
        """
        # 使用正則表達式匹配可能的章節標題
        section_pattern = r'^(?:Chapter|Section|\d+\.)\s+[A-Z].*$'
        
        sections = {}
        current_section = "未分類"
        current_content = []
        
        for line in text.split('\n'):
            if re.match(section_pattern, line.strip()):
                # 保存前一個章節
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最後一個章節
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def find_entities(self, text: str) -> Dict[str, List[str]]:
        """
        查找文本中的實體（人名、地名、組織等）
        
        Args:
            text: 輸入文本
            
        Returns:
            分類的實體列表
        """
        try:
            # 嘗試使用 NLTK 的 NER
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
            
            # 分詞和標註
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            entities = nltk.chunk.ne_chunk(tagged)
            
            # 解析實體
            extracted_entities = {
                "PERSON": [],
                "ORGANIZATION": [],
                "GPE": [],  # Geo-Political Entity
                "OTHER": []
            }
            
            for entity in entities:
                if hasattr(entity, 'label'):
                    entity_text = ' '.join([child[0] for child in entity])
                    label = entity.label()
                    if label in extracted_entities:
                        extracted_entities[label].append(entity_text)
        except Exception as e:
            print(f"查找實體時發生錯誤: {str(e)}")
            return {}
        
        return extracted_entities