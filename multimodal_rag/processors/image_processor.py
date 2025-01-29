import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """圖像處理器，負責圖像的預處理、轉換和優化"""
    
    def __init__(self, output_dir: Optional[str] = "data/processed/images"):
        """
        初始化圖像處理器
        
        Args:
            output_dir: 處理後圖像的輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def image_to_base64(self, image_path: str) -> str:
        """
        將圖像轉換為 base64 編碼
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            base64 編碼的圖像字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"轉換圖像到 base64 時發生錯誤: {str(e)}")
            return ""
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """
        將 base64 編碼轉換回圖像
        
        Args:
            base64_string: base64 編碼的圖像字符串
            
        Returns:
            numpy 數組格式的圖像
        """
        try:
            # 解碼 base64 字符串
            img_data = base64.b64decode(base64_string)
            # 轉換為 numpy 數組
            nparr = np.frombuffer(img_data, np.uint8)
            # 解碼為圖像
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"從 base64 轉換圖像時發生錯誤: {str(e)}")
            return None
    
    def optimize_image(
        self,
        image: np.ndarray,
        target_size: tuple = (800, 800),
        quality: int = 85
    ) -> str:
        """
        優化圖像質量和大小
        
        Args:
            image: 輸入圖像
            target_size: 目標尺寸
            quality: JPEG 質量（0-100）
            
        Returns:
            優化後的 base64 編碼圖像
        """
        try:
            # 轉換為 PIL 圖像
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 調整大小
            img_pil.thumbnail(target_size, Image.LANCZOS)
            
            # 保存為 JPEG 格式的字節流
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=quality)
            
            # 轉換為 base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"優化圖像時發生錯誤: {str(e)}")
            return None
    
    def batch_process_images(
        self,
        image_paths: List[str],
        optimize: bool = True
    ) -> List[str]:
        """
        批量處理圖像
        
        Args:
            image_paths: 圖像文件路徑列表
            optimize: 是否優化圖像
            
        Returns:
            處理後的 base64 編碼圖像列表
        """
        processed_images = []
        
        for path in image_paths:
            try:
                # 讀取圖像
                image = cv2.imread(path)
                if image is None:
                    print(f"無法讀取圖像: {path}")
                    continue
                
                # 根據需要優化圖像
                if optimize:
                    base64_img = self.optimize_image(image)
                else:
                    base64_img = self.image_to_base64(path)
                
                if base64_img:
                    processed_images.append(base64_img)
            except Exception as e:
                print(f"處理圖像 {path} 時發生錯誤: {str(e)}")
                continue
        
        return processed_images
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        提取圖像特徵
        
        Args:
            image: 輸入圖像
            
        Returns:
            圖像特徵字典
        """
        try:
            # 計算基本特徵
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            
            # 計算平均顏色
            avg_color = np.mean(image, axis=(0, 1))
            
            # 計算主要顏色（使用 K-means）
            pixels = image.reshape(-1, channels)
            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            return {
                "dimensions": (width, height),
                "channels": channels,
                "average_color": avg_color.tolist(),
                "dominant_colors": centers.tolist(),
                "size_kb": image.nbytes / 1024
            }
        except Exception as e:
            print(f"提取圖像特徵時發生錯誤: {str(e)}")
            return {}
    
    def detect_image_type(self, image: np.ndarray) -> str:
        """
        檢測圖像類型（如：自然圖像、圖表、截圖等）
        
        Args:
            image: 輸入圖像
            
        Returns:
            圖像類型描述
        """
        try:
            # 計算顏色直方圖
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # 計算邊緣
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.count_nonzero(edges) / edges.size
            
            # 基於特徵進行簡單分類
            if edge_ratio > 0.1:  # 高邊緣比例
                if np.std(hist) < 0.01:  # 顏色分佈均勻
                    return "chart_or_diagram"
                else:
                    return "natural_image"
            else:
                if np.std(hist) < 0.005:  # 非常均勻的顏色分佈
                    return "screenshot"
                else:
                    return "document_scan"
                    
        except Exception as e:
            print(f"檢測圖像類型時發生錯誤: {str(e)}")
            return "unknown"