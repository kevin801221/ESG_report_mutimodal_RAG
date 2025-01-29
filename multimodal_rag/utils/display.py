import base64
from typing import List, Dict, Any, Optional
from IPython.display import Image, display, HTML
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime

class DisplayManager:
    """顯示管理器，負責格式化和展示各種類型的內容"""
    
    def __init__(self, use_rich: bool = True):
        """
        初始化顯示管理器
        
        Args:
            use_rich: 是否使用 rich 庫進行格式化輸出
        """
        self.console = Console() if use_rich else None
    
    def display_image(self, image_data: str, width: int = 800):
        """
        顯示 base64 編碼的圖片
        
        Args:
            image_data: base64 編碼的圖片數據
            width: 顯示寬度
        """
        try:
            # 確保移除 base64 header 如果存在
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
                
            # 解碼並顯示
            image_binary = base64.b64decode(image_data)
            display(Image(data=image_binary, width=width))
        except Exception as e:
            print(f"顯示圖片時發生錯誤: {str(e)}")
    
    def display_text_analysis(self, analysis: Dict[str, Any]):
        """
        顯示文本分析結果
        
        Args:
            analysis: 文本分析結果字典
        """
        if self.console:
            table = Table(show_header=True, header_style="bold magenta", 
                        box=box.ROUNDED)
            table.add_column("指標", style="dim")
            table.add_column("值")
            
            for key, value in analysis.items():
                table.add_row(
                    key.replace('_', ' ').title(),
                    str(value)
                )
            
            self.console.print(table)
        else:
            for key, value in analysis.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    def display_retrieval_results(
        self,
        results: Dict[str, List],
        max_text_length: int = 200
    ):
        """
        顯示檢索結果
        
        Args:
            results: 檢索結果字典
            max_text_length: 文本顯示的最大長度
        """
        if self.console:
            self.console.print("[bold blue]檢索結果摘要[/bold blue]")
            
            # 顯示文本結果
            if results.get("texts"):
                self.console.print("\n[bold green]文本結果:[/bold green]")
                for i, text in enumerate(results["texts"], 1):
                    content = text[:max_text_length] + "..." if len(text) > max_text_length else text
                    self.console.print(f"{i}. {content}\n")
            
            # 顯示圖片結果
            if results.get("images"):
                self.console.print("\n[bold green]圖片結果:[/bold green]")
                self.console.print(f"找到 {len(results['images'])} 張相關圖片")
                
            # 顯示表格結果
            if results.get("tables"):
                self.console.print("\n[bold green]表格結果:[/bold green]")
                self.console.print(f"找到 {len(results['tables'])} 個相關表格")
        else:
            print("\n--- 檢索結果摘要 ---")
            
            if results.get("texts"):
                print("\n文本結果:")
                for i, text in enumerate(results["texts"], 1):
                    content = text[:max_text_length] + "..." if len(text) > max_text_length else text
                    print(f"{i}. {content}\n")
            
            if results.get("images"):
                print(f"\n圖片結果: 找到 {len(results['images'])} 張相關圖片")
            
            if results.get("tables"):
                print(f"\n表格結果: 找到 {len(results['tables'])} 個相關表格")
    
    def create_html_report(
        self,
        title: str,
        content: Dict[str, Any],
        include_timestamp: bool = True
    ) -> str:
        """
        創建 HTML 格式的報告
        
        Args:
            title: 報告標題
            content: 報告內容
            include_timestamp: 是否包含時間戳
            
        Returns:
            HTML 格式的報告字符串
        """
        html_content = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-bottom: 20px;
                }}
                .section {{
                    margin: 20px 0;
                    padding: 15px;
                    background: #f9f9f9;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_content += f'<div class="timestamp">生成時間: {timestamp}</div>'
        
        for section_title, section_content in content.items():
            html_content += f"""
            <div class="section">
                <h2>{section_title}</h2>
                <div>{section_content}</div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def format_table_data(
        self,
        data: List[Dict[str, Any]],
        headers: Optional[List[str]] = None
    ) -> str:
        """
        格式化表格數據
        
        Args:
            data: 表格數據列表
            headers: 表格標頭列表
            
        Returns:
            HTML 格式的表格字符串
        """
        if not data:
            return ""
            
        if not headers:
            headers = list(data[0].keys())
        
        table_html = """
        <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f5f6fa;">
        """
        
        # 添加表頭
        for header in headers:
            table_html += f"""
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">
                    {header}
                </th>"""
        
        table_html += """
                </tr>
            </thead>
            <tbody>
        """
        
        # 添加數據行
        for row in data:
            table_html += '<tr style="border: 1px solid #ddd;">'
            for header in headers:
                value = row.get(header, "")
                table_html += f"""
                    <td style="padding: 12px; border: 1px solid #ddd;">
                        {value}
                    </td>"""
            table_html += "</tr>"
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def show_progress(self, current: int, total: int, prefix: str = ""):
        """
        顯示進度條
        
        Args:
            current: 當前進度
            total: 總數
            prefix: 進度條前綴文字
        """
        if self.console:
            percentage = (current / total) * 100
            self.console.print(f"{prefix} [{percentage:.1f}%]")
        else:
            print(f"{prefix} {current}/{total} ({(current/total)*100:.1f}%)")