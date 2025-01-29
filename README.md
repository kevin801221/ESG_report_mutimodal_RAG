# 多模態 RAG 系統

這是一個用於處理永續報告書的多模態 RAG (Retrieval-Augmented Generation) 系統。

一個強大的多模態檢索增強生成(RAG)系統，支持PDF文檔處理、文本分析和圖像理解。
# 功能特點

📄 完整的PDF處理

智能文檔分割與內容提取
自動圖片識別與提取
表格識別與結構化數據提取
文檔結構分析與元數據提取


📝 進階文本處理

智能文本清理與標準化
精確的分句和分詞
關鍵詞提取與主題分析
靈活的文本分塊策略


🖼️ 圖像處理能力

多格式圖像轉換
圖像優化與壓縮
圖像特徵提取
Base64編碼支持


🤖 AI模型整合

Groq模型支持（文本處理）
OpenAI多模態分析
批量處理優化
自適應模型選擇

## 功能
- PDF 文件處理
- 文本摘要生成
- 圖像處理和分析
- 多模態檢索系統

## 基本安裝

1. 安裝系統依賴：
   ```bash
   cd setup
   chmod +x install_deps.sh
   ./install_deps.sh
   ```

2. 設置 NLTK 數據：
   ```bash
   python setup/setup_nltk.py
   ```

3. 配置環境變數：
   - 複製 `.env.example` 到 `.env`
   - 填入必要的 API keys

# MultiModal RAG System

一個強大的多模態檢索增強生成(RAG)系統，支持PDF文檔處理、文本分析和圖像理解。

## 功能特點

- 📄 **完整的PDF處理**
  - 智能文檔分割與內容提取
  - 自動圖片識別與提取
  - 表格識別與結構化數據提取
  - 文檔結構分析與元數據提取

- 📝 **進階文本處理**
  - 智能文本清理與標準化
  - 精確的分句和分詞
  - 關鍵詞提取與主題分析
  - 靈活的文本分塊策略

- 🖼️ **圖像處理能力**
  - 多格式圖像轉換
  - 圖像優化與壓縮
  - 圖像特徵提取
  - Base64編碼支持

- 🤖 **AI模型整合**
  - Groq模型支持（文本處理）
  - OpenAI多模態分析
  - 批量處理優化
  - 自適應模型選擇

## 系統要求

- Python 3.10+
- 系統依賴：
  - Poppler (PDF處理)
  - Tesseract (OCR功能)
  - libmagic (文件類型檢測)

## 快速開始

1. **克隆儲存庫**
```bash
git clone https://github.com/kevin801221/ESG_report_mutimodal_RAG.git
cd multimodal-rag
```

2. **環境配置**
```bash
# 創建並激活虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安裝依賴
pip install --upgrade pip
pip install -r requirements.txt
```

3. **系統依賴安裝**

MacOS:
```bash
brew install poppler tesseract libmagic
```

Linux:
```bash
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng libmagic-dev
```

4. **環境變數配置**

複製環境變數範例文件：
```bash
cp .env.example .env
```

編輯 `.env` 文件，填入必要的API密鑰：
```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
OPENAI_API_KEY=your_openai_api_key
```

5. **初始化NLTK數據**
```bash
python setup/nltk_setup.py
```

## 基本使用

```python
from multimodal_rag import MultiModalRAGSystem

# 初始化系統
system = MultiModalRAGSystem()

# 處理PDF文檔
result = system.process_document("data/raw/sample.pdf")

# 執行查詢
response = system.query_document("文檔的主要主題是什麼？")

# 生成分析報告
report = system.generate_report(response)
```

## 項目結構

```
multimodal_rag/
├── multimodal_rag/          # 核心代碼
│   ├── config/             # 配置管理
│   ├── models/            # AI模型封裝
│   ├── processors/        # 文檔處理器
│   ├── rag/              # RAG實現
│   ├── utils/            # 工具函數
│   └── main.py           # 主程序
├── data/                  # 數據目錄
├── logs/                 # 日誌文件
├── setup/               # 安裝腳本
└── tests/               # 測試代碼
```

## 開發指南

1. **安裝開發依賴**
```bash
pip install -e ".[dev]"
```

2. **運行測試**
```bash
pytest tests/
```

3. **代碼風格檢查**
```bash
flake8 multimodal_rag
black multimodal_rag
```

## API文檔

詳細的API文檔可在 [docs/](docs/) 目錄中找到。主要組件包括：

- `PDFProcessor`: PDF文檔處理
- `TextProcessor`: 文本分析處理
- `ImageProcessor`: 圖像處理
- `GroqModel`: Groq模型封裝
- `OpenAIModel`: OpenAI模型封裝

## 貢獻指南

1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟合併請求

## 許可證

本項目採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件

## 聯繫方式

- 作者：[KevinLuo]
- Email：[kilong31442@gmail.com]
- GitHub：[kevin801221](https://github.com/kevin801221)

## 致謝

- [Groq](https://groq.com/) - 提供高效的LLM服務
- [OpenAI](https://openai.com/) - 提供多模態AI能力
- [LangChain](https://langchain.com/) - 提供RAG框架支持