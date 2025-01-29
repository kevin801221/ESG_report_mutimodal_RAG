#!/bin/bash

# 檢測作業系統
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing dependencies for macOS..."
    brew install poppler tesseract libmagic
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Installing dependencies for Linux..."
    apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng libmagic-dev
else
    echo "Unsupported operating system"
    exit 1
fi

# 安裝 Python 依賴
pip install -r ../requirements.txt
