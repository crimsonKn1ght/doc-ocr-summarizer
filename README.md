<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=📑%20PDF%20Q%20'n'%20A%20Assistant&fontSize=60&fontAlign=50" alt="PDF Q&A Assistant">
</p>

<p align="center">
  <a href="https://github.com/crimsonKn1ght/doc-ocr-summarizer/stargazers">
    <img src="https://img.shields.io/github/stars/crimsonKn1ght/doc-ocr-summarizer?style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/crimsonKn1ght/doc-ocr-summarizer/network/members">
    <img src="https://img.shields.io/github/forks/crimsonKn1ght/doc-ocr-summarizer?style=for-the-badge" alt="GitHub forks">
  </a>
  <a href="https://github.com/crimsonKn1ght/doc-ocr-summarizer/graphs/commit-activity">
    <img src="https://img.shields.io/maintenance/yes/2025?style=for-the-badge" alt="Maintained">
  </a>
  <a href="https://github.com/crimsonKn1ght/doc-ocr-summarizer">
    <img src="https://img.shields.io/github/languages/top/crimsonKn1ght/doc-ocr-summarizer?style=for-the-badge" alt="Language">
  </a>
</p>

---

## Overview

**📑 Multimodal PDF Q&A Assistant** is a Python application that allows you to **upload PDFs (scanned or digital)** and ask questions about their content. The system uses:

- **Text extraction via PyMuPDF**  
- **OCR fallback with Tesseract** for scanned PDFs  
- **Local LLaMA 3.1 embeddings and LLM** via Ollama  
- **FAISS vector search** for retrieval  
- **Gradio** as an interactive web interface  

This tool runs **entirely offline**, leveraging your local LLaMA 3.1 model for embedding and question-answering.

---

## Features

- Extract text from standard PDFs or scanned documents.
- Perform OCR if text extraction fails.
- Chunk text into embeddings for efficient semantic search.
- Local LLaMA 3.1 for embeddings and answers (no external API needed).
- Interactive Gradio UI for uploading PDFs and querying content.

---

## Demo

```bash
# Run the Gradio app
python pdf_detector.py
```
- Upload your PDF file.
- Type a question.
- Receive an answer from the document.

## Installation
1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

3. Ensure Tesseract OCR is installed and added to your PATH:
- Windows: https://github.com/tesseract-ocr/tesseract
- Linux/macOS: Use your package manager (e.g., `sudo apt install tesseract-ocr`)

## Usage
1. Start the Gradio interface:
```bash
python pdf_detector.py
```

2. Upload a PDF and ask questions in the textbox.

3. Receive answers from the document based on semantic search.

## Requirements

- Python 3.11+
- PyMuPDF
- pytesseract
- Pillow
- Gradio
- torch
- numpy
- langchain
- langchain-ollama
- faiss-cpu

(See `requirements.txt` for exact versions.)

## Project Structure

```bash
pdf-qna-assistant/
├── pdf_detector.py         # Main app
├── requirements.txt
├── README.md
└── tmp_text.txt            # Temporary text file (auto-generated)
```

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=footer&animation=twinkling" width="100%"/>
</p>
