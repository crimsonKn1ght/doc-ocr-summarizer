<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=📑%20PDF%20Q&A%20Assistant&fontSize=60&fontAlign=50" alt="PDF Q&A Assistant">
</p>

<p align="center">
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/stargazers">
    <img src="https://img.shields.io/github/stars/YOUR_USERNAME/YOUR_REPO?style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/network/members">
    <img src="https://img.shields.io/github/forks/YOUR_USERNAME/YOUR_REPO?style=for-the-badge" alt="GitHub forks">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/graphs/commit-activity">
    <img src="https://img.shields.io/maintenance/yes/2025?style=for-the-badge" alt="Maintained">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO">
    <img src="https://img.shields.io/github/languages/top/YOUR_USERNAME/YOUR_REPO?style=for-the-badge" alt="Language">
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
