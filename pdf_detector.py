import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import gradio as gr

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# ---- Extract text normally ----
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text if text.strip() else None


# ---- OCR fallback if PDF is scanned ----
def ocr_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text


# ---- Build LangChain QA system ----
def build_qa_chain(pdf_path):
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        text = ocr_pdf(pdf_path)

    if not text or not text.strip():
        raise ValueError("No text could be extracted from the PDF!")

    # Wrap in a LangChain Document
    doc = Document(page_content=text, metadata={"source": pdf_path})
    docs = [doc]

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")

    # Embeddings + vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Retrieval QA chain
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa


# ---- Gradio pipeline ----
def process(pdf, question):
    try:
        qa = build_qa_chain(pdf.name)
        result = qa.run(question)
    except Exception as e:
        result = f"Error: {e}"
    return result


# ---- Gradio interface ----
demo = gr.Interface(
    fn=process,
    inputs=[gr.File(type="filepath", label="Upload PDF"),
            gr.Textbox(label="Ask a Question")],
    outputs="text",
    title="📑 Multimodal PDF Q&A Assistant",
    description="Upload a PDF (scanned or digital) and ask questions. Uses Hugging Face embeddings + Groq Llama 3.3 70B."
)

if __name__ == "__main__":
    demo.launch()
