import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st
import tempfile
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

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
@st.cache_resource
def build_qa_chain(_pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(_pdf_file.getvalue())
        pdf_path = tmpfile.name

    try:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            text = ocr_pdf(pdf_path)

        if not text or not text.strip():
            raise ValueError("No text could be extracted from the PDF!")

        doc = Document(page_content=text, metadata={"source": pdf_path})
        docs = [doc]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("Text splitting produced no chunks.")

        # Use Hugging Face for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)

        # Use Groq for the LLM
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return qa
    finally:
        os.remove(pdf_path)


# ---- Streamlit interface ----
st.set_page_config(page_title="Multimodal PDF Q&A Assistant", page_icon="📑")

st.title("📑 Multimodal PDF Q&A Assistant")
st.markdown("Upload a PDF (scanned or digital) and ask questions. This app is powered by Llama 3.3 70B via Groq.")

# Securely get the Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Ask a question about the PDF content")

if uploaded_file is not None and question:
    with st.spinner("Processing PDF and getting your answer..."):
        try:
            qa_chain = build_qa_chain(uploaded_file)
            result = qa_chain.run(question)
            st.write("### Answer")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
