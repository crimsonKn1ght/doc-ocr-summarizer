import os
import tempfile
from typing import List, Tuple
import io

import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PIL import Image
import pytesseract

# --- OCR Function ---

def ocr_image(image_bytes: bytes) -> str:
    """Performs OCR on an image and returns the extracted text."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.warning(f"OCR failed for an image: {e}")
        return ""

# --- Text Extraction ---

def extract_text_from_pdf(pdf_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        # Extract regular text
        text += page.get_text()
        # Extract and OCR images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            text += ocr_image(image_bytes)
    return text

def extract_text_from_docx(docx_path: str) -> str:
    from docx import Document
    doc = Document(docx_path)
    text = "\n".join(para.text for para in doc.paragraphs)

    # Extract and OCR images
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_bytes = rel.target_part.blob
            text += ocr_image(image_bytes)
    return text

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def get_text_from_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Tuple[str, str]]:
    """Extracts text from uploaded files."""
    extracted_texts = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmp_path = tmpfile.name

            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".pdf":
                text = extract_text_from_pdf(tmp_path)
            elif file_extension == ".docx":
                text = extract_text_from_docx(tmp_path)
            elif file_extension == ".txt":
                text = extract_text_from_txt(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            if text and text.strip():
                extracted_texts.append((uploaded_file.name, text))
            else:
                st.warning(f"No text could be extracted from {uploaded_file.name}")

            os.remove(tmp_path)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    return extracted_texts

# --- Modern LangChain QA System ---

class DocumentQA:
    def __init__(self, texts: List[Tuple[str, str]]):
        # Create documents
        self.documents = [Document(page_content=text, metadata={"source": source}) for source, text in texts]
        
        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(self.documents)
        
        if not self.chunks:
            raise ValueError("Text splitting produced no chunks.")
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = FAISS.from_documents(self.chunks, embeddings)
        
        # Initialize LLM
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
        )
    
    def answer_question(self, question: str) -> str:
        # Retrieve relevant documents
        relevant_docs = self.vectordb.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create chain and get answer
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        response = chain.run(context=context, question=question)
        
        return response

# --- Streamlit Interface ---

st.set_page_config(page_title="Multimodal Document Q&A Assistant", page_icon="📄", layout="wide")

st.title("📄 Multimodal Document Q&A Assistant")
st.markdown("Upload your documents (PDF, DOCX, TXT) and ask questions. This app uses OCR to extract text from images in your documents. Powered by Llama 3.3 70B via Groq.")

# API Key Management
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    with st.sidebar:
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password", key="api_key_input")
        st.markdown("[Get your Groq API key here](https://console.groq.com/keys)")

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# Set the API key for langchain_groq
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize QA system
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

# File Uploader
with st.sidebar:
    st.header("Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    
    if uploaded_files and (st.session_state.qa_system is None or st.button("Reprocess Documents")):
        with st.spinner("Processing documents..."):
            try:
                extracted_texts = get_text_from_files(uploaded_files)
                if extracted_texts:
                    st.session_state.qa_system = DocumentQA(extracted_texts)
                    st.success(f"Successfully processed {len(extracted_texts)} documents!")
                else:
                    st.error("Could not extract any text from the uploaded documents.")
            except Exception as e:
                st.error(f"Error processing documents: {e}")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.qa_system is not None:
        with st.spinner("Getting your answer..."):
            try:
                result = st.session_state.qa_system.answer_question(prompt)
                with st.chat_message("assistant"):
                    st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        with st.chat_message("assistant"):
            message = "Please upload at least one document first to start asking questions."
            st.markdown(message)
        st.session_state.messages.append({"role": "assistant", "content": message})
