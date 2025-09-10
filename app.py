import os
import tempfile
from typing import List, Tuple
import io
import json

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
def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        text += page.get_text()
        if use_ocr:
            image_list = page.get_images(full=True)
            for _, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                text += ocr_image(image_bytes)
    return text


def extract_text_from_docx(docx_path: str, use_ocr: bool = True) -> str:
    from docx import Document
    doc = Document(docx_path)
    text = "\n".join(para.text for para in doc.paragraphs)

    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_bytes = rel.target_part.blob
                text += ocr_image(image_bytes)
    return text


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_text_from_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], use_ocr: bool) -> List[Tuple[str, str]]:
    """Extracts text from uploaded files."""
    extracted_texts = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmp_path = tmpfile.name

            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".pdf":
                text = extract_text_from_pdf(tmp_path, use_ocr)
            elif file_extension == ".docx":
                text = extract_text_from_docx(tmp_path, use_ocr)
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


# --- VectorDB Builder ---
def build_vectordb(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 32}
    )
    return FAISS.from_documents(chunks, embeddings)


# --- Modern LangChain QA System ---
class DocumentQA:
    def __init__(self, texts: List[Tuple[str, str]]):
        self.documents = [Document(page_content=text, metadata={"source": source}) for source, text in texts]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(self.documents)

        if not self.chunks:
            raise ValueError("Text splitting produced no chunks.")

        self.vectordb = build_vectordb(self.chunks)
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
        )

    def add_documents(self, texts: List[Tuple[str, str]]):
        new_docs = [Document(page_content=text, metadata={"source": source}) for source, text in texts]
        self.documents.extend(new_docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_chunks = splitter.split_documents(new_docs)
        self.chunks.extend(new_chunks)

        self.vectordb.add_documents(new_chunks)

    def answer_question(self, question: str) -> str:
        trigger_phrases = ["all documents", "uploaded docs", "both files", "both documents"]
        if any(phrase in question.lower() for phrase in trigger_phrases):
            return self.summarize_all_documents()

        relevant_docs = self.vectordb.similarity_search(question, k=3)
        context = "\n\n".join(
            [f"Source: {doc.metadata.get('source','unknown')}\n{doc.page_content}" for doc in relevant_docs]
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(context=context, question=question)

    def summarize_all_documents(self) -> str:
        summaries = []
        for doc in self.documents:
            try:
                summary = self.llm.predict(
                    f"Summarize the document '{doc.metadata['source']}' in 5 bullet points. "
                    f"Clearly state what kind of document it is (e.g., CV, research paper, report). "
                    f"Here is the content:\n\n{doc.page_content}"
                )
                summaries.append(f"### {doc.metadata['source']}\n{summary}")
            except Exception as e:
                summaries.append(f"### {doc.metadata['source']}\n(Summarization failed: {e})")
        return "\n\n".join(summaries)


# --- Streamlit Interface ---
st.set_page_config(page_title="Multimodal Document Q&A Assistant", page_icon="📄", layout="wide")

st.title("📄 Multimodal Document Q&A Assistant")
st.markdown("Upload your documents (PDF, DOCX, TXT) and ask questions. "
            "This app uses OCR (optional) to extract text from images in your documents. Powered by Llama 3.3 70B via Groq.")

# API Key Management
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    with st.sidebar:
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password", key="api_key_input")
        st.markdown("[Get your Groq API key here](https://console.groq.com/keys)")

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# --- Session-based State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Sidebar: File Uploader + OCR toggle
with st.sidebar:
    st.header("Upload Your Documents")
    use_ocr = st.checkbox("Enable OCR for images", value=True)
    uploaded_files = st.file_uploader(
        "Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            with st.spinner("Processing new documents..."):
                try:
                    extracted_texts = get_text_from_files(new_files, use_ocr)
                    if extracted_texts:
                        if st.session_state.qa_system is None:
                            st.session_state.qa_system = DocumentQA(extracted_texts)
                        else:
                            st.session_state.qa_system.add_documents(extracted_texts)

                        for f in new_files:
                            st.session_state.processed_files.add(f.name)

                        st.success(f"Successfully added {len(extracted_texts)} new documents!")
                    else:
                        st.error("No text extracted from new documents.")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    st.divider()
    st.subheader("🗑️ New Chat")
    if st.button("Start New Chat"):
        st.session_state.messages = []
        st.session_state.qa_system = None
        st.session_state.processed_files = set()
        st.success("New chat started!")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat
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
