import os
import io
import hashlib
import streamlit as st
from PIL import Image
import pytesseract
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.embeddings.base import Embeddings

# ----------------- CONFIG ----------------- #
st.set_page_config(
    page_title="DocQ&A - Smart Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- BACKGROUND ANIMATION ----------------- #
st.markdown(
    """
<style>
.background-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    opacity: 0.18; /* adjust transparency */
    pointer-events: none; /* allow clicks through background */
}
</style>

<div class="background-wrapper">
<svg viewBox="0 0 1440 900" fill="none" xmlns="http://www.w3.org/2000/svg">
""" +
"\n".join(
    f"""
    <path d="{path}" stroke="{color}" stroke-width="2.3" stroke-linecap="round">
        <animate attributeName="stroke-dasharray" values="50 800;20 800;50 800" dur="10s" repeatCount="indefinite"/>
        <animate attributeName="stroke-dashoffset" values="800;0;800" dur="10s" repeatCount="indefinite"/>
    </path>
    """ for path, color in zip(paths, colors)
) + """
</svg>
</div>
""",
    unsafe_allow_html=True
)

# ----------------- CUSTOM CSS ----------------- #
st.markdown(
    """
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    /* (keep rest of your CSS for cards, chat, sidebar, etc.) */
</style>
""",
    unsafe_allow_html=True
)

# ----------------- UTILS ----------------- #
def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

def normalize_text(text: str) -> str:
    return " ".join(text.split())

def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception:
        return ""

def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
        if use_ocr:
            for img_meta in page.get_images(full=True):
                try:
                    base_image = doc.extract_image(img_meta[0])
                    ocr_text = ocr_image(base_image["image"])
                    if ocr_text.strip() and ocr_text not in text:
                        text += "\n" + ocr_text
                except Exception:
                    continue
    return normalize_text(text)

def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        try:
            for rel in doc.part.rels.values():
                try:
                    if "image" in rel.target_ref:
                        ocr_text = ocr_image(rel.target_part.blob)
                        if ocr_text.strip() and ocr_text not in text:
                            text += "\n" + ocr_text
                except Exception:
                    continue
        except Exception:
            pass
    return normalize_text(text)

def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    try:
        text = file_bytes.read().decode("utf-8")
    except Exception:
        text = file_bytes.read().decode("utf-8", errors="ignore")
    return normalize_text(text)

# ----------------- TFIDF EMBEDDINGS ----------------- #
class TFIDFEmbeddings(Embeddings):
    def __init__(self, max_features: int = 384):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )
        self.is_fitted = False
        self.dimension = max_features

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        vectors = self.vectorizer.transform(texts).toarray()
        return [self._pad(v) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * self.dimension
        vec = self.vectorizer.transform([text]).toarray()[0]
        return self._pad(vec)

    def _pad(self, vector: np.ndarray) -> List[float]:
        if len(vector) < self.dimension:
            vector = np.pad(vector, (0, self.dimension - len(vector)), "constant")
        return vector[: self.dimension].tolist()

# ----------------- DOCUMENT MANAGER ----------------- #
class DocumentManager:
    def __init__(self):
        self.documents: List[Document] = []
        self.processed_files = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        try:
            self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        except Exception:
            self.llm = None
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question comprehensively. If you cannot find the answer in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:""",
        )

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int):
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed (duplicate content)"
        if not content.strip():
            return False, f"File '{filename}' appears to be empty or unreadable"
        doc = Document(page_content=content, metadata={"source": filename, "file_hash": file_hash, "file_size": file_size})
        self.documents.append(doc)
        self.processed_files[file_hash] = {"name": filename, "size": file_size, "word_count": len(content.split())}
        return True, f"✅ Successfully processed '{filename}' ({len(content.split())} words)"

    def _rebuild_vectordb(self):
        if not self.documents:
            self.vectordb = None
            return
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(self.documents)
            all_texts = [c.page_content for c in chunks]
            self.embeddings = TFIDFEmbeddings()
            _ = self.embeddings.embed_documents(all_texts)
            self.vectordb = FAISS.from_documents(chunks, self.embeddings)
        except Exception as e:
            st.error(f"Failed to build vector database: {e}")
            self.vectordb = None

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "❌ No documents uploaded. Please upload some documents first to ask questions."
        if not self.vectordb:
            return "⚠️ Document search index is not ready. Please try uploading documents again."
        if not self.llm:
            return "❌ Language model is not available. Please check your API configuration."
        try:
            docs = self.vectordb.similarity_search(question, k=5)
            if not docs:
                return "🔍 I cannot find any relevant information in the uploaded documents for your question."
            context = "\n\n".join(
                [f"📄 Source: {d.metadata.get('source','Unknown')}\n{d.page_content}" for d in docs]
            )
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            return chain.run(context=context, question=question)
        except Exception as e:
            return f"❌ Error processing your question: {str(e)}"

    def get_stats(self):
        total_files = len(self.processed_files)
        total_words = sum([info.get("word_count", 0) for info in self.processed_files.values()])
        total_size = sum([info.get("size", 0) for info in self.processed_files.values()])
        return {"files": total_files, "words": total_words, "size_mb": round(total_size / (1024 * 1024), 2)}

# ----------------- SESSION STATE ----------------- #
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- MAIN APP ----------------- #
st.markdown(
    """
<div class="main-header">
    <h1>🧠 DocQ&A - Smart Document Assistant</h1>
    <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
        Upload your documents and ask intelligent questions powered by AI
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar for document upload
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True, help="Supported formats: PDF, DOCX, TXT"
    )
    use_ocr = st.checkbox("🔍 Enable OCR for images", value=True, help="Extract text from images in documents")
    if st.button("🗑️ Clear All Documents"):
        st.session_state.doc_manager = DocumentManager()
        st.session_state.messages = []
        st.success("All documents cleared!")
        st.rerun()
    stats = st.session_state.doc_manager.get_stats()
    if stats["files"] > 0:
        st.markdown("### 📊 Document Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Files", stats["files"])
            st.metric("💾 Size (MB)", stats["size_mb"])
        with col2:
            st.metric("📝 Words", f"{stats['words']:,}")
        st.markdown("### 📋 Processed Files")
        seen_names = set()
        for file_info in st.session_state.doc_manager.processed_files.values():
            if file_info["name"] not in seen_names:
                st.markdown(f"• **{file_info['name']}** ({file_info['word_count']:,} words)")
                seen_names.add(file_info["name"])

# Process uploaded files
if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f"Processing {uploaded_file.name}...")
        file_data = uploaded_file.getvalue()
        file_bytes = io.BytesIO(file_data)
        file_bytes.seek(0)
        file_hash = get_file_hash(file_data)
        if file_hash in st.session_state.doc_manager.processed_files:
            continue
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        text_content = ""
        try:
            if file_extension == ".pdf":
                text_content = extract_text_from_pdf(file_bytes, use_ocr)
            elif file_extension == ".docx":
                text_content = extract_text_from_docx(file_bytes, use_ocr)
            elif file_extension == ".txt":
                text_content = extract_text_from_txt(file_bytes)
            success, message = st.session_state.doc_manager.add_file(uploaded_file.name, text_content, file_hash, uploaded_file.size)
            if success: st.success(message)
            else: st.info(message)
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    with st.spinner("🔄 Building search index..."):
        st.session_state.doc_manager._rebuild_vectordb()
    progress_bar.empty()
    status_text.empty()
    st.success("🎉 All documents processed successfully!")

# Main chat interface
if st.session_state.doc_manager.documents:
    st.markdown("### 💬 Chat with your Documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask me anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.doc_manager.answer_question(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.markdown(
        """
    <div class="feature-card">
        <h3>🚀 Get Started</h3>
        <p>Welcome to DocQ&A! Here's how to use this intelligent document assistant:</p>
        <ol>
            <li><strong>Upload Documents:</strong> Use the sidebar to upload PDF, DOCX, or TXT files</li>
            <li><strong>Enable OCR:</strong> Check the OCR option to extract text from images in your documents</li>
            <li><strong>Ask Questions:</strong> Once uploaded, ask any questions about your documents</li>
            <li><strong>Get Smart Answers:</strong> The AI will search through your documents and provide detailed answers</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
<div class="footer-content">
    <p>🚀 Built with Streamlit • Powered by AI • Made with ❤️</p>
</div>
""", unsafe_allow_html=True)
