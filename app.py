import os
import io
from typing import List
import hashlib
import json
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PIL import Image
import pytesseract
from supabase import create_client

# --- Streamlit Page Config ---
st.set_page_config(page_title="DocQ&A", page_icon="📄", layout="wide")

# --- Supabase Setup ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- OAuth Login Handling ---
st.sidebar.header("Login")
user_id = None

if SUPABASE_URL and SUPABASE_KEY:
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL}"
    query_params = st.query_params
    access_token = query_params.get("access_token")
    refresh_token = query_params.get("refresh_token")

    if "session" not in st.session_state:
        st.session_state.session = None

    if access_token and refresh_token and st.session_state.session is None:
        try:
            session = supabase.auth.set_session(access_token, refresh_token)
            st.session_state.session = session
        except Exception as e:
            st.warning(f"Auth session failed: {e}")

    logged_in = (
        st.session_state.session is not None
        and getattr(st.session_state.session, "user", None) is not None
    )
    if logged_in:
        user = st.session_state.session.user
        user_id = user.id
        st.sidebar.success(f"✅ Logged in as {user.email}")
        if st.sidebar.button("🚪 Logout"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.session = None
            st.query_params.clear()
            st.rerun()
    else:
        st.sidebar.markdown(f"[🔑 Login with Google]({login_url})")
else:
    st.sidebar.warning("⚠️ Supabase not configured")

# --- Text Extraction Functions ---
def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""

def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        if use_ocr:
            for img_meta in page.get_images(full=True):
                xref = img_meta
                base_image = doc.extract_image(xref)
                text += ocr_image(base_image["image"])
    return text

def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                text += ocr_image(rel.target_part.blob)
    return text

def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    return file_bytes.read().decode("utf-8")

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

# --- Simple TF-IDF Embeddings ---
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.embeddings.base import Embeddings

class TFIDFEmbeddings(Embeddings):
    def __init__(self, max_features=384):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        self.is_fitted = False
        self.dimension = max_features

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        vectors = self.vectorizer.transform(texts)
        dense_vectors = vectors.toarray()
        result = []
        for vector in dense_vectors:
            if len(vector) < self.dimension:
                padded = np.pad(vector, (0, self.dimension - len(vector)), 'constant')
                result.append(padded.tolist())
            else:
                result.append(vector[:self.dimension].tolist())
        return result

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * self.dimension
        dense_vector = self.vectorizer.transform([text]).toarray()
        if len(dense_vector) < self.dimension:
            padded = np.pad(dense_vector, (0, self.dimension - len(dense_vector)), 'constant')
            return padded.tolist()
        else:
            return dense_vector[:self.dimension].tolist()

# --- Enhanced Document Manager ---
class DocumentManager:
    def __init__(self):
        self.documents = []
        self.processed_files = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. If you cannot find the answer in the context, say \"I cannot find the answer in the provided documents.\"\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        )

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int):
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed (duplicate content detected)"
        doc = Document(page_content=content, metadata={
            "source": filename,
            "file_hash": file_hash,
            "file_size": file_size
        })
        self.documents.append(doc)
        self.processed_files[file_hash] = {
            "name": filename,
            "size": file_size,
            "processed_time": str(st.session_state.get('current_time', 'unknown'))
        }
        return True, f"Successfully processed '{filename}'"

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
            return "No documents have been uploaded yet. Please upload some documents first."
        if not self.vectordb:
            return "Sorry, the document search system is not working properly."
        try:
            if not getattr(self.embeddings, "is_fitted", False):
                return "Sorry, the document search system is initializing. Please try again after processing documents."
            docs = self.vectordb.similarity_search(question, k=5)
            if not docs:
                return "I cannot find any relevant information in the documents to answer your question."
            context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs])
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            return chain.run(context=context, question=question)
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def get_document_count(self):
        return len(self.documents)

    def get_document_list(self):
        return [(doc.metadata['source'], doc.metadata.get('file_size', 0)) for doc in self.documents]

    def get_processed_files_info(self):
        return self.processed_files

    def remove_file(self, filename: str):
        self.documents = [doc for doc in self.documents if doc.metadata['source'] != filename]
        file_hash_to_remove = None
        for file_hash, file_info in self.processed_files.items():
            if file_info['name'] == filename:
                file_hash_to_remove = file_hash
                break
        if file_hash_to_remove:
            del self.processed_files[file_hash_to_remove]
        if self.documents:
            self._rebuild_vectordb()
        else:
            self.vectordb = None

    def clear_all(self):
        self.documents = []
        self.processed_files = {}
        self.vectordb = None

# --- Chat Persistence ---
def save_chat(user_id, messages):
    if user_id and supabase:
        try:
            supabase.table("chats").upsert({"user_id": user_id, "messages": messages}).execute()
        except Exception as e:
            st.warning(f"Failed to save chat: {e}")

def load_chat(user_id):
    if user_id and supabase:
        try:
            res = supabase.table("chats").select("messages").eq("user_id", user_id).execute()
            return res.data[0]["messages"] if res.data else []
        except Exception as e:
            st.warning(f"Failed to load chat: {e}")
            return []
    return []

# --- Initialize Session State ---
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if "messages" not in st.session_state:
    if user_id:
        st.session_state.messages = load_chat(user_id)
    else:
        st.session_state.messages = []

if "staged_files" not in st.session_state:
    st.session_state.staged_files = []

# --- Main Interface ---
st.title("📄 DocQ&A — Your AI Assistant")

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader_stable"
    )

    use_ocr = st.checkbox("Enable OCR for images", True)

    if uploaded_files:
        processed_hashes = set(st.session_state.doc_manager.get_processed_files_info().keys())
        staged_hashes = {f["file_hash"] for f in st.session_state.staged_files}
        for f in uploaded_files:
            try:
                data = f.getvalue()
                file_hash = get_file_hash(data)
                if file_hash in processed_hashes or file_hash in staged_hashes:
                    continue
                st.session_state.staged_files.append({
                    "name": f.name,
                    "bytes": data,
                    "size": f.size,
                    "ext": os.path.splitext(f.name)[1].lower(),
                    "file_hash": file_hash
                })
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")

    if st.session_state.staged_files:
        st.write(f"**Files staged for processing:** {len(st.session_state.staged_files)}")
        for i, f in enumerate(st.session_state.staged_files, 1):
            st.write(f"⏳ {i}. {f['name']} ({f['size']/(1024*1024):.1f} MB)")

    if st.session_state.staged_files and st.button("📤 Process Files", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_count = 0
        skipped_count = 0
        error_count = 0
        total_files = len(st.session_state.staged_files)

        for i, staged in enumerate(st.session_state.staged_files):
            try:
                status_text.text(f"Processing {staged['name']}...")
                progress_bar.progress(i / max(1, total_files))

                ext = staged["ext"]
                file_bytes_io = io.BytesIO(staged["bytes"])

                text = ""
                if ext == ".pdf":
                    text = extract_text_from_pdf(file_bytes_io, use_ocr)
                elif ext == ".docx":
                    text = extract_text_from_docx(file_bytes_io, use_ocr)
                elif ext == ".txt":
                    text = extract_text_from_txt(file_bytes_io)
                else:
                    st.warning(f"Unsupported file type: {staged['name']}")
                    error_count += 1
                    continue

                if text.strip():
                    success, message = st.session_state.doc_manager.add_file(
                        staged["name"], text, staged["file_hash"], staged["size"]
                    )
                    if success:
                        st.success(f"✅ {message}")
                        processed_count += 1
                    else:
                        st.info(f"ℹ️ {message}")
                        skipped_count += 1
                else:
                    st.warning(f"⚠️ No text found in {staged['name']}")
                    error_count += 1
            except Exception as e:
                st.error(f"❌ Error with {staged['name']}: {e}")
                error_count += 1

        st.session_state.doc_manager._rebuild_vectordb()

        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        st.success("🎉 Processing Summary:")
        st.write(f"- ✅ Processed: {processed_count}")
        st.write(f"- ℹ️ Skipped (duplicates): {skipped_count}")
        st.write(f"- ❌ Errors: {error_count}")

        st.session_state.staged_files = []
        st.rerun()

# --- Document status and management ---
st.divider()

doc_count = st.session_state.doc_manager.get_document_count()
if doc_count > 0:
    st.success(f"📚 {doc_count} documents loaded")
    with st.expander("📋 Manage Documents", expanded=False):
        doc_list = st.session_state.doc_manager.get_document_list()
        for i, (doc_name, file_size) in enumerate(doc_list, 1):
            file_size_mb = file_size / (1024 * 1024) if file_size > 0 else 0
            col1, col2 = st.columns([5,1])
            with col1:
                st.write(f"{i}. {doc_name} ({file_size_mb:.1f} MB)")
            with col2:
                if st.button("🗑️", key=f"delete_{i}", help=f"Remove {doc_name}"):
                    st.session_state.doc_manager.remove_file(doc_name)
                    st.success(f"Removed {doc_name}")
                    st.rerun()
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state.doc_manager.clear_all()
            st.session_state.messages = []
            st.success("All documents cleared!")
            st.rerun()
else:
    st.info("📤 Upload documents to get started")

processed_files_info = st.session_state.doc_manager.get_processed_files_info()
if processed_files_info:
    with st.expander("📊 Processing Statistics"):
        total_size = sum(info['size'] for info in processed_files_info.values())
        total_size_mb = total_size / (1024 * 1024)
        st.write(f"Total files processed: {len(processed_files_info)}")
        st.write(f"Total size: {total_size_mb:.1f} MB")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about your documents..."):
    st.session_state.messages.append({"role": "user
