import os
import io
import hashlib
import json
from typing import List, Tuple, Dict, Any

import streamlit as st
from PIL import Image
import pytesseract

# Optional dependencies (may not be installed in all environments)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    # ChatGroq may not be available in all envs; keep a safe fallback
    try:
        from langchain_groq import ChatGroq
    except Exception:
        ChatGroq = None
except Exception:
    # If langchain isn't installed, we'll still provide a graceful fallback
    Document = None
    RecursiveCharacterTextSplitter = None
    FAISS = None
    PromptTemplate = None
    LLMChain = None
    ChatGroq = None

# Simple TF-IDF embeddings (pure python sklearn implementation)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from langchain.embeddings.base import Embeddings
except Exception:
    TfidfVectorizer = None
    np = None
    Embeddings = None

# Supabase optional client
try:
    from supabase import create_client
except Exception:
    create_client = None

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="DocQ&A", page_icon="📄", layout="wide")

# ---------------- Supabase / OAuth setup (optional) ----------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if st.secrets else None
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if st.secrets else None
REDIRECT_URL = st.secrets.get("REDIRECT_URL") if st.secrets else None

supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# ---------------- Utilities ----------------

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()


def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""


def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if fitz is None:
        st.error("PyMuPDF (fitz) not available. Install with `pip install pymupdf` to extract PDFs.")
        return ""
    doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
    text = ""
    for page in doc:
        try:
            text += page.get_text()
        except Exception:
            pass
        if use_ocr:
            try:
                img_list = page.get_images(full=True)
                for img_meta in img_list:
                    xref = img_meta[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if img_bytes:
                        text += "\n" + ocr_image(img_bytes)
            except Exception:
                # continue gracefully if image extraction fails
                pass
    return text


def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if DocxDocument is None:
        st.error("python-docx not available. Install with `pip install python-docx` to extract .docx files.")
        return ""
    file_bytes.seek(0)
    doc = DocxDocument(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        try:
            for rel in doc.part.rels.values():
                if hasattr(rel, "target_part") and getattr(rel.target_part, "blob", None):
                    img_bytes = rel.target_part.blob
                    text += "\n" + ocr_image(img_bytes)
        except Exception:
            pass
    return text


def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    file_bytes.seek(0)
    try:
        return file_bytes.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


# ---------------- TF-IDF Embeddings ----------------
class TFIDFEmbeddings(Embeddings if Embeddings is not None else object):
    def __init__(self, max_features: int = 384):
        if TfidfVectorizer is None or np is None:
            raise RuntimeError("scikit-learn and numpy are required for TFIDFEmbeddings")
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
        dense_vector = self.vectorizer.transform([text]).toarray()[0]
        if len(dense_vector) < self.dimension:
            padded = np.pad(dense_vector, (0, self.dimension - len(dense_vector)), 'constant')
            return padded.tolist()
        else:
            return dense_vector[:self.dimension].tolist()


# ---------------- Document Manager ----------------
class DocumentManager:
    def __init__(self):
        self.documents: List[Any] = []
        self.processed_files: Dict[str, Dict[str, Any]] = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        # Prefer ChatGroq if available, otherwise llm stays None and LLMChain will not run
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0) if ChatGroq else None
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question. If you cannot find the answer in the context, "
                "say \"I cannot find the answer in the provided documents.\"\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        ) if PromptTemplate else None

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int) -> Tuple[bool, str]:
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed (duplicate content detected)"
        if Document is None:
            # store minimal metadata if langchain Document not present
            doc = type("SimpleDoc", (), {"page_content": content, "metadata": {"source": filename, "file_hash": file_hash, "file_size": file_size}})()
        else:
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
        # Do NOT rebuild index here; do it once per batch
        return True, f"Successfully processed '{filename}'"

    def _rebuild_vectordb(self):
        if not self.documents:
            self.vectordb = None
            return
        if RecursiveCharacterTextSplitter is None or FAISS is None:
            st.error("Langchain text splitter or FAISS integration missing. Install langchain and langchain-commun ity packages.")
            return
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(self.documents)
            all_texts = [c.page_content for c in chunks]

            # Refit TF-IDF on full chunk corpus
            self.embeddings = TFIDFEmbeddings()
            _ = self.embeddings.embed_documents(all_texts)

            # Build FAISS index
            self.vectordb = FAISS.from_documents(chunks, self.embeddings)
        except Exception as e:
            st.error(f"Failed to build vector database: {e}")
            self.vectordb = None

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "No documents have been uploaded yet. Please upload some documents first."
        if not self.vectordb:
            return "Sorry, the document search system is not working properly (index not built)."
        try:
            if not getattr(self.embeddings, "is_fitted", False):
                return "Sorry, the document search system is initializing. Please try again after processing documents."
            docs = self.vectordb.similarity_search(question, k=5)
            if not docs:
                return "I cannot find any relevant information in the documents to answer your question."
            context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs])
            if self.llm and self.prompt_template and LLMChain:
                chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
                return chain.run(context=context, question=question)
            else:
                # Fallback: provide the top-k chunk contents as the "answer" for manual inspection
                snippets = "\n---\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:1000]}..." for d in docs])
                return f"(LLM unavailable) Found the following relevant snippets:\n\n{snippets}"
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_document_list(self) -> List[Tuple[str, int]]:
        result = []
        for doc in self.documents:
            meta = getattr(doc, "metadata", {})
            result.append((meta.get('source', 'unknown'), meta.get('file_size', 0)))
        return result

    def get_processed_files_info(self) -> Dict[str, Dict[str, Any]]:
        return self.processed_files

    def remove_file(self, filename: str):
        self.documents = [doc for doc in self.documents if doc.metadata.get('source') != filename]
        file_hash_to_remove = None
        for file_hash, file_info in list(self.processed_files.items()):
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


# ---------------- Chat Persistence ----------------
def save_chat(user_id: str, messages: List[Dict[str, str]]):
    if user_id and supabase:
        try:
            supabase.table("chats").upsert({"user_id": user_id, "messages": messages}).execute()
        except Exception as e:
            st.warning(f"Failed to save chat: {e}")


def load_chat(user_id: str) -> List[Dict[str, str]]:
    if user_id and supabase:
        try:
            res = supabase.table("chats").select("messages").eq("user_id", user_id).execute()
            data = res.data
            if data:
                return data[0].get("messages", [])
        except Exception as e:
            st.warning(f"Failed to load chat: {e}")
    return []


# ---------------- Initialize session state ----------------
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if "messages" not in st.session_state:
    # user_id may be None if not logged in
    st.session_state.messages = []

if "staged_files" not in st.session_state:
    st.session_state.staged_files = []

# ---------------- UI ----------------
st.title("📄 DocQ&A — Your AI Assistant")

# Sidebar: Login (simple link if supabase available)
with st.sidebar:
    st.header("Login")
    user_id = None
    if supabase and REDIRECT_URL:
        query_params = st.experimental_get_query_params()
        access_token = query_params.get("access_token", [None])[0]
        refresh_token = query_params.get("refresh_token", [None])[0]
        if access_token and refresh_token and st.session_state.get("session") is None:
            try:
                session = supabase.auth.set_session({"access_token": access_token, "refresh_token": refresh_token})
                st.session_state.session = session
            except Exception as e:
                st.warning(f"Auth session failed: {e}")
        logged_in = st.session_state.get("session") is not None and getattr(st.session_state.get("session"), "user", None) is not None
        if logged_in:
            user = st.session_state.session.user
            user_id = user.id
            st.success(f"✅ Logged in as {getattr(user, 'email', 'unknown')}")
            if st.button("🚪 Logout"):
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                st.session_state.session = None
                st.experimental_set_query_params()
                st.experimental_rerun()
        else:
            login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL}"
            st.markdown(f"[🔑 Login with Google]({login_url})")
    else:
        st.warning("⚠️ Supabase not configured")

    st.divider()
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader_stable"
    )

    use_ocr = st.checkbox("Enable OCR for images", value=True)

    # Stage uploaded files (persist across reruns via session_state)
    if uploaded_files:
        processed_hashes = set(st.session_state.doc_manager.get_processed_files_info().keys())
        staged_hashes = {f.get("file_hash") for f in st.session_state.staged_files}
        for f in uploaded_files:
            try:
                data = f.getvalue()
                file_hash = get_file_hash(data)
                if file_hash in processed_hashes or file_hash in staged_hashes:
                    continue
                _, ext = os.path.splitext(f.name)
                st.session_state.staged_files.append({
                    "name": f.name,
                    "bytes": data,
                    "size": f.size,
                    "ext": ext.lower(),
                    "file_hash": file_hash
                })
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")

    if st.session_state.staged_files:
        st.write(f"**Files staged for processing:** {len(st.session_state.staged_files)}")
        for i, sf in enumerate(st.session_state.staged_files, 1):
            st.write(f"{i}. {sf['name']} ({sf['size']/(1024*1024):.2f} MB)")

    if st.session_state.staged_files and st.button("📤 Process Files"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        total_files = len(st.session_state.staged_files)
        processed_count = skipped_count = error_count = 0

        for i, staged in enumerate(list(st.session_state.staged_files)):
            status_text.text(f"Processing {staged['name']} ({i+1}/{total_files})...")
            progress_bar.progress(i / max(1, total_files))
            try:
                file_bytes_io = io.BytesIO(staged["bytes"])
                text_content = ""
                if staged["ext"] == ".pdf":
                    text_content = extract_text_from_pdf(file_bytes_io, use_ocr)
                elif staged["ext"] == ".docx":
                    text_content = extract_text_from_docx(file_bytes_io, use_ocr)
                elif staged["ext"] == ".txt":
                    text_content = extract_text_from_txt(file_bytes_io)
                else:
                    st.warning(f"Unsupported file type: {staged['name']}")
                    error_count += 1
                    continue

                if text_content.strip():
                    success, message = st.session_state.doc_manager.add_file(
                        staged["name"], text_content, staged["file_hash"], staged["size"]
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

        # Rebuild index once
        st.session_state.doc_manager._rebuild_vectordb()
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        st.success("🎉 Processing Summary:")
        st.write(f"- ✅ Processed: {processed_count}")
        st.write(f"- ℹ️ Skipped (duplicates): {skipped_count}")
        st.write(f"- ❌ Errors: {error_count}")

        # Clear staging
        st.session_state.staged_files = []
        st.experimental_rerun()

    st.divider()
    # Document management
    doc_count = st.session_state.doc_manager.get_document_count()
    if doc_count > 0:
        st.success(f"📚 {doc_count} documents loaded")
        if st.button("🗑️ Clear All Documents"):
            st.session_state.doc_manager.clear_all()
            st.session_state.messages = []
            st.success("All documents cleared!")
            st.experimental_rerun()
    else:
        st.info("📤 Upload documents to get started")

    # Processing statistics
    processed_files_info = st.session_state.doc_manager.get_processed_files_info()
    if processed_files_info:
        with st.expander("📊 Processing Statistics"):
            total_size = sum(info['size'] for info in processed_files_info.values())
            total_size_mb = total_size / (1024 * 1024)
            st.write(f"Total files processed: {len(processed_files_info)}")
            st.write(f"Total size: {total_size_mb:.1f} MB")

# ---------------- Main chat area ----------------
st.divider()
cols = st.columns([3, 1])
with cols[0]:
    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) if message.get("content") else None

    prompt = st.chat_input("Ask me about your documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.doc_manager.answer_question(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # save chat if possible
        try:
            save_chat(user_id, st.session_state.messages)
        except Exception:
            pass

with cols[1]:
    st.header("Documents")
    doc_list = st.session_state.doc_manager.get_document_list()
    if doc_list:
        for i, (doc_name, file_size) in enumerate(doc_list, 1):
            file_size_mb = file_size / (1024 * 1024) if file_size > 0 else 0
            st.write(f"{i}. {doc_name} ({file_size_mb:.2f} MB)")
            if st.button("🗑️ Remove", key=f"remove_{i}"):
                st.session_state.doc_manager.remove_file(doc_name)
                st.success(f"Removed {doc_name}")
                st.experimental_rerun()
    else:
        st.info("No documents loaded.")


# End of file
