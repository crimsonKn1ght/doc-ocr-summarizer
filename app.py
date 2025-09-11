import os
import io
from typing import List, Tuple

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

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # --- OAuth Login Handling ---
    st.sidebar.header("Login")
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL}"
    
    query_params = st.query_params
    access_token = query_params.get("access_token")
    refresh_token = query_params.get("refresh_token")
    
    if access_token and refresh_token:
        session = supabase.auth.set_session(
            {"access_token": access_token, "refresh_token": refresh_token}
        )
        st.session_state.session = session
    
    logged_in = (
        "session" in st.session_state
        and st.session_state.session
        and st.session_state.session.user
    )
    
    if logged_in:
        user = st.session_state.session.user
        user_id = user.id
        st.sidebar.success(f"✅ Logged in as {user.email}")
        if st.sidebar.button("🚪 Logout"):
            supabase.auth.sign_out()
            st.session_state.session = None
            st.query_params.clear()
            st.rerun()
    else:
        st.sidebar.markdown(f"[🔒 Login with Google]({login_url})")
        user_id = None
else:
    st.sidebar.warning("⚠️ Supabase not configured")
    user_id = None
    supabase = None

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
                xref = img_meta[0]
                base_image = doc.extract_image(xref)
                text += ocr_image(base_image["image"])
    return text

def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    from docx import Document
    doc = Document(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                text += ocr_image(rel.target_part.blob)
    return text

def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    return file_bytes.read().decode("utf-8")

# --- Simple TF-IDF Embeddings ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        
        vector = self.vectorizer.transform([text])
        dense_vector = vector.toarray()[0]
        
        if len(dense_vector) < self.dimension:
            padded = np.pad(dense_vector, (0, self.dimension - len(dense_vector)), 'constant')
            return padded.tolist()
        else:
            return dense_vector[:self.dimension].tolist()

# --- Document Manager ---
class DocumentManager:
    def __init__(self):
        self.documents = []
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        )
    
    def add_file(self, filename: str, content: str):
        """Add a single file to the document collection"""
        doc = Document(page_content=content, metadata={"source": filename})
        self.documents.append(doc)
        
        # Rebuild vector database with all documents
        self._rebuild_vectordb()
    
    def _rebuild_vectordb(self):
        """Rebuild the vector database with all documents"""
        if self.documents:
            try:
                # Split all documents
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(self.documents)
                
                # Create vector database
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
            # Search for relevant documents
            docs = self.vectordb.similarity_search(question, k=3)
            
            if not docs:
                return "I cannot find any relevant information in the documents to answer your question."
            
            # Build context
            context = "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])
            
            # Generate answer
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            return chain.run(context=context, question=question)
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_document_count(self):
        return len(self.documents)
    
    def get_document_list(self):
        return [doc.metadata['source'] for doc in self.documents]
    
    def clear_all(self):
        self.documents = []
        self.vectordb = None

# --- Chat Functions ---
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
    st.session_state.messages = load_chat(user_id)

# --- Main Interface ---
st.title("📄 DocQ&A — Your AI Assistant")

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Upload Documents")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="multi_file_uploader"
    )
    
    use_ocr = st.checkbox("Enable OCR for images", True)
    
    # Display uploaded files info
    if uploaded_files:
        st.write(f"**Files selected:** {len(uploaded_files)}")
        for i, file in enumerate(uploaded_files, 1):
            file_size_mb = file.size / (1024 * 1024)
            st.write(f"{i}. {file.name} ({file_size_mb:.1f} MB)")
    
    # Process files button
    if uploaded_files and st.button("📤 Process Files", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_count = 0
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i) / total_files)
                
                # Extract text based on file type
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                file_bytes = io.BytesIO(uploaded_file.getvalue())
                
                text = ""
                if ext == ".pdf":
                    text = extract_text_from_pdf(file_bytes, use_ocr)
                elif ext == ".docx":
                    text = extract_text_from_docx(file_bytes, use_ocr)
                elif ext == ".txt":
                    text = extract_text_from_txt(file_bytes)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                if text.strip():
                    st.session_state.doc_manager.add_file(uploaded_file.name, text)
                    processed_count += 1
                    st.success(f"✅ {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No text found in {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error with {uploaded_file.name}: {e}")
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        if processed_count > 0:
            st.success(f"🎉 Successfully processed {processed_count} files!")
        else:
            st.error("No files were successfully processed.")
    
    # Document status
    st.divider()
    doc_count = st.session_state.doc_manager.get_document_count()
    if doc_count > 0:
        st.success(f"📚 {doc_count} documents loaded")
        
        # Show document list
        with st.expander("📋 View Documents"):
            for i, doc_name in enumerate(st.session_state.doc_manager.get_document_list(), 1):
                st.write(f"{i}. {doc_name}")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents"):
            st.session_state.doc_manager.clear_all()
            st.session_state.messages = []
            st.success("All documents cleared!")
            st.rerun()
    else:
        st.info("📤 Upload documents to get started")

# --- Chat Interface ---
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.doc_manager.answer_question(prompt)
            st.markdown(response)
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save chat history
    save_chat(user_id, st.session_state.messages)
