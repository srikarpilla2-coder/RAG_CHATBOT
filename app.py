import streamlit as st
import os
import time
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional

# --- Third-party imports ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================================
# 1. CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E1E1E;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables (local .env or Streamlit secrets)
load_dotenv()

# ============================================================================
# 2. INTERNAL CLASSES (From rag_engine.py)
# ============================================================================

class RateLimiter:
    """Simple rate limiter to prevent API quota exhaustion."""
    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.min_delay = 2
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Enforce rate limiting."""
        current_time = time.time()
        # Remove calls older than 60 seconds
        self.calls = [t for t in self.calls if current_time - t < 60]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = 60 - (current_time - self.calls[0]) + 1
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.calls = []
        
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        self.calls.append(time.time())
        self.last_call_time = time.time()

class ResponseCache:
    """Cache responses to avoid redundant API calls."""
    # Using st.session_state for cache persistence in Streamlit app
    def get(self, query: str, context: str) -> Optional[str]:
        if "response_cache" not in st.session_state:
            st.session_state.response_cache = {}
        key = f"{query}_{context}"
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return st.session_state.response_cache.get(hashed_key)
    
    def set(self, query: str, context: str, response: str):
        if "response_cache" not in st.session_state:
            st.session_state.response_cache = {}
        key = f"{query}_{context}"
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        st.session_state.response_cache[hashed_key] = response

# ============================================================================
# 3. CORE LOGIC (RAG ENGINE ADAPTED FOR STREAMLIT)
# ============================================================================

@st.cache_resource
def get_embeddings():
    """Initialize embeddings - using FREE local HuggingFace (cached)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

class RAGService:
    """Service class to handle RAG operations within Streamlit."""
    def __init__(self):
        self.data_path = "data"
        self.db_path = "faiss_index"
        self.embeddings = get_embeddings()
        self.rate_limiter = RateLimiter(max_calls_per_minute=8)
        self.cache = ResponseCache()

    def ingest_data(self) -> str:
        """Process PDFs and create vector database."""
        # Ensure data directory exists
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return f"Created '{self.data_path}' folder. Please upload PDF files."
        
        pdf_files = list(Path(self.data_path).glob("*.pdf"))
        if not pdf_files:
            return "⚠️ No PDF files found in 'data' folder."
        
        try:
            # Load documents
            loader = DirectoryLoader(
                self.data_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=False,
                use_multithreading=True
            )
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            if len(texts) > 100:
                vectorstore = FAISS.from_documents(texts[:100], self.embeddings)
                for i in range(100, len(texts), 100):
                    batch = texts[i:i + 100]
                    vectorstore.add_documents(batch)
                    time.sleep(0.5) 
            else:
                vectorstore = FAISS.from_documents(texts, self.embeddings)
            
            # Save to disk
            vectorstore.save_local(self.db_path)
            return f"✅ Success! Processed {len(pdf_files)} PDFs and created knowledge base."
            
        except Exception as e:
            return f"❌ Error during ingestion: {str(e)}"

    def get_qa_chain(self, api_key):
        """Setup the QA chain."""
        if not os.path.exists(self.db_path):
            return None, "Vector database not found. Please ingest documents first."
            
        try:
            # Load vector store
            vectorstore = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Initialize LLM
            genai.configure(api_key=api_key)
            
            # Attempt to initialize a working model
            models_to_try = [
                "models/gemini-2.5-flash",
                "models/gemini-2.0-flash",
                "models/gemini-1.5-flash",
                "models/gemini-pro"
            ]
            
            llm = None
            for model_name in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=api_key,
                        temperature=0.3,
                        max_output_tokens=1024
                    )
                    # Simple test
                    llm.invoke("Hi")
                    break 
                except:
                    continue
            
            if not llm:
                return None, "❌ Could not initialize any Gemini model. Check API Key or Quota."

            # Prompt
            template = """You are a helpful AI assistant that answers questions based strictly on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer ONLY using information from the context above
- If the answer is not in the context, say "I don't have enough information in the documents to answer that question."
- Be concise and accurate
- Cite specific details when possible

Answer:"""
            
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            return qa_chain, None

        except Exception as e:
            return None, f"❌ Error setting up chain: {str(e)}"

# Initialize Service
rag_service = RAGService()

# ============================================================================
# 4. UI LOGIC (From ui.py)
# ============================================================================

def handle_local_intent(user_input):
    """Intercepts user input to handle greetings and exits locally."""
    u_in = user_input.lower().strip()
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good evening"]
    if any(u_in.startswith(g) for g in greetings) and len(u_in) < 20:
        return (
            "Hello! 👋\n\n"
            "I am your **AI Document Assistant**. I can read your uploaded PDF documents "
            "and answer questions based strictly on their content.\n\n"
            "**How can I help you today?**"
        )
    exits = ["bye", "exit", "quit", "goodbye"]
    gratitude = ["thank you", "thanks", "thx"]
    
    if any(u_in == e for e in exits):
        return "Thank you for using the system. Have a productive day! 👋"
    if any(u_in.startswith(t) for t in gratitude):
        return "You're very welcome! Feel free to ask more questions."
    return None

# ============================================================================
# 5. SIDEBAR & MAIN INTERFACE
# ============================================================================

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.markdown("### Control Panel")
    
    # API Key
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        st.warning("⚠️ API Key Missing")
        user_key = st.text_input("Enter Google API Key", type="password")
        if user_key:
            api_key = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
            st.success("Key Set!")
    else:
        st.success("✅ API Key Active")

    st.markdown("---")
    
    # Document Management
    st.markdown("**Document Management**")
    
    # File Uploader for Streamlit Cloud
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if not os.path.exists("data"):
            os.makedirs("data")
        for uploaded_file in uploaded_files:
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files to 'data/' folder.")
    
    # Ingestion Button
    if st.button("🔄 Process / Ingest Documents", type="primary"):
        with st.spinner("Analyzing documents..."):
            result_msg = rag_service.ingest_data()
            if "Success" in result_msg:
                st.success(result_msg)
                st.toast("Knowledge Base Updated", icon="✅")
            else:
                st.error(result_msg)

    st.markdown("---")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("v1.0.0 | Enterprise Edition")


# Main Chat Interface
st.markdown('<p class="main-header">📄 Enterprise RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Chat with your internal documents securely and efficiently.</p>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I'm ready to analyze your documents. What would you like to know?"
    }]

# Display History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle Input
if prompt := st.chat_input("Type your question here..."):
    
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 2. Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        # Local Intent Check
        local_response = handle_local_intent(prompt)
        
        if local_response:
            st.markdown(local_response)
            st.session_state.messages.append({"role": "assistant", "content": local_response})
        else:
            # RAG Query
            if not api_key:
                st.error("Please provide a Google API Key in the sidebar.")
            else:
                with st.spinner("Analyzing knowledge base..."):
                    # Get Chain
                    qa_chain, err = rag_service.get_qa_chain(api_key)
                    
                    if err:
                        st.error(err)
                    else:
                        try:
                            # Rate limit check could go here if needed
                            rag_service.rate_limiter.wait_if_needed()
                            
                            # Execute Chain
                            res = qa_chain.invoke({"query": prompt})
                            answer = res.get("result", "No answer found.")
                            source_docs = res.get("source_documents", [])
                            
                            # Format Response
                            full_resp = answer
                            unique_sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
                            if unique_sources:
                                clean_sources = [os.path.basename(s) for s in unique_sources]
                                full_resp += f"\n\n---\n**📚 References:**\n" + "\n".join([f"- `{s}`" for s in clean_sources])
                            
                            st.markdown(full_resp)
                            st.session_state.messages.append({"role": "assistant", "content": full_resp})
                            
                        except Exception as e:
                            if "429" in str(e):
                                st.error("⚠️ API Quota Exceeded. Please try again later or use a new key.")
                            else:
                                st.error(f"Error processing request: {str(e)}")
