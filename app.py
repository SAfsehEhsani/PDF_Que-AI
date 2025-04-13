

import streamlit as st
from dotenv import load_dotenv
import os
import time
import hashlib

# Langchain specific imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --- Configuration ---
VECTORSTORE_BASE_FOLDER = "faiss_vectorstores"
TEMP_UPLOAD_FOLDER = "temp_pdf_uploads"

# --- Available Groq Models ---
# You can update this list based on models available on GroqCloud
AVAILABLE_GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]

# --- Helper Functions with Caching ---

@st.cache_resource
def get_embeddings():
    """Loads the HuggingFace embedding model ONCE."""
    print("\n--- Loading Embedding Model ---")
    st.write("Loading embedding model...")
    model_name = "all-MiniLM-L6-v2"
    device = 'cpu' # Change to 'cuda' if GPU available
    print(f"Using device: {device}")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        print("--- Embedding Model Loaded ---\n")
        st.write("Embedding model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Fatal Error loading embedding model: {e}")
        print(f"Fatal Error loading embedding model: {e}")
        st.stop()

def get_file_hash(file_path):
    """Calculates SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192): hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        st.error(f"File not found for hashing: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error hashing file {file_path}: {e}")
        return None

def process_pdf_and_get_vectorstore(pdf_path, embeddings):
    """Processes PDF, creates/loads vector store based on PDF content hash."""
    if not pdf_path or not os.path.exists(pdf_path):
        st.error(f"Invalid PDF path: {pdf_path}")
        return None

    st.write(f"Processing: {os.path.basename(pdf_path)}")
    file_hash = get_file_hash(pdf_path)
    if not file_hash: return None

    vectorstore_folder_path = os.path.join(VECTORSTORE_BASE_FOLDER, file_hash)

    if os.path.exists(vectorstore_folder_path):
        try:
            print(f"Cache hit! Loading vector store for hash {file_hash[:8]}...")
            with st.spinner("Loading pre-processed data..."):
                vectorstore = FAISS.load_local(vectorstore_folder_path, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded from cache.")
            return vectorstore
        except Exception as e:
            st.warning(f"Failed loading cache: {e}. Recreating...")
            print(f"Warning: Cache load failed {vectorstore_folder_path}: {e}")

    st.write("First time processing this PDF content...")
    print(f"Cache miss. Processing PDF: {pdf_path}")

    try:
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
        if not pages:
            st.error("Could not load pages.")
            return None
        print(f"Loaded {len(pages)} pages.")

        with st.spinner("Splitting text..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)
        if not chunks:
            st.error("Could not split PDF.")
            return None
        print(f"Split into {len(chunks)} chunks.")

        st.write(f"Calculating embeddings ({len(chunks)} chunks)...")
        start_time = time.time()
        with st.spinner(f"Calculating text embeddings..."):
            print("Starting FAISS index creation...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
        end_time = time.time()
        print(f"FAISS index creation took {end_time - start_time:.2f} secs.")
        st.write(f"Embeddings calculated in {end_time - start_time:.2f} secs.")

        with st.spinner("Saving processed data..."):
            if not os.path.exists(VECTORSTORE_BASE_FOLDER): os.makedirs(VECTORSTORE_BASE_FOLDER)
            vectorstore.save_local(vectorstore_folder_path)
        print(f"Vector store saved to {vectorstore_folder_path}")
        st.write("Processed data saved.")
        return vectorstore

    except Exception as e:
        st.error(f"Error during PDF processing: {e}")
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

# --- RAG Chain Creation (No change needed here for model selection) ---
@st.cache_resource(show_spinner="Setting up the Chatbot...")
def create_rag_chain(_vectorstore, _llm):
    """Creates the RetrievalQA chain."""
    print("Creating RAG chain...")
    if _vectorstore is None or _llm is None:
        st.error("Cannot create RAG chain: Dependencies missing.")
        return None
    try:
        retriever = _vectorstore.as_retriever(search_kwargs={'k': 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )
        print("RAG chain created.")
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        print(f"Error creating RAG chain: {e}")
        return None

# --- Main Streamlit App Logic ---

st.set_page_config(page_title="History Chatbot", layout="wide")
st.title("⚡️ DOC AI ChatBot")
st.caption("Ask questions about the Anything covered in the provided PDF.")

# Load API Key Securely
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🔴 Groq API key not found in .env file.")
    st.stop()

# Load Embedding Model (Cached)
embeddings = get_embeddings()
if not embeddings: st.stop()

# --- Sidebar Setup (PDF Upload and Model Selection) ---
st.sidebar.header("1. Configuration")

# PDF Upload
uploaded_file = st.sidebar.file_uploader("Upload History PDF", type="pdf", key="pdf_uploader")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Choose Groq Model",
    options=AVAILABLE_GROQ_MODELS,
    index=0, # Default to the first model in the list
    key="groq_model_select"
)

# --- State Management ---
# Initialize state variables if they don't exist
default_state = {
    "pdf_path": None,
    "processing_complete": False,
    "current_pdf_hash": None,
    "qa_chain": None,
    "llm": None,
    "messages": [],
    "current_model_name": None # Track which model was used for processing
}
for key, default_value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Ensure temp directory exists
if not os.path.exists(TEMP_UPLOAD_FOLDER): os.makedirs(TEMP_UPLOAD_FOLDER)

# --- Handle PDF Upload ---
current_selection_info = "Upload a PDF to get started."
if uploaded_file is not None:
    temp_pdf_path = os.path.join(TEMP_UPLOAD_FOLDER, uploaded_file.name)
    try:
        with open(temp_pdf_path, "wb") as f: f.write(uploaded_file.getbuffer())
        new_hash = get_file_hash(temp_pdf_path)
        current_selection_info = f"Selected: {uploaded_file.name}"

        if new_hash and new_hash != st.session_state.current_pdf_hash:
            # Reset state if PDF changes
            print(f"New PDF selected: {uploaded_file.name}")
            st.session_state.pdf_path = temp_pdf_path
            st.session_state.processing_complete = False
            st.session_state.current_pdf_hash = new_hash
            st.session_state.messages = []
            st.session_state.qa_chain = None
            st.session_state.llm = None
            st.session_state.current_model_name = None
        elif new_hash and not st.session_state.processing_complete:
             st.session_state.pdf_path = temp_pdf_path # Update path if same file re-uploaded before processing
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        st.session_state.pdf_path = None
        current_selection_info = "Error with uploaded file."

st.sidebar.info(current_selection_info)

# --- Handle Model Change After Processing ---
# If processing is done BUT the selected model is different from the one used, reset.
if st.session_state.processing_complete and selected_model_name != st.session_state.current_model_name:
    st.sidebar.warning(f"Model changed to {selected_model_name}. Please re-process the PDF.")
    st.session_state.processing_complete = False
    st.session_state.qa_chain = None
    st.session_state.llm = None
    st.session_state.messages = [] # Clear chat on re-process needed

# --- Processing Trigger ---
st.sidebar.header("2. Process & Chat")
process_button_disabled = st.session_state.pdf_path is None or st.session_state.processing_complete

if st.sidebar.button("Process PDF and Start Chat", key="process_button", disabled=process_button_disabled):
    if st.session_state.pdf_path:
        st.session_state.processing_complete = False
        st.session_state.qa_chain = None
        st.session_state.llm = None
        st.session_state.messages = [] # Clear messages on process start

        vectorstore = process_pdf_and_get_vectorstore(st.session_state.pdf_path, embeddings)

        if vectorstore:
            try:
                # Use the selected model from the sidebar stored in session state implicitly by widget key
                st.session_state.current_model_name = selected_model_name # Store model used for processing
                print(f"Initializing LLM: Groq - {st.session_state.current_model_name}")
                st.session_state.llm = ChatGroq(
                    model_name=st.session_state.current_model_name, # Use stored name
                    groq_api_key=groq_api_key,
                    temperature=0.7
                )

                st.session_state.qa_chain = create_rag_chain(vectorstore, st.session_state.llm)

                if st.session_state.qa_chain:
                    st.session_state.processing_complete = True
                    st.sidebar.success("✅ Ready to Chat!")
                    print("Processing complete. Chatbot ready.")
                    st.rerun() # Refresh UI state cleanly
                else:
                    st.sidebar.error("🔴 Failed to create RAG chain.")
                    print("Error: Failed to create RAG chain.")
            except Exception as e:
                st.sidebar.error(f"🔴 Error initializing LLM/Chain: {e}")
                print(f"LLM/Chain Init Error: {e}")
                st.session_state.llm = None
                st.session_state.qa_chain = None
                st.session_state.current_model_name = None # Reset if failed
        else:
            st.sidebar.error("🔴 PDF processing failed.")
            print("Error: PDF processing returned None.")
            st.session_state.qa_chain = None
            st.session_state.llm = None
            st.session_state.current_model_name = None # Reset if failed
    else:
        st.sidebar.warning("Please upload a PDF first.")

# --- Display Chatbot Status ---
if st.session_state.processing_complete:
    st.sidebar.success(f"✅ Ready! (Model: {st.session_state.current_model_name})")

# --- Chat Interface ---
st.divider()

# Only show chat if processing is done, chain exists, and we have a PDF path
if st.session_state.processing_complete and st.session_state.qa_chain and st.session_state.pdf_path:
    st.header(f"💬 Chat about: {os.path.basename(st.session_state.pdf_path)}")

    # Initialize chat message if needed
    if not st.session_state.messages:
        # ** FIX: Use the model name stored in session state **
        initial_message = f"PDF '{os.path.basename(st.session_state.pdf_path)}' processed using Groq ({st.session_state.current_model_name}). Ask me anything!"
        st.session_state.messages = [{"role": "assistant", "content": initial_message}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Thinking indicator
        thinking_message = st.chat_message("assistant").empty()
        thinking_message.markdown("🤔 Thinking...")

        try:
            start_time = time.time()
            qa_chain = st.session_state.qa_chain
            response = qa_chain.invoke({"query": prompt})
            answer = response.get('result', "Sorry, couldn't find an answer.")
            source_docs = response.get('source_documents')
            end_time = time.time()
            print(f"Groq LLM call took {end_time - start_time:.2f} secs.")

            # Update thinking message with actual answer
            thinking_message.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display sources (optional, only if answer displayed successfully)
            if source_docs:
                 with st.expander("View Sources"):
                    for i, doc in enumerate(source_docs):
                        page_num_raw = doc.metadata.get('page', 'N/A')
                        page_num = int(page_num_raw) + 1 if isinstance(page_num_raw, int) else 'N/A'
                        st.markdown(f"**Source {i+1} (Page: {page_num})**")
                        st.caption(f"> {doc.page_content[:300].strip()}...")
                        st.divider()

        except Exception as e:
            error_message = f"An error occurred: {e}"
            thinking_message.error(error_message) # Show error in UI
            print(f"Chat Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, error occurred: {e}"})

# Display guidance messages if chat is not ready
elif st.session_state.pdf_path and not st.session_state.processing_complete:
    st.info("☝️ PDF selected. Please click 'Process PDF and Start Chat' in the sidebar.")
elif not st.session_state.pdf_path:
     st.info("Please upload a PDF using the sidebar to begin.")