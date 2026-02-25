import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure LlamaIndex to use Gemini LLM and local HuggingFace embeddings
Settings.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def load_index():
    """Load documents from the data directory and build a vector index."""
    documents = SimpleDirectoryReader("/Users/julienlundgren/PycharmProjects/Bab/data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


def query_chatbot(user_query, index):
    """Query the RAG index and return a response."""
    query_engine = index.as_query_engine()
    response = query_engine.query(user_query)
    return str(response)


# --- Streamlit UI ---
st.title("📚 Babson Student Handbook Chatbot")
st.write("Ask me anything about the Babson Student Handbook!")

# Load the index (cached so it only runs once)
with st.spinner("Loading handbook data... this may take a moment the first time."):
    index = load_index()

# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the student handbook..."):
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_chatbot(prompt, index)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
