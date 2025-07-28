import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import streamlit as st

# Set keys to environment before initializing any clients
hf_token = st.session_state.get("HUGGINGFACE_API_KEY", "")
groq_token = st.session_state.get("GROQ_API_KEY", "")

if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
if groq_token:
    os.environ["GROQ_API_KEY"] = groq_token

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(model="llama3-70b-8192")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  # ✅ explicitly allow safe local file
)

# Set up retriever to return top 10 most relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
