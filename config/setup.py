import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from chromadb.config import Settings

# Set keys to environment before initializing any clients
hf_token = st.session_state.get("HUGGINGFACE_API_KEY", "")
groq_token = st.session_state.get("GROQ_API_KEY", "")

if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
if groq_token:
    os.environ["GROQ_API_KEY"] = groq_token

# Now safe to init the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM model (Groq)
llm = ChatGroq(model="llama3-70b-8192")

# Chroma vector DB setup
chroma_settings = Settings(
    persist_directory="chroma_legal_db",
    anonymized_telemetry=False
)

vectorstore = Chroma(
    persist_directory="chroma_legal_db",
    embedding_function=embedding_model,
    client_settings=chroma_settings
)

retriever = vectorstore.as_retriever()
