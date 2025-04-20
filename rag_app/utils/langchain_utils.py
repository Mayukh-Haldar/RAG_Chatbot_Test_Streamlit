# rag_app/utils/langchain_utils.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Optional
from langchain_core.runnables import Runnable
import os
# Use absolute import for utils and pipeline packages relative to rag_app root
# Use getter functions
from utils.chroma_utils import get_vector_store, get_embedding_function
from pipeline.logger import logger

# Define a custom exception for initialization errors


class RAGChainInitializationError(Exception):
    """Custom exception for errors during RAG chain setup."""
    pass


# --- Define prompts (can stay at module level) ---
output_parser = StrOutputParser()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Context: {context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
# --- End prompts ---

# --- Cached Resource: RAG Chain ---
# This function now takes API keys to pass down to resource getters


@st.cache_resource(show_spinner="Initializing RAG chain...")
def get_cached_rag_chain(model: str, nomic_api_key: str, groq_api_key: str) -> Optional[Runnable]:
    """
    Initializes and returns the RAG chain using cached resources (embedding function and vector store).
    Requires Nomic and Groq API keys. Returns None if initialization fails.
    """
    logger.info(
        f"Cache miss or args changed. Initializing RAG chain (model: {model})...")

    # --- Pre-checks ---
    if not nomic_api_key:
        logger.error("Cannot initialize RAG chain: Nomic API key missing.")
        st.error("Nomic API key missing. Please provide it in the sidebar.")
        return None
    if not groq_api_key:
        logger.error("Cannot initialize RAG chain: Groq API key missing.")
        st.error("Groq API key missing. Please provide it in the sidebar.")
        return None
    # --- --- --- --- ---

    # --- Get Embedding Function and Vector Store using cached getters ---
    try:
        logger.debug("Retrieving embedding function...")
        # Retrieve Nomic config details (could also pass them as args if needed)
        nomic_model = os.getenv("NOMIC_MODEL_NAME", "nomic-embed-text-v1.5")
        nomic_dim_str = os.getenv("NOMIC_DIMENSIONALITY")
        nomic_dim = int(nomic_dim_str) if nomic_dim_str else None
        embedding_func = get_embedding_function(
            nomic_api_key, nomic_model, nomic_dim)
        if embedding_func is None:
            # Error was already logged/shown by the getter function
            raise RAGChainInitializationError(
                "Embedding function initialization failed.")

        logger.debug("Retrieving vector store...")
        vectorstore_instance = get_vector_store(embedding_func)
        if vectorstore_instance is None:
            # Error was already logged/shown by the getter function
            raise RAGChainInitializationError(
                "Vector store initialization failed.")

        logger.info(
            "Embedding function and vector store retrieved successfully.")

    except Exception as vs_init_err:
        # Catch potential errors from the getter functions themselves
        logger.error(
            f"Failed to retrieve embedding/vector store: {vs_init_err}", exc_info=True)
        # Error likely already shown via st.error in getter
        return None
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    # --- Create retriever from the retrieved vector store ---
    try:
        # Configure retriever settings
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 2})
        logger.info("Retriever created successfully.")
    except Exception as ret_e:
        logger.error(
            f"Failed to create retriever from vectorstore: {ret_e}", exc_info=True)
        st.error(f"Failed to create document retriever: {ret_e}")
        return None
    # --- --- --- --- --- --- --- --- --- --- --- --- ---

    # --- Initialize LLM and Chains ---
    try:
        # Initialize Groq LLM, passing the key explicitly
        llm = ChatGroq(model=model, temperature=0, api_key=groq_api_key)
        logger.info(f"ChatGroq LLM initialized with model: {model}")

        # Create history-aware retriever chain
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt)
        logger.info("History-aware retriever chain created.")

        # Create final question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        logger.info("Question-answer chain created.")

        # Combine them into the final RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)
        logger.info("RAG retrieval chain created successfully.")

        return rag_chain  # Return the final chain object

    except Exception as chain_e:
        logger.error(
            f"Failed to create RAG chain components: {chain_e}", exc_info=True)
        st.error(f"Failed to initialize RAG Chain: {chain_e}")
        return None
    # --- --- --- --- --- --- --- --- ---
