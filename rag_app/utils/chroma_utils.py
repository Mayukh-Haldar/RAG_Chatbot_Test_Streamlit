# rag_app/utils/chroma_utils.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- EMBEDDING IMPORT ---
# Import typing tools needed for hints
from typing import Optional, List, Type

try:
    # Import the actual class
    from langchain_nomic import NomicEmbeddings
    # Define the type alias pointing to the real class
    NomicEmbeddingsType = NomicEmbeddings  # Store the type if import succeeds
    print("[Chroma Utils] Using NomicEmbeddings from langchain_nomic package.")
except ImportError:
    print("[Chroma Utils] ERROR: langchain_nomic package not found. Please install it (`pip install langchain-nomic`).")
    # Define NomicEmbeddingsType as None *only* if import fails
    # This variable itself isn't used in hints anymore, but helps logic later
    NomicEmbeddingsType = None
# --- --- --- --- --- --- --- ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import sys
from dotenv import load_dotenv, find_dotenv
from pipeline.exception import CustomException
from pipeline.logger import logger

# --- Global variables (less critical now with caching) ---
# embedding_function = None
# vectorstore = None
# is_ready = False

# --- Define Nomic Configuration ---
NOMIC_MODEL_NAME = os.getenv("NOMIC_MODEL_NAME", "nomic-embed-text-v1.5")
nomic_dimensionality_str = os.getenv("NOMIC_DIMENSIONALITY")
NOMIC_DIMENSIONALITY = int(
    nomic_dimensionality_str) if nomic_dimensionality_str else None
# --- --- --- --- --- --- --- --- ---

# --- Cached Resource: Embedding Function ---


@st.cache_resource(show_spinner="Initializing embedding model...")
# --- CORRECTED TYPE HINT USING STRING FORWARD REFERENCE ---
# Use string
def get_embedding_function(nomic_api_key: str, model: str, dimensionality: Optional[int]) -> Optional['NomicEmbeddings']:
    # --- END CORRECTION ---
    """Initializes and returns the Nomic Embeddings object."""
    if NomicEmbeddingsType is None:  # Check if the import failed using the variable
        logger.error(
            "Cannot get embedding function: NomicEmbeddings class not imported.")
        st.error(
            "Nomic Embeddings library not found. Please install langchain-nomic.")
        return None
    if not nomic_api_key:
        logger.warning(
            "Cannot get embedding function: Nomic API key is missing.")
        return None

    logger.info(
        f"Cache miss or arguments changed. Initializing NomicEmbeddings (Model: {model}, Dim: {dimensionality or 'Default'})...")
    try:
        os.environ["NOMIC_API_KEY"] = nomic_api_key
        # Use the original class name now, guarded by the check above
        embeddings = NomicEmbeddings(
            model=model,
            dimensionality=dimensionality
        )
        logger.info("NomicEmbeddings initialized successfully.")
        return embeddings
    except ValueError as ve:
        logger.error(
            f"Failed to initialize NomicEmbeddings: {ve}. Check dimensionality.", exc_info=True)
        st.error(f"Nomic Embeddings Initialization Error: {ve}")
        return None
    except Exception as emb_e:
        logger.error(
            f"Failed to initialize NomicEmbeddings: {emb_e}. Check NOMIC_API_KEY and model name.", exc_info=True)
        st.error(f"Nomic Embeddings Initialization Error: {emb_e}")
        return None

# --- Cached Resource: Vector Store ---


@st.cache_resource(show_spinner="Connecting to vector store...")
def get_vector_store(_embedding_function: Optional['NomicEmbeddings']) -> Optional[Chroma]:
    if _embedding_function is None:
        logger.error(
            "Cannot initialize vector store: Embedding function is None.")
        return None
    logger.info("Attempting Chroma initialization...")
    try:
        persist_directory = os.path.join(os.getcwd(), 'data', 'chroma_db')
        logger.info(f"Using persist directory: {persist_directory}")
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            logger.info("Created persist directory.")

        logger.info("Calling Chroma constructor...")
        vs = Chroma(persist_directory=persist_directory,
                    embedding_function=_embedding_function)
        logger.info("Chroma constructor finished.")
        # Test if collection exists (might indicate success)
        logger.info(
            f"Chroma collection name: {vs._collection.name if hasattr(vs, '_collection') else 'N/A'}")
        return vs
    except Exception as chroma_e:
        # Log error details
        logger.error(
            f"Failed during Chroma initialization: {chroma_e}", exc_info=True)
        st.error(f"Failed to initialize Vector Store: {chroma_e}")
        return None

# --- Readiness Check Function ---


def is_vectorstore_ready(nomic_api_key: str) -> bool:
    """Checks if the vectorstore can be initialized successfully."""
    emb_func = get_embedding_function(
        nomic_api_key, NOMIC_MODEL_NAME, NOMIC_DIMENSIONALITY)
    vs = get_vector_store(emb_func)
    is_ready_now = emb_func is not None and vs is not None
    logger.debug(
        f"Readiness check result: {is_ready_now} (Embedding: {'OK' if emb_func else 'Failed'}, Vectorstore: {'OK' if vs else 'Failed'})")
    return is_ready_now


# --- Document Processing Functions ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len)

# (Keep load_and_split_document, index_document_to_chroma, delete_doc_from_chroma
# exactly as they were in the previous version - they correctly call the getter functions)


def load_and_split_document(file_path: str) -> List[Document]:
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.html':
            loader = UnstructuredHTMLLoader(file_path)
        else:
            logger.error(f"Unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")
        logger.info(f"Loading document: {file_path}")
        documents = loader.load()
        logger.info(
            f"Splitting {len(documents)} pages/sections from: {file_path}")
        splits = text_splitter.split_documents(documents)
        logger.info(f"Document {file_path} split into {len(splits)} chunks.")
        return splits
    except ValueError as ve:
        logger.error(
            f"Value error during document processing for {file_path}: {ve}")
        raise ve
    except Exception as e:
        error_details = CustomException(e, sys)
        logger.error(
            f"Failed to load/split document {file_path}: {error_details}", exc_info=True)
        st.error(
            f"Failed to process document {os.path.basename(file_path)}: {e}")
        return []


def index_document_to_chroma(file_path: str, file_id: int, nomic_api_key: str) -> bool:
    # 1. GET EMBEDDING FUNCTION / VECTORSTORE (SHOULD BE CACHED & OK NOW)
    emb_func = get_embedding_function(
        nomic_api_key, NOMIC_MODEL_NAME, NOMIC_DIMENSIONALITY)
    vs = get_vector_store(emb_func)
    if not vs:
        # Should not happen if init was ok
        logger.error(
            "Cannot index document: Vectorstore not initialized/retrieved.")
        return False

    try:
        logger.info(
            f"Starting indexing process for file: {file_path}, file_id: {file_id}")

        # 2. LOAD AND SPLIT (POTENTIAL FAILURE POINT A)
        splits = load_and_split_document(file_path)
        if not splits:  # Check if load_and_split had an error and returned []
            logger.error(
                f"No content generated from splitting {file_path}. Indexing aborted.")
            return False  # Return False as indexing didn't happen

        # 3. PREPARE METADATA (Usually safe)
        file_id_str = str(file_id)
        docs_to_add = []
        for split in splits:
            split.metadata = split.metadata or {}
            split.metadata['file_id'] = file_id_str
            docs_to_add.append(split)
        if not docs_to_add:
            logger.warning(
                f"No document splits generated for file_id {file_id_str}. Nothing to index.")
            return False

        logger.info(
            f"Adding {len(docs_to_add)} document chunks with file_id {file_id_str} to Chroma...")

        # 4. ADD DOCUMENTS TO CHROMA (POTENTIAL FAILURE POINT B - API CALL)
        # --->>> THIS IS WHERE THE NOMIC EMBEDDING API CALL HAPPENS <<<---
        vs.add_documents(docs_to_add)
        # --->>> ---------------------------------------------- <<<---
        logger.info(
            f"Successfully added {len(docs_to_add)} chunks for file_id {file_id_str} to Chroma.")
        return True

    except Exception as e:  # Catch errors during load/split or add_documents
        error_details = CustomException(e, sys)
        # Errors here might relate to Nomic API key validity, usage limits, PDF processing etc.
        logger.error(
            f"Error indexing document {file_path} (file_id: {file_id}): {error_details}", exc_info=True)
        st.error(f"Indexing failed: {e}")  # Show specific error in UI
        return False  # Explicitly return False on any exception during the process


def delete_doc_from_chroma(file_id: int, nomic_api_key: str) -> bool:
    emb_func = get_embedding_function(
        nomic_api_key, NOMIC_MODEL_NAME, NOMIC_DIMENSIONALITY)
    vs = get_vector_store(emb_func)
    if not vs:
        logger.error("Cannot delete document: Vectorstore not available.")
        st.error("Vector Store not initialized, cannot delete.")
        return False
    try:
        file_id_str = str(file_id)
        logger.info(
            f"Attempting to delete documents with file_id {file_id_str} from Chroma.")
        if hasattr(vs, '_collection') and vs._collection:
            vs._collection.delete(where={"file_id": file_id_str})
            logger.info(
                f"Executed delete command for file_id {file_id_str} in Chroma.")
            return True
        else:
            logger.error(
                "Cannot delete document: Chroma collection attribute not available in vectorstore instance.")
            st.error(
                "Internal Error: Could not access vector store collection for deletion.")
            return False
    except Exception as e:
        error_details = CustomException(e, sys)
        logger.error(
            f"Error deleting document file_id {file_id} from Chroma: {error_details}", exc_info=True)
        st.error(f"Deletion failed: {e}")
        return False
