# rag_app/app/sidebar.py

import streamlit as st
import os
import tempfile  # Use tempfile for uploads
import shutil
# --- Use ABSOLUTE imports for backend logic ---
from utils.chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from utils.db_utils import insert_document_record, delete_document_record, get_all_documents
from pipeline.logger import logger
# --- --- --- --- --- --- --- --- --- --- ---


def display_sidebar():
    st.sidebar.header("API Keys")
    st.sidebar.caption("Needed for Embeddings (Nomic) and Chat (Groq)")

    # Initialize keys in session state if not present
    if "nomic_api_key" not in st.session_state:
        st.session_state.nomic_api_key = ""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""

    # Get keys from user input, store in session state
    nomic_key_input = st.sidebar.text_input(
        "Nomic API Key",
        type="password",
        key="nomic_key_input_widget",  # Use unique key for widget
        value=st.session_state.nomic_api_key  # Pre-fill if already entered
    )
    groq_key_input = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        key="groq_key_input_widget",
        value=st.session_state.groq_api_key
    )

    # Update session state when input changes
    st.session_state.nomic_api_key = nomic_key_input
    st.session_state.groq_api_key = groq_key_input

    # Display warning if keys are missing
    if not st.session_state.nomic_api_key:
        st.sidebar.warning(
            "Nomic API Key is required for document uploads and chat.")
    if not st.session_state.groq_api_key:
        st.sidebar.warning("Groq API Key is required for chat.")

    st.sidebar.divider()

    st.sidebar.header("Configuration")
    # Model selection
    model_options = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    selected_model = st.sidebar.selectbox("Select Chat Model",
                                          options=model_options,
                                          key="model_selection")
    # Store selected model in session state
    st.session_state.model = selected_model

    st.sidebar.divider()

    # Document upload
    st.sidebar.header("Document Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document", type=["pdf", "docx", "html"])

    if uploaded_file is not None and st.sidebar.button("Upload Document", key="upload_button"):
        if not st.session_state.nomic_api_key:
            st.sidebar.error(
                "Please enter the Nomic API Key before uploading.")
        else:
            with st.spinner("Processing and indexing document..."):
                # 1. Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)
                    temp_file_path = tmp_file.name
                    logger.info(
                        f"File '{uploaded_file.name}' saved temporarily to {temp_file_path}")

                file_id = None
                indexing_success = False
                try:
                    # 2. Insert DB record
                    file_id = insert_document_record(uploaded_file.name)
                    if file_id:
                        # 3. Index document (pass API key from session state)
                        indexing_success = index_document_to_chroma(
                            temp_file_path,
                            file_id,
                            st.session_state.nomic_api_key
                        )
                    else:
                        st.sidebar.error(
                            "Failed to get File ID from database.")

                except ValueError as ve:  # Handle duplicate filename error from db_utils
                    st.sidebar.error(f"Upload failed: {ve}")
                except Exception as e:
                    logger.error(
                        f"Error during upload/indexing process: {e}", exc_info=True)
                    st.sidebar.error(
                        f"An unexpected error occurred during upload: {e}")
                finally:
                    # 4. Clean up temporary file
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logger.info(
                            f"Removed temporary file: {temp_file_path}")

                # 5. Handle results
                if file_id and indexing_success:
                    st.sidebar.success(
                        f"File '{uploaded_file.name}' indexed successfully (ID: {file_id}).")
                    # Optionally clear file uploader state here if needed (can be tricky)
                elif file_id and not indexing_success:
                    st.sidebar.error(
                        f"File '{uploaded_file.name}' uploaded but indexing failed. Rolling back DB entry.")
                    # Attempt rollback (db_utils function logs errors)
                    delete_document_record(file_id)
                # else: error already shown or handled

        # Refresh documents list after any upload attempt
        with st.spinner("Refreshing document list..."):
            st.session_state.documents = get_all_documents()

    st.sidebar.divider()

    # List and delete documents
    st.sidebar.subheader("Indexed Documents")
    if st.sidebar.button("Refresh Document List", key="refresh_docs_button"):
        with st.spinner("Refreshing document list..."):
            st.session_state.documents = get_all_documents()

    # Initialize documents if not present
    if "documents" not in st.session_state:
        st.session_state.documents = get_all_documents()  # Load initially

    if st.session_state.documents is None:  # Check if loading failed
        st.sidebar.warning("Could not retrieve document list from database.")
        st.session_state.documents = []  # Set to empty list to avoid errors

    if st.session_state.documents:
        doc_options = {f"{doc['filename']} (ID: {doc['id']})": doc['id']
                       for doc in st.session_state.documents}
        if doc_options:
            selected_doc_label = st.sidebar.selectbox("Select document to delete", options=list(
                doc_options.keys()), key="delete_doc_select")
            selected_file_id = doc_options[selected_doc_label]

            if st.sidebar.button("Delete Selected Document", key="delete_doc_button"):
                if not st.session_state.nomic_api_key:
                    st.sidebar.error("Nomic API Key needed for deletion.")
                else:
                    with st.spinner(f"Deleting document ID {selected_file_id}..."):
                        # Call deletion functions directly
                        chroma_deleted = delete_doc_from_chroma(
                            selected_file_id, st.session_state.nomic_api_key)
                        db_deleted = False
                        if chroma_deleted:  # Only delete from DB if Chroma delete seemed successful
                            db_deleted = delete_document_record(
                                selected_file_id)
                            if db_deleted:
                                st.sidebar.success(
                                    f"Document ID {selected_file_id} deleted successfully.")
                            else:
                                st.sidebar.error(
                                    f"Deleted from vector store, but failed to delete DB record for ID {selected_file_id}.")
                        else:
                            st.sidebar.error(
                                f"Failed to delete document ID {selected_file_id} from vector store.")

                    # Refresh list after delete attempt
                    st.session_state.documents = get_all_documents()
        else:
            st.sidebar.write("No documents available for deletion.")
    else:
        st.sidebar.write("No documents indexed yet.")
