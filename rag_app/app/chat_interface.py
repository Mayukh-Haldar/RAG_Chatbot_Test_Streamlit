# rag_app/app/chat_interface.py

import streamlit as st
import uuid  # Import uuid for session ID generation
# --- Use ABSOLUTE imports for backend logic ---
from utils.langchain_utils import get_cached_rag_chain, RAGChainInitializationError
from utils.db_utils import get_chat_history, insert_application_logs
from pipeline.logger import logger
# --- --- --- --- --- --- --- --- --- --- ---


def display_chat_interface():
    logger.debug("Displaying chat interface")
    # Ensure necessary keys are in session state
    if "nomic_api_key" not in st.session_state:
        st.session_state.nomic_api_key = ""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = "llama-3.1-8b-instant"
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(
            uuid.uuid4())  # Generate initial session ID
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Enter your query here..."):
        logger.info(f"User input received: {prompt}")
        # Check for API Keys first
        if not st.session_state.nomic_api_key or not st.session_state.groq_api_key:
            st.warning(
                "Please enter both Nomic and Groq API keys in the sidebar to enable chat.")
            return  # Stop processing if keys are missing

        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get cached RAG chain (pass keys from session state)
        selected_model = st.session_state.model
        session_id = st.session_state.session_id

        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Create placeholder for streaming/final answer
            with st.spinner("Thinking..."):
                try:
                    # Get RAG chain (uses cache)
                    rag_chain = get_cached_rag_chain(
                        selected_model,
                        st.session_state.nomic_api_key,
                        st.session_state.groq_api_key
                    )

                    if rag_chain:
                        logger.info(
                            f"Invoking RAG chain for session {session_id}")
                        # Get chat history for the chain
                        chat_history = get_chat_history(session_id)

                        # --- Invoke RAG chain DIRECTLY ---
                        response = rag_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history
                        })
                        answer = response.get(
                            'answer', "Sorry, I couldn't extract an answer.")
                        # --- --- --- --- --- --- --- ---

                        logger.info(
                            f"RAG chain invocation successful for session {session_id}")
                        # Log interaction to DB
                        insert_application_logs(
                            session_id, prompt, answer, selected_model)

                        # Update session state and display answer
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer})
                        message_placeholder.markdown(
                            answer)  # Update placeholder

                        # Optional: Details expander
                        # with st.expander("Response Details"): st.json(response)

                    else:
                        # Error handled by get_cached_rag_chain via st.error
                        logger.error("RAG chain is None, cannot process chat.")
                        message_placeholder.error(
                            "The RAG chat chain failed to initialize. Please check the logs or API keys.")

                except RAGChainInitializationError as init_err:
                    logger.error(
                        f"RAG Chain Init Error during chat: {init_err}")
                    message_placeholder.error(
                        f"Could not initialize the chat service: {init_err}")
                except Exception as e:
                    logger.error(
                        f"Error during chat processing or RAG chain invocation: {e}", exc_info=True)
                    # Consider using CustomException if more detail is needed
                    message_placeholder.error(
                        f"An unexpected error occurred: {e}")
