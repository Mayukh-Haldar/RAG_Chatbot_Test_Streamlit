# rag_app/app/streamlit_app.py

# --- SQLite Patch for ChromaDB ---
# This MUST be run before chromadb is imported.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("[SQLite Patch] Successfully patched sqlite3 with pysqlite3.")
except ImportError:
    print("[SQLite Patch] pysqlite3-binary not found, defaulting to system sqlite3. ChromaDB might fail.")
except Exception as e:
    print(f"[SQLite Patch] Error patching sqlite3: {e}")
# --- End SQLite Patch ---

import streamlit as st
import sys
import os
import uuid

# --- Path Modification Workaround (if needed, test without first) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
rag_app_dir = os.path.dirname(script_dir)
if rag_app_dir not in sys.path:
    print(
        f"[streamlit_app.py] Adding rag_app directory to sys.path: {rag_app_dir}")
    sys.path.insert(0, rag_app_dir)
# --- End Path Modification ---

# --- Import core components ---
try:
    from app.sidebar import display_sidebar
    from app.chat_interface import display_chat_interface
    from utils.db_utils import ensure_db_initialized  # Import DB initializer
    from pipeline.logger import logger  # Import logger
    print("[streamlit_app.py] Successfully imported components.")
except ImportError as e:
    # This error should ideally not happen if running 'streamlit run rag_app/app/streamlit_app.py' from root
    print(
        f"[streamlit_app.py] ERROR: Failed to import components: {e}", file=sys.stderr)
    st.error(
        f"Critical component import failed: {e}. Application cannot start.")
    sys.exit(1)

# --- Initialize Database ---
try:
    logger.info("streamlit_app.py: Ensuring database is initialized...")
    ensure_db_initialized()  # Call the DB init function once
    logger.info("streamlit_app.py: Database initialization check complete.")
except Exception as e:
    logger.critical(
        f"streamlit_app.py: Database initialization failed: {e}", exc_info=True)
    st.error(f"Fatal Error: Could not initialize database: {e}")
    sys.exit(1)
# --- --- --- --- --- --- ---

# === Streamlit App Logic ===
st.set_page_config(page_title="Langchain RAG Chatbot", layout="wide")
st.title("Langchain RAG Chatbot")

# Session state initialization (simple checks, detailed init moved to components)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Please enter API Keys in the sidebar."}]
if "session_id" not in st.session_state:
    # Need uuid imported if generating here
    st.session_state.session_id = str(uuid.uuid4())

# Display components (assuming imports succeeded)
display_sidebar()
display_chat_interface()

logger.info("Streamlit app execution finished.")
# === END Streamlit App Logic ===

# No main() function needed.
