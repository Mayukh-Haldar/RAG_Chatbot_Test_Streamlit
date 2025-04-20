# rag_app/utils/db_utils.py
import sqlite3
from datetime import datetime
import os
import sys
from pipeline.logger import logger  # Absolute import from package root

DB_DIR = os.path.join(os.getcwd(), "data")
DB_NAME = "rag_app.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)

# Flag to ensure initialization runs only once
_db_initialized = False


def _initialize_database():
    """Creates the database directory and tables if they don't exist."""
    global _db_initialized
    if _db_initialized:
        logger.debug("Database already initialized, skipping.")
        return
    try:
        logger.info(f"Ensuring database directory exists: {DB_DIR}")
        os.makedirs(DB_DIR, exist_ok=True)
        logger.info(f"Database path set to: {DB_PATH}")

        with get_db_connection() as conn:  # Use the connection function
            logger.info(
                "Initializing database tables (if they don't exist)...")

            # --- EXECUTE CORRECT SQL ---
            conn.execute('''CREATE TABLE IF NOT EXISTS application_logs (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id TEXT,
                                user_query TEXT,
                                gpt_response TEXT,
                                model TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                           )''')
            logger.debug(
                "Executed CREATE TABLE IF NOT EXISTS for application_logs.")

            conn.execute('''CREATE TABLE IF NOT EXISTS document_store (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                filename TEXT UNIQUE,
                                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                           )''')
            logger.debug(
                "Executed CREATE TABLE IF NOT EXISTS for document_store.")
            # --- END CORRECT SQL ---

            logger.info("Database tables initialization complete.")
            _db_initialized = True  # Mark as initialized ONLY if successful

    except sqlite3.Error as e:
        logger.critical(
            f"Database initialization failed during SQL execution: {e}", exc_info=True)
        # Don't set _db_initialized = True if it fails
        raise RuntimeError(
            f"Failed to initialize database tables at {DB_PATH}") from e
    except OSError as e:
        logger.critical(
            f"Failed to create database directory {DB_DIR}: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to create database directory {DB_DIR}") from e
    except Exception as e:
        logger.critical(
            f"Unexpected error during database initialization: {e}", exc_info=True)
        raise RuntimeError(
            "Unexpected error during database initialization") from e


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn
    except sqlite3.Error as e:
        logger.error(
            f"Failed to connect to database at {DB_PATH}: {e}", exc_info=True)
        raise ConnectionError(f"Could not connect to database: {e}") from e

# --- Keep all other functions ---
# insert_application_logs, get_chat_history, insert_document_record,
# delete_document_record, get_all_documents
# Ensure they use the logger correctly


def insert_application_logs(session_id, user_query, gpt_response, model):
    sql = 'INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)'
    try:
        with get_db_connection() as conn:
            conn.execute(sql, (session_id, user_query, gpt_response, model))
        logger.debug(f"Inserted log for session_id: {session_id}")
    except sqlite3.Error as e:
        logger.error(f"Failed to insert application log: {e}", exc_info=True)


def get_chat_history(session_id):
    messages = []
    sql = 'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at'
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (session_id,))
            for row in cursor.fetchall():
                user_query = row['user_query'] if 'user_query' in row.keys(
                ) else None
                gpt_response = row['gpt_response'] if 'gpt_response' in row.keys(
                ) else None
                if user_query:
                    # Langchain expects user/ai or human/ai
                    messages.append({"role": "user", "content": user_query})
                if gpt_response:
                    # Use assistant role for AI
                    messages.append(
                        {"role": "assistant", "content": gpt_response})
    except sqlite3.Error as e:
        logger.error(
            f"Failed to get chat history for session {session_id}: {e}", exc_info=True)
    return messages


def insert_document_record(filename):
    sql = 'INSERT INTO document_store (filename) VALUES (?)'
    file_id = None
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (filename,))
            file_id = cursor.lastrowid
            logger.info(
                f"Inserted document record '{filename}' with ID: {file_id}")
    except sqlite3.IntegrityError as e:
        logger.warning(
            f"Failed to insert document record for '{filename}', possibly duplicate: {e}")
        raise ValueError(f"Document '{filename}' might already exist.") from e
    except sqlite3.Error as e:
        logger.error(
            f"Failed to insert document record for '{filename}': {e}", exc_info=True)
        raise RuntimeError(
            f"Database error inserting document record for {filename}") from e
    return file_id


def delete_document_record(file_id):
    sql = 'DELETE FROM document_store WHERE id = ?'
    success = False
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_id,))
            if cursor.rowcount > 0:
                logger.info(
                    f"Successfully deleted document record with ID: {file_id}")
                success = True
            else:
                logger.warning(
                    f"Attempted to delete document record ID {file_id}, but it was not found.")
                success = True
    except sqlite3.Error as e:
        logger.error(
            f"Failed to delete document record ID {file_id}: {e}", exc_info=True)
        success = False
    return success


def get_all_documents():
    documents_data = []
    sql = 'SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC'
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            documents_data = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"Retrieved {len(documents_data)} documents from DB.")
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve all documents: {e}", exc_info=True)
    return documents_data

# --- Explicit Initialization ---
# Call this ONCE at the start of your streamlit_app.py


def ensure_db_initialized():
    _initialize_database()
