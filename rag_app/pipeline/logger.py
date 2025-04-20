# rag_app/pipeline/logger.py
# (Keep the last version that logs relative to os.getcwd() and includes console handler)
import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
print(f"[Logger Setup] Attempting to create/use log directory: {log_path}")

try:
    os.makedirs(log_path, exist_ok=True)
    LOG_FILEPATH = os.path.join(log_path, LOG_FILE)
    print(f"[Logger Setup] Log file path set to: {LOG_FILEPATH}")

    logging.basicConfig(level=logging.INFO,
                        filename=LOG_FILEPATH,
                        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                        )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(console_handler)
except Exception as e:
    print(
        f"[CRITICAL Logger Setup Error] Failed to configure logging: {e}", file=sys.stderr)

logger = logging.getLogger("RAGAppLogger")
if not logger.hasHandlers() and not logging.getLogger().hasHandlers():
    print("[Logger Setup Warning] Logger may not have been configured correctly.", file=sys.stderr)
else:
    logger.info("--- RAGAppLogger Initialized ---")
