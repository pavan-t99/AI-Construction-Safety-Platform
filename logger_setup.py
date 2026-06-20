# logger_setup.py
import logging
import os
from datetime import datetime

def get_logger(camera_id: str) -> logging.Logger:
    """
    Returns a logger for a specific camera.
    Writes to both: terminal (INFO) and a log file (DEBUG).
    One log file per camera per day.
    """
    # Create logs folder
    log_dir = os.path.join("data", camera_id, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log file named by date — auto-rotates daily
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{camera_id}_{today}.log")

    # Get or create logger (named by camera_id — avoids duplicate handlers)
    logger = logging.getLogger(camera_id)

    # Only add handlers once — prevents duplicate log lines on restart
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # === FILE HANDLER — captures everything including DEBUG ===
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)

        # === TERMINAL HANDLER — only INFO and above ===
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger