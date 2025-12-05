import logging
import os
from logging.handlers import RotatingFileHandler
from .config import LOG_FILE, LOG_MAX_SIZE, LOG_BACKUP_COUNT

def setup_logger():
    """Configures the rotating log system."""
    # Ensure log directory exists
    log_dir = os.path.dirname(LOG_FILE) or '.'
    os.makedirs(log_dir, exist_ok=True)
    
    handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()
