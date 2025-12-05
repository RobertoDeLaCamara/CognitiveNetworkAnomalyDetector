"""Logging setup module for the anomaly detection system."""

import logging
import os
import stat
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .config import LOG_FILE, LOG_MAX_SIZE, LOG_BACKUP_COUNT

def _secure_log_file(log_path: str) -> None:
    """Set secure permissions on log file.
    
    Args:
        log_path: Path to the log file
    """
    try:
        if os.path.exists(log_path):
            # Set file permissions to 640 (owner read/write, group read)
            os.chmod(log_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    except (OSError, PermissionError) as e:
        # Don't fail if we can't set permissions, just log it
        print(f"Warning: Could not set secure permissions on log file: {e}")

def _validate_log_path(log_path: str) -> str:
    """Validate and sanitize log file path.
    
    Args:
        log_path: Log file path to validate
        
    Returns:
        Validated log file path
        
    Raises:
        ValueError: If path is invalid or unsafe
    """
    # Convert to Path object for better handling
    path = Path(log_path)
    
    # Check for path traversal attempts
    if '..' in str(path) or str(path).startswith('/'):
        if not str(path).startswith(os.getcwd()):
            raise ValueError(f"Unsafe log path: {log_path}")
    
    # Ensure filename is reasonable
    if len(path.name) > 255:
        raise ValueError("Log filename too long")
    
    # Check for valid characters (basic validation)
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in str(path) for char in invalid_chars):
        raise ValueError(f"Invalid characters in log path: {log_path}")
    
    return str(path)

def setup_logger(name: str = None) -> logging.Logger:
    """Configures the rotating log system with security considerations.
    
    Args:
        name: Logger name (defaults to root logger)
        
    Returns:
        Configured logger instance
        
    Raises:
        ValueError: If log configuration is invalid
        OSError: If log directory cannot be created
    """
    try:
        # Validate log file path
        validated_log_file = _validate_log_path(LOG_FILE)
        
        # Ensure log directory exists with secure permissions
        log_dir = os.path.dirname(validated_log_file) or '.'
        os.makedirs(log_dir, mode=0o750, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Create rotating file handler with size limits
        try:
            handler = RotatingFileHandler(
                validated_log_file, 
                maxBytes=LOG_MAX_SIZE, 
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
        except (OSError, PermissionError) as e:
            # Fallback to console logging if file logging fails
            print(f"Warning: Cannot create log file {validated_log_file}: {e}")
            print("Falling back to console logging")
            handler = logging.StreamHandler()
        
        handler.setLevel(logging.INFO)
        
        # Create secure formatter (avoid logging sensitive data)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Set secure permissions on log file
        if isinstance(handler, RotatingFileHandler):
            _secure_log_file(validated_log_file)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
        
    except Exception as e:
        # Create a basic console logger as fallback
        fallback_logger = logging.getLogger(name or 'anomaly_detector')
        if not fallback_logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            fallback_logger.addHandler(console_handler)
            fallback_logger.setLevel(logging.INFO)
        
        fallback_logger.error(f"Failed to setup file logging: {e}")
        return fallback_logger

# Create default logger instance
logger = setup_logger('anomaly_detector')
