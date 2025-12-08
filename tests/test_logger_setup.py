"""Tests for logger_setup module."""

import os
import stat
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

from src.logger_setup import (
    setup_logger,
    _validate_log_path,
    _secure_log_file
)


class TestValidateLogPath:
    """Test path validation logic."""
    
    def test_valid_simple_path(self):
        """Test validation of a simple valid path."""
        result = _validate_log_path("app.log")
        assert result == "app.log"
    
    def test_valid_relative_path(self):
        """Test validation of relative path."""
        result = _validate_log_path("logs/app.log")
        assert result == "logs/app.log"
    
    def test_valid_absolute_path_in_cwd(self):
        """Test validation of absolute path within current directory."""
        cwd_path = os.path.join(os.getcwd(), "logs", "app.log")
        result = _validate_log_path(cwd_path)
        assert result == cwd_path
    
    def test_path_traversal_outside_cwd_raises_error(self):
        """Test that path traversal outside cwd raises ValueError."""
        with pytest.raises(ValueError, match="Unsafe log path"):
            _validate_log_path("../../etc/passwd")
    
    def test_long_filename_raises_error(self):
        """Test that overly long filename raises ValueError."""
        long_name = "a" * 256 + ".log"
        with pytest.raises(ValueError, match="Log filename too long"):
            _validate_log_path(long_name)
    
    def test_invalid_characters_raise_error(self):
        """Test that invalid characters raise ValueError."""
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            with pytest.raises(ValueError, match="Invalid characters"):
                _validate_log_path(f"log{char}file.log")


class TestSecureLogFile:
    """Test secure file permission setting."""
    
    def test_set_permissions_on_existing_file(self, tmp_path):
        """Test setting permissions on existing log file."""
        log_file = tmp_path / "test.log"
        log_file.touch()
        
        _secure_log_file(str(log_file))
        
        # Check permissions (owner read/write, group read)
        file_stat = os.stat(log_file)
        permissions = stat.filemode(file_stat.st_mode)
        # Should be -rw-r----- or similar
        assert file_stat.st_mode & stat.S_IRUSR
        assert file_stat.st_mode & stat.S_IWUSR
    
    def test_nonexistent_file_no_error(self, tmp_path):
        """Test that nonexistent file doesn't raise error."""
        log_file = tmp_path / "nonexistent.log"
        # Should not raise
        _secure_log_file(str(log_file))
    
    def test_permission_error_handled(self, tmp_path, capsys):
        """Test that permission errors are handled gracefully."""
        log_file = tmp_path / "test.log"
        log_file.touch()
        
        with patch('os.chmod', side_effect=PermissionError("No permission")):
            _secure_log_file(str(log_file))
            captured = capsys.readouterr()
            assert "Warning: Could not set secure permissions" in captured.out


class TestSetupLogger:
    """Test logger setup functionality."""
    
    def test_basic_logger_creation(self, tmp_path):
        """Test basic logger creation with valid path."""
        log_file = tmp_path / "test.log"
        
        with patch('src.logger_setup.LOG_FILE', str(log_file)):
            logger = setup_logger('test_logger')
            
            assert logger is not None
            assert logger.name == 'test_logger'
            assert len(logger.handlers) > 0
            assert logger.level == logging.INFO
    
    def test_logger_with_existing_handlers(self, tmp_path):
        """Test that logger with existing handlers is returned as-is."""
        log_file = tmp_path / "test.log"
        
        with patch('src.logger_setup.LOG_FILE', str(log_file)):
            logger1 = setup_logger('test_logger_2')
            logger2 = setup_logger('test_logger_2')
            
            # Should return same logger without adding duplicate handlers
            assert logger1 is logger2
            assert len(logger2.handlers) == 1  # Only one handler
    
    def test_fallback_to_console_on_file_error(self, tmp_path):
        """Test fallback to console logging when file creation fails."""
        invalid_path = "/nonexistent/cannot/create/this.log"
        
        with patch('src.logger_setup.LOG_FILE', invalid_path):
            with patch('src.logger_setup._validate_log_path', return_value=invalid_path):
                logger = setup_logger('test_logger_3')
                
                assert logger is not None
                assert len(logger.handlers) > 0
                # Should have StreamHandler as fallback
                assert any(isinstance(h, logging.StreamHandler) 
                          for h in logger.handlers)
    
    def test_logger_creates_directory(self, tmp_path):
        """Test that logger creates log directory if needed."""
        # Use a path within cwd to avoid path traversal validation
        log_dir = Path('test_logs_subdir')
        log_file = log_dir / 'test.log'
        
        with patch('src.logger_setup.LOG_FILE', str(log_file)):
            logger = setup_logger('test_logger_4')
            
            assert logger is not None
            # Directory should be created
            assert log_dir.exists()
            
            # Cleanup
            import shutil
            if log_dir.exists():
                shutil.rmtree(log_dir)
    
    def test_exception_fallback_logger(self):
        """Test that exception during setup creates fallback logger."""
        with patch('src.logger_setup._validate_log_path', 
                   side_effect=Exception("Validation failed")):
            logger = setup_logger('test_logger_5')
            
            assert logger is not None
            # Should have fallback console handler
            assert any(isinstance(h, logging.StreamHandler) 
                      for h in logger.handlers)
    
    def test_logger_propagate_false(self, tmp_path):
        """Test that logger.propagate is set to False."""
        # Use rel path within cwd
        log_file = 'test_propagate.log'
        
        with patch('src.logger_setup.LOG_FILE', log_file):
            logger = setup_logger('test_logger_6')
            
            assert logger.propagate is False
            
            # Cleanup
            import os
            if os.path.exists(log_file):
                os.remove(log_file)
    
    def test_logger_formatter(self, tmp_path):
        """Test that logger has correct formatter."""
        log_file = tmp_path / "test.log"
        
        with patch('src.logger_setup.LOG_FILE', str(log_file)):
            logger = setup_logger('test_logger_7')
            
            handler = logger.handlers[0]
            formatter = handler.formatter
            assert formatter is not None
            assert 'asctime' in formatter._fmt
            assert 'levelname' in formatter._fmt
    
    def test_default_logger_instance_created(self):
        """Test that default logger instance is created at module level."""
        from src.logger_setup import logger
        assert logger is not None
        assert isinstance(logger, logging.Logger)
