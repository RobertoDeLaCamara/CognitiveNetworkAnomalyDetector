"""Tests for mlflow_config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.mlflow_config import (
    get_tracking_uri,
    is_remote_tracking,
    get_experiment_name,
    is_mlflow_enabled,
    get_run_name,
    get_s3_config,
    apply_s3_config,
    validate_remote_config,
    DEFAULT_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME
)


class TestGetTrackingUri:
    """Test get_tracking_uri function."""
    
    def test_returns_env_variable_when_set(self):
        """Test that environment variable takes precedence."""
        with patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://custom:5000'}):
            uri = get_tracking_uri()
            assert uri == 'http://custom:5000'
    
    def test_returns_default_when_env_not_set(self):
        """Test that default config is used when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove MLFLOW_TRACKING_URI if it exists
            os.environ.pop('MLFLOW_TRACKING_URI', None)
            uri = get_tracking_uri()
            assert uri is not None
            # Should return local file store or configured URI


class TestIsRemoteTracking:
    """Test is_remote_tracking function."""
    
    def test_http_uri_returns_true(self):
        """Test that HTTP URIs are detected as remote."""
        with patch('src.mlflow_config.get_tracking_uri', return_value='http://mlflow:5000'):
            assert is_remote_tracking() is True
    
    def test_https_uri_returns_true(self):
        """Test that HTTPS URIs are detected as remote."""
        with patch('src.mlflow_config.get_tracking_uri', return_value='https://mlflow.example.com'):
            assert is_remote_tracking() is True
    
    def test_file_uri_returns_false(self):
        """Test that file URIs are detected as local."""
        with patch('src.mlflow_config.get_tracking_uri', return_value='file:///tmp/mlruns'):
            assert is_remote_tracking() is False
    
    def test_local_path_returns_false(self):
        """Test that local paths are detected as local."""
        with patch('src.mlflow_config.get_tracking_uri', return_value='/tmp/mlruns'):
            assert is_remote_tracking() is False


class TestGetExperimentName:
    """Test get_experiment_name function."""
    
    def test_returns_default_when_no_custom_name(self):
        """Test default experiment name is returned."""
        name = get_experiment_name()
        assert name == DEFAULT_EXPERIMENT_NAME
    
    def test_returns_custom_name_when_provided(self):
        """Test custom name is returned when provided."""
        custom = "my-custom-experiment"
        name = get_experiment_name(custom)
        assert name == custom
    
    def test_none_returns_default(self):
        """Test that None returns default name."""
        name = get_experiment_name(None)
        assert name == DEFAULT_EXPERIMENT_NAME


class TestIsMLflowEnabled:
    """Test is_mlflow_enabled function."""
    
    def test_returns_true_when_env_true(self):
        """Test returns True when environment variable is 'true'."""
        with patch.dict(os.environ, {'MLFLOW_ENABLED': 'true'}):
            assert is_mlflow_enabled() is True
    
    def test_returns_true_when_env_1(self):
        """Test returns True when environment variable is '1'."""
        with patch.dict(os.environ, {'MLFLOW_ENABLED': '1'}):
            assert is_mlflow_enabled() is True
    
    def test_returns_true_when_env_yes(self):
        """Test returns True when environment variable is 'yes'."""
        with patch.dict(os.environ, {'MLFLOW_ENABLED': 'yes'}):
            assert is_mlflow_enabled() is True
    
    def test_returns_false_when_env_false(self):
        """Test returns False when environment variable is 'false'."""
        with patch.dict(os.environ, {'MLFLOW_ENABLED': 'false'}):
            assert is_mlflow_enabled() is False
    
    def test_returns_false_when_env_0(self):
        """Test returns False when environment variable is '0'."""
        with patch.dict(os.environ, {'MLFLOW_ENABLED': '0'}):
            assert is_mlflow_enabled() is False
    
    def test_case_insensitive(self):
        """Test that comparison is case insensitive."""
        with patch.dict(os.environ, {'MLFLOW_ENABLED': 'TRUE'}):
            assert is_mlflow_enabled() is True
        with patch.dict(os.environ, {'MLFLOW_ENABLED': 'FALSE'}):
            assert is_mlflow_enabled() is False


class TestGetRunName:
    """Test get_run_name function."""
    
    def test_default_run_name_format(self):
        """Test default run name format."""
        name = get_run_name()
        assert name.startswith('run_')
        # Should have timestamp format YYYYMMDD_HHMMSS
        assert len(name) > 10
    
    def test_custom_prefix(self):
        """Test custom prefix in run name."""
        name = get_run_name(prefix='training')
        assert name.startswith('training_')
    
    def test_with_version(self):
        """Test run name with version number."""
        name = get_run_name(prefix='experiment', version=5)
        assert 'v5' in name
        assert name.startswith('experiment_v5_')
    
    def test_timestamp_in_name(self):
        """Test that timestamp is included in name."""
        import time
        before = time.strftime('%Y%m%d')
        name = get_run_name()
        # Should contain today's date
        assert before in name


class TestGetS3Config:
    """Test get_s3_config function."""
    
    def test_empty_config_when_no_env_vars(self):
        """Test returns empty dict when no S3 env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('MLFLOW_S3_ENDPOINT_URL', None)
            os.environ.pop('AWS_ACCESS_KEY_ID', None)
            os.environ.pop('AWS_SECRET_ACCESS_KEY', None)
            config = get_s3_config()
            # Config might be empty or have default values
            assert isinstance(config, dict)
    
    def test_includes_endpoint_when_set(self):
        """Test includes endpoint URL when set."""
        with patch.dict(os.environ, {'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000'}):
            config = get_s3_config()
            assert 'MLFLOW_S3_ENDPOINT_URL' in config
            assert config['MLFLOW_S3_ENDPOINT_URL'] == 'http://minio:9000'
    
    def test_includes_credentials_when_set(self):
        """Test includes AWS credentials when set."""
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        }):
            config = get_s3_config()
            assert config.get('AWS_ACCESS_KEY_ID') == 'test_key'
            assert config.get('AWS_SECRET_ACCESS_KEY') == 'test_secret'
    
    def test_complete_s3_config(self):
        """Test complete S3 configuration."""
        with patch.dict(os.environ, {
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'AWS_ACCESS_KEY_ID': 'minioadmin',
            'AWS_SECRET_ACCESS_KEY': 'minioadmin'
        }):
            config = get_s3_config()
            assert len(config) == 3
            assert config['MLFLOW_S3_ENDPOINT_URL'] == 'http://minio:9000'
            assert config['AWS_ACCESS_KEY_ID'] == 'minioadmin'
            assert config['AWS_SECRET_ACCESS_KEY'] == 'minioadmin'


class TestApplyS3Config:
    """Test apply_s3_config function."""
    
    def test_sets_environment_variables(self):
        """Test that S3 config is applied to environment."""
        test_config = {
            'MLFLOW_S3_ENDPOINT_URL': 'http://test:9000',
            'AWS_ACCESS_KEY_ID': 'test_id',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        }
        
        with patch('src.mlflow_config.get_s3_config', return_value=test_config):
            with patch.dict(os.environ, {}, clear=True):
                apply_s3_config()
                
                assert os.environ.get('MLFLOW_S3_ENDPOINT_URL') == 'http://test:9000'
                assert os.environ.get('AWS_ACCESS_KEY_ID') == 'test_id'
                assert os.environ.get('AWS_SECRET_ACCESS_KEY') == 'test_secret'
    
    def test_skips_none_values(self):
        """Test that None values are not set in environment."""
        test_config = {
            'MLFLOW_S3_ENDPOINT_URL': None,
            'AWS_ACCESS_KEY_ID': 'test_id'
        }
        
        with patch('src.mlflow_config.get_s3_config', return_value=test_config):
            with patch.dict(os.environ, {}, clear=True):
                apply_s3_config()
                
                assert 'MLFLOW_S3_ENDPOINT_URL' not in os.environ
                assert os.environ.get('AWS_ACCESS_KEY_ID') == 'test_id'


class TestValidateRemoteConfig:
    """Test validate_remote_config function."""
    
    def test_valid_local_config(self):
        """Test validation passes for local tracking."""
        with patch('src.mlflow_config.is_remote_tracking', return_value=False):
            is_valid, issues = validate_remote_config()
            assert is_valid is True
            assert len(issues) == 0
    
    def test_valid_remote_config_without_minio(self):
        """Test validation passes for remote tracking without MinIO."""
        with patch('src.mlflow_config.is_remote_tracking', return_value=True):
            with patch('src.mlflow_config.REMOTE_MLFLOW_SERVER', 'http://mlflow:5000'):
                with patch('src.mlflow_config.MINIO_ENDPOINT', None):
                    is_valid, issues = validate_remote_config()
                    assert is_valid is True
                    assert len(issues) == 0
    
    def test_invalid_remote_config_missing_credentials(self):
        """Test validation fails when MinIO credentials missing."""
        with patch('src.mlflow_config.is_remote_tracking', return_value=True):
            with patch('src.mlflow_config.REMOTE_MLFLOW_SERVER', 'http://mlflow:5000'):
                with patch('src.mlflow_config.MINIO_ENDPOINT', 'http://minio:9000'):
                    with patch('src.mlflow_config.AWS_ACCESS_KEY_ID', None):
                        with patch('src.mlflow_config.AWS_SECRET_ACCESS_KEY', None):
                            is_valid, issues = validate_remote_config()
                            assert is_valid is False
                            assert len(issues) > 0
                            assert any('AWS_ACCESS_KEY_ID' in issue for issue in issues)
    
    def test_returns_specific_issues(self):
        """Test that specific issues are returned."""
        with patch('src.mlflow_config.is_remote_tracking', return_value=True):
            with patch('src.mlflow_config.REMOTE_MLFLOW_SERVER', 'http://mlflow:5000'):
                with patch('src.mlflow_config.MINIO_ENDPOINT', 'http://minio:9000'):
                    with patch('src.mlflow_config.AWS_ACCESS_KEY_ID', None):
                        with patch('src.mlflow_config.AWS_SECRET_ACCESS_KEY', 'secret'):
                            is_valid, issues = validate_remote_config()
                            assert is_valid is False
                            # Should have issue about missing access key
                            assert any('AWS_ACCESS_KEY_ID' in issue for issue in issues)
                            # Should NOT have issue about secret key
                            assert not any('AWS_SECRET_ACCESS_KEY' in issue and 'not set' in issue for issue in issues)


class TestMLflowConfigConstants:
    """Test that module constants are properly defined."""
    
    def test_default_experiment_name_defined(self):
        """Test that default experiment name is defined."""
        assert DEFAULT_EXPERIMENT_NAME is not None
        assert isinstance(DEFAULT_EXPERIMENT_NAME, str)
        assert len(DEFAULT_EXPERIMENT_NAME) > 0
    
    def test_registered_model_name_defined(self):
        """Test that registered model name is defined."""
        assert REGISTERED_MODEL_NAME is not None
        assert isinstance(REGISTERED_MODEL_NAME, str)
        assert len(REGISTERED_MODEL_NAME) > 0
