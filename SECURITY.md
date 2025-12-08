# Security Guidelines

## Critical Security Fixes Applied

### 1. Credential Management
- **Issue**: Hardcoded credentials in `.env` file
- **Fix**: Removed real credentials and added placeholder values
- **Action Required**: 
  - Never commit real credentials to version control
  - Use environment-specific `.env` files (`.env.local`, `.env.production`)
  - Ensure `.env` is in `.gitignore` (already configured)

### 2. Path Traversal Protection
- **Issue**: Potential path traversal in model loading
- **Fix**: Added path validation in `_validate_model_path()` method
- **Protection**: 
  - Validates file extensions (.joblib only)
  - Blocks `..` and `~` in paths
  - Resolves to absolute paths

### 3. Resource Exhaustion Prevention
- **Issue**: Loading arbitrarily large model files
- **Fix**: Added 100MB file size limit for model files
- **Protection**: Prevents DoS attacks via large file uploads

### 4. Input Validation
- **Issue**: Insufficient validation of user inputs
- **Fix**: Added comprehensive validation in `train_model.py`
- **Protection**:
  - Validates experiment/run names (alphanumeric only)
  - Validates network interface names
  - Validates file paths and sizes
  - Limits string lengths

### 5. Secure File Permissions
- **Issue**: Model directory created with default permissions
- **Fix**: Set directory permissions to 0o750 (owner: rwx, group: r-x, others: none)
- **Protection**: Prevents unauthorized access to model files

## Security Best Practices

### Environment Variables
```bash
# DO NOT commit these values
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export MLFLOW_TRACKING_URI="http://your-server:5050"
```

### Running with Least Privilege
```bash
# Use sudo only when necessary for packet capture
sudo python main.py

# For training without live capture, no sudo needed
python train_model.py --from-file data/training/baseline.csv
```

### Model File Security
- Store models in a dedicated directory with restricted permissions
- Validate model files before loading
- Use checksums to verify model integrity
- Regularly audit model files for tampering

### Network Security
- Run packet capture on isolated network segments when possible
- Use firewall rules to restrict MLflow/MinIO access
- Enable TLS/SSL for MLflow tracking server
- Use VPN for remote MLflow access

## Vulnerability Reporting

If you discover a security vulnerability, please:
1. **DO NOT** open a public issue
2. Email security concerns to the maintainers
3. Include detailed reproduction steps
4. Allow time for patching before disclosure

## Security Checklist

- [ ] `.env` file is not committed to version control
- [ ] Real credentials are stored securely (not in code)
- [ ] Model files have restricted permissions (750 or stricter)
- [ ] MLflow server uses authentication
- [ ] MinIO uses strong access keys
- [ ] Network interfaces are properly configured
- [ ] Log files don't contain sensitive data
- [ ] Regular security updates applied

## Known Limitations

1. **Pickle/Joblib Deserialization**: Model loading uses joblib which relies on pickle. Only load models from trusted sources.
2. **Packet Capture**: Requires root/sudo privileges which increases attack surface.
3. **Log Files**: May contain IP addresses and network patterns. Secure appropriately.

## Compliance Notes

- **GDPR**: IP addresses may be considered personal data. Implement appropriate retention policies.
- **PCI DSS**: Do not capture payment card data in network traffic.
- **HIPAA**: Do not use on networks carrying protected health information without proper safeguards.
