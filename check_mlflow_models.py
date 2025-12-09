
import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

# Load env vars
load_dotenv()

# Set tracking URI
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
print(f"Tracking URI: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

# List registered models
print("\n--- Registered Models ---")
models = client.search_registered_models()
for rm in models:
    print(f"Name: {rm.name}")
    for v in rm.latest_versions:
        print(f"  Version: {v.version}, Stage: {v.current_stage}, RunID: {v.run_id}")
