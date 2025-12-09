
import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

client = MlflowClient()
model_name = "isolation-forest-anomaly-detector"
version = 1

print(f"Transitioning {model_name} version {version} to Production...")
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True
)
print("Done.")
