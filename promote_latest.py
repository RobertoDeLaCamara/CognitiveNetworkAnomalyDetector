import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.mlflow_config import REGISTERED_MODEL_NAME

# Load environment variables
load_dotenv()

def promote_latest_model():
    print(f"Connecting to MLflow at {os.environ.get('MLFLOW_TRACKING_URI')}...")
    client = MlflowClient()
    
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    
    if not versions:
        print("No model versions found!")
        return
        
    # Sort by version number (descending)
    versions.sort(key=lambda x: int(x.version), reverse=True)
    latest_version = versions[0]
    
    print(f"Latest version is v{latest_version.version} (current stage: {latest_version.current_stage})")
    
    if latest_version.current_stage != "Production":
        print(f"Promoting v{latest_version.version} to Production...")
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print("✅ Promotion complete!")
    else:
        print("Latest version is already in Production.")

if __name__ == "__main__":
    promote_latest_model()
