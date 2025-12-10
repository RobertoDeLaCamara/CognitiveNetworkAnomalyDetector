import os
import mlflow
from src.mlflow_config import apply_s3_config, get_tracking_uri

# Apply env vars
apply_s3_config()
mlflow.set_tracking_uri(get_tracking_uri())

RUN_ID = "fdf06609b0c24765a976b68e7267e0ee"

print(f"Listing artifacts for run {RUN_ID}...")
try:
    artifacts = mlflow.artifacts.list_artifacts(run_id=RUN_ID)
    for artifact in artifacts:
        print(f"- {artifact.path} (is_dir={artifact.is_dir})")
        if artifact.is_dir:
            sub_artifacts = mlflow.artifacts.list_artifacts(run_id=RUN_ID, artifact_path=artifact.path)
            for sub in sub_artifacts:
                print(f"  - {sub.path}")
except Exception as e:
    print(f"Error: {e}")
