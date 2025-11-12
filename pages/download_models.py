# download_models.py
import os
from pathlib import Path
import gdown

def download_models_if_missing():
    """Automatically downloads .joblib models from Google Drive if not found locally."""

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ✅ Your Google Drive file IDs
    FILES = {
        "best_classification_model_new.joblib": "12OJpdV-0int-VZlP6h4sh7YgsIfiEzMq",
        "best_regression_model_new.joblib": "1n6TQjMc2GCd1SPmr_aIzONb0uDp6WOzs",
        "label_encoder.joblib": "1NBRmnhWk_a8le-v7uR1tJ89yEB__ax5e",
    }

    for filename, file_id in FILES.items():
        dest_path = models_dir / filename
        if not dest_path.exists():
            print(f"📥 Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, str(dest_path), quiet=False)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"✅ {filename} already exists.")

if __name__ == "__main__":
    download_models_if_missing()
