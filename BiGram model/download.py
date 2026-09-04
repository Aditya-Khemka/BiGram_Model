import huggingface_hub
from huggingface_hub import snapshot_download
from tqdm import tqdm
import os

# Target directory
TARGET_DIR = r"C:/Users/USER/Desktop/ML/dataset"

# Enable tqdm progress bars (like Colab)
huggingface_hub.utils.enable_progress_bars()

print("Starting OpenWebText download...\n")

snapshot_download(
    repo_id="Skylion007/openwebtext",
    repo_type="dataset",
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,  # IMPORTANT for Windows
    resume_download=True,
)

print("\nDownload completed successfully!")
