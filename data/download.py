"""Download HAM10000 from Kaggle and print where it was saved.

Usage (Colab or local):
    python data/download.py

Authentication: set the environment variable KAGGLE_API_TOKEN (in Colab, add it to
Colab Secrets and load it with `os.environ["KAGGLE_API_TOKEN"] = userdata.get(...)`),
or place an old-style kaggle.json in ~/.kaggle/. Never commit the token.
"""

from pathlib import Path

import kagglehub

DATASET = "kmader/skin-cancer-mnist-ham10000"


def download() -> Path:
    """Download the dataset (or reuse the local cache) and return its folder."""
    return Path(kagglehub.dataset_download(DATASET))


if __name__ == "__main__":
    folder = download()
    print("HAM10000 is in:", folder)
    for item in sorted(folder.iterdir()):
        print("  ", item.name)
