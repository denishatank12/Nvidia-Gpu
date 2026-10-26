#!/usr/bin/env python3
"""Download the public Twitter sentiment CSV used by HW4.a and HW4.5."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "Twitter_Data.csv"


def main() -> None:
    if DESTINATION.exists():
        print(f"Dataset already present: {DESTINATION}")
        return

    import kagglehub

    print("Downloading saurabhshahane/twitter-sentiment-dataset from Kaggle...")
    download_dir = Path(kagglehub.dataset_download("saurabhshahane/twitter-sentiment-dataset"))
    candidates = list(download_dir.rglob("Twitter_Data.csv"))
    if not candidates:
        candidates = [
            path
            for path in download_dir.rglob("*.csv")
            if path.name.lower() == "twitter_data.csv"
        ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one Twitter_Data.csv under {download_dir}, found {len(candidates)}."
        )
    shutil.copy2(candidates[0], DESTINATION)
    print(f"Copied dataset to {DESTINATION}")


if __name__ == "__main__":
    main()
