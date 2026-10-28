#!/usr/bin/env python3
"""Download the public Twitter sentiment CSV used by HW4.a and HW4.5."""

from __future__ import annotations

import shutil
import hashlib
import time
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "Twitter_Data.csv"
EXPECTED_HASH = "5a7af1ec5fc749ec8e5ea13c574aeb5c06254aa1c081e3421868079d5356b3f4"


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''): result.update(block)
    return result.hexdigest()


def publish_partial():
    partial = DESTINATION.with_suffix('.csv.partial')
    if digest(partial) != EXPECTED_HASH:
        raise RuntimeError('The downloaded CSV does not match the original HW4.a source hash. It was not used for training.')
    partial.replace(DESTINATION)


def main() -> None:
    if DESTINATION.exists() and digest(DESTINATION) == EXPECTED_HASH:
        print(f"Dataset already present: {DESTINATION}")
        return
    if DESTINATION.exists():
        invalid = DESTINATION.with_name('Twitter_Data.invalid.' + str(time.time_ns()) + '.csv')
        DESTINATION.replace(invalid)
        print('Preserved an invalid/incomplete download at', invalid)

    import kagglehub

    print("Downloading saurabhshahane/twitter-sentiment-dataset from Kaggle...")
    try:
        download_dir = Path(kagglehub.dataset_download("saurabhshahane/twitter-sentiment-dataset"))
    except Exception as error:
        print(f"Kaggle client download failed ({type(error).__name__}); trying the public dataset archive.")
        url = "https://www.kaggle.com/api/v1/datasets/download/saurabhshahane/twitter-sentiment-dataset"
        with tempfile.TemporaryDirectory(prefix="twitter-download-") as temporary:
            archive = Path(temporary) / "twitter.zip"
            request = urllib.request.Request(url, headers={"User-Agent": "DATA266-GPU-assignment/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            with zipfile.ZipFile(archive) as bundle:
                matches = [entry for entry in bundle.infolist() if Path(entry.filename).name.lower() == "twitter_data.csv"]
                if len(matches) != 1:
                    raise FileNotFoundError("The public Kaggle archive does not contain exactly one Twitter_Data.csv.")
                # Copy one known file; do not extract arbitrary archive paths.
                with bundle.open(matches[0]) as source, DESTINATION.with_suffix('.csv.partial').open("wb") as output:
                    shutil.copyfileobj(source, output)
        publish_partial()
        print(f"Copied dataset to {DESTINATION}")
        return
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
    shutil.copy2(candidates[0], DESTINATION.with_suffix('.csv.partial'))
    publish_partial()
    print(f"Copied dataset to {DESTINATION}")


if __name__ == "__main__":
    main()
