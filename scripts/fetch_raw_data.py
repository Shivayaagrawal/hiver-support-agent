"""Download the Kaggle Customer Support on Twitter dump into data/raw/twcs.csv."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "data" / "raw" / "twcs.csv"


def main() -> None:
    """Fetch twcs.csv via kagglehub unless it already exists locally."""
    if DST.exists() and DST.stat().st_size > 0:
        print(f"Already present: {DST} ({DST.stat().st_size} bytes)")
        return
    import kagglehub

    cache_path = Path(kagglehub.dataset_download("thoughtvector/customer-support-on-twitter"))
    src = cache_path / "twcs" / "twcs.csv"
    if not src.exists():
        raise SystemExit(f"Expected twcs.csv under {cache_path}, not found")
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, DST)
    print(f"Copied {src} -> {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
