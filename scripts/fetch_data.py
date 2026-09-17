"""Fetch the dataset and wire it into the code tree.

    python scripts/fetch_data.py                 # download from the Hugging Face Hub
    python scripts/fetch_data.py --only 3d       # one suite
    python scripts/fetch_data.py --local PATH    # use an existing copy of the dataset

Afterwards `suite2d/data/2d` holds the 2D suite and `suite3d/` holds the volumetric
probes, segmentations and cached model outputs in the layout the code reads.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = "jhq0709/broken-evidence"
WRITABLE = {"figdata"}          # analysis scripts rewrite these files, so they are copied
SHALLOW = {"results_new"}       # analysis scripts add files here, so only the contents are linked


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        return
    if src.name in WRITABLE and src.is_dir():
        shutil.copytree(src, dst)
    elif src.name in SHALLOW and src.is_dir():
        dst.mkdir()
        for child in sorted(src.iterdir()):
            (dst / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="path to a local copy of the dataset")
    ap.add_argument("--only", choices=["2d", "3d"], help="fetch one suite")
    args = ap.parse_args()
    if args.local:
        root = Path(args.local)
    else:
        from huggingface_hub import snapshot_download
        patterns = [f"{args.only}/**"] if args.only else None
        root = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=patterns,
                                      local_dir=REPO / "data" / "hub"))
    if args.only in (None, "2d"):
        place(root / "2d", REPO / "suite2d" / "data" / "2d")
    if args.only in (None, "3d"):
        for entry in sorted((root / "3d").iterdir()):
            place(entry, REPO / "suite3d" / entry.name)
    print(f"data ready from {root}")


if __name__ == "__main__":
    main()
