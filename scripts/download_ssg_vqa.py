from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


SSG_URLS = {
    "qa": "https://s3.unistra.fr/camma_public/github/ssg-qa/ssg-qa.zip",
    "cropped_images": "https://s3.unistra.fr/camma_public/github/ssg-qa/cropped_images.zip",
    "roi_yolo_coord": "https://s3.unistra.fr/camma_public/github/ssg-qa/roi_yolo_coord.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public SSG-VQA QA pairs and optional feature archives.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/SSG-VQA"))
    parser.add_argument("--features", action="store_true", help="Also download pre-extracted feature archives.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    download_and_extract("qa", args.out)
    if args.features:
        download_and_extract("cropped_images", args.out / "visual_feats")
        download_and_extract("roi_yolo_coord", args.out / "visual_feats")

    print(f"Downloaded SSG-VQA files to {args.out}")
    print("Note: QA pairs are public, but raw frame images still require CholecT45/Cholec80 access.")
    print(f"Next: python -m src.data.prepare --dataset ssg-vqa --raw {args.out} --out data/processed")


def download_and_extract(key: str, out_dir: Path) -> None:
    url = SSG_URLS[key]
    archive = out_dir / f"{key}.zip"
    if not archive.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive)
    print(f"Extracting {archive}")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(out_dir)
    if key == "qa":
        qa_dir = out_dir / "qa_txt"
        qa_dir.mkdir(exist_ok=True)
        extracted = out_dir / "ssg-qa"
        if extracted.exists() and not (qa_dir / "ssg-qa").exists():
            shutil.move(str(extracted), str(qa_dir / "ssg-qa"))


if __name__ == "__main__":
    main()
