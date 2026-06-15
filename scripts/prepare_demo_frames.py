from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "reports" / "demo_cases.json"
DEFAULT_OUT = PROJECT_ROOT / "demo_samples"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a small set of authorized held-out frames for the Modal demo."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=[],
        help="Extra root to search, for example /workspace/surgical-video-assistant.",
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    roots = [PROJECT_ROOT, Path.home() / "Downloads", *args.image_root]

    for case in cases:
        if len(manifest) >= args.limit:
            break
        source = find_image(case.get("image_path", ""), roots)
        if source is None:
            print(f"missing: {case.get('image_path')}")
            continue
        suffix = source.suffix or ".png"
        filename = f"{case['sample_id']}{suffix}"
        target = args.out / filename
        shutil.copy2(source, target)
        manifest.append(
            {
                "image": filename,
                "sample_id": case["sample_id"],
                "task_type": "phase",
                "question": case.get("question", "What surgical phase is shown?"),
                "ground_truth": case.get("ground_truth"),
                "source_note": "held-out evaluation example",
            }
        )
        print(f"copied: {source} -> {target}")

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest)} examples to {manifest_path}")
    if not manifest:
        print("No images were copied. Use --image-root to point to your CholecT50 project/data folder.")


def find_image(raw_path: str, roots: list[Path]) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else []
    candidates.extend(root / raw_path for root in roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
