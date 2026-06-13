import json
from pathlib import Path

from PIL import Image

from src.data.cholec import parse_cholec_labels


def test_parse_native_cholect50_labels(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos" / "VID01"
    label_dir = tmp_path / "labels"
    video_dir.mkdir(parents=True)
    label_dir.mkdir()
    Image.new("RGB", (8, 8)).save(video_dir / "000001.png")

    label_payload = {
        "video": 1,
        "fps": 1,
        "num_frames": 2,
        "categories": {
            "triplet": {"1": "<grasper, dissect, gallbladder>"},
            "instrument": {"0": "grasper"},
            "verb": {"2": "dissect"},
            "target": {"0": "gallbladder"},
            "phase": {"3": "calot triangle dissection"},
        },
        "annotations": {
            "1": [
                [
                    1,
                    0,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    2,
                    0,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    3,
                ]
            ]
        },
    }
    (label_dir / "VID01.json").write_text(json.dumps(label_payload), encoding="utf-8")

    samples = list(parse_cholec_labels(tmp_path))

    assert len(samples) == 4
    assert {sample.task_type for sample in samples} == {"phase", "tool_type", "action", "triplet"}
    assert all(Path(sample.image_path).exists() for sample in samples)
    assert all(sample.split == "val" for sample in samples)
    assert next(sample for sample in samples if sample.task_type == "phase").answer == "calot triangle dissection"
    assert next(sample for sample in samples if sample.task_type == "tool_type").answer == "grasper"
    assert next(sample for sample in samples if sample.task_type == "action").answer == "dissect"
