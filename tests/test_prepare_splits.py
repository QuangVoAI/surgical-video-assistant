from pathlib import Path

from PIL import Image

from src.data.surgmllmbench import parse_surgmllmbench


def test_record_level_split_overrides_file_level_split(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (8, 8)).save(root / "frame.jpg")
    (root / "mixed.jsonl").write_text(
        '{"image":"frame.jpg","question":"What surgical phase is shown?","answer":"Preparation","split":"test"}\n',
        encoding="utf-8",
    )

    samples = list(parse_surgmllmbench(root))
    assert len(samples) == 1
    assert samples[0].split == "test"
