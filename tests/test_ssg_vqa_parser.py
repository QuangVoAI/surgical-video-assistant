from pathlib import Path

from PIL import Image

from src.data.ssg_vqa import parse_ssg_vqa


def test_parse_ssg_vqa_pipe_delimited_qa(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa_txt" / "ssg-qa" / "VID01"
    qa_dir.mkdir(parents=True)
    (qa_dir / "1035.txt").write_text(
        "Which tools are present?|grasper, hook\n"
        "How many objects are visible?|2|single_and.json|count|NA\n",
        encoding="utf-8",
    )

    samples = list(parse_ssg_vqa(tmp_path))

    assert len(samples) == 2
    assert samples[0].dataset == "SSG-VQA"
    assert samples[0].task_type == "tool_type"
    assert samples[1].task_type == "tool_count"
    assert samples[1].metadata["video_id"] == "VID01"


def test_parse_ssg_vqa_resolves_png_frames(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa_txt" / "ssg-qa" / "VID01"
    image_dir = tmp_path / "images" / "VID01"
    qa_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_dir / "001035.png")
    (qa_dir / "1035.txt").write_text("What action is visible?|dissect\n", encoding="utf-8")

    samples = list(parse_ssg_vqa(tmp_path))

    assert len(samples) == 1
    assert Path(samples[0].image_path).exists()
    assert samples[0].image_path.endswith("001035.png")
