from pathlib import Path

from PIL import Image

from src.data.surgmllmbench import parse_surgmllmbench


def test_parse_conversations_and_direct_qa(tmp_path: Path) -> None:
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "frame.jpg")
    (image_dir / "qa.jsonl").write_text(
        "\n".join(
            [
                '{"id":"a","image":"frame.jpg","conversations":[{"from":"human","value":"<image>\\nWhat surgical phase is shown?"},{"from":"gpt","value":"Preparation"}]}',
                '{"question_id":2,"image":"frame.jpg","question":"Which surgical instruments are visible?","answer":"grasper"}',
            ]
        ),
        encoding="utf-8",
    )

    samples = list(parse_surgmllmbench(tmp_path))

    assert len(samples) == 2
    assert samples[0].question == "What surgical phase is shown?"
    assert samples[0].answer == "Preparation"
    assert samples[0].task_type == "phase"
    assert samples[1].sample_id == "2"
    assert samples[1].task_type == "tool_type"
