from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.data.schema import SurgicalSample, infer_task_type, resolve_image_path


def parse_ssg_vqa(raw_root: Path) -> Iterator[SurgicalSample]:
    qa_root = find_qa_root(raw_root)
    for text_file in sorted(qa_root.rglob("*.txt")):
        if text_file.name.upper() == "LICENSE":
            continue
        video_id = text_file.parent.name
        frame_id = text_file.stem
        image_value, image_path = resolve_ssg_image(raw_root, video_id, frame_id, text_file)
        split = infer_ssg_split(video_id)
        for line_index, line in enumerate(text_file.read_text(encoding="utf-8", errors="replace").splitlines()):
            parsed = parse_qa_line(line)
            if parsed is None:
                continue
            question, answer, metadata = parsed
            task_type = infer_ssg_task_type(question, metadata)
            yield SurgicalSample(
                sample_id=f"SSG-VQA-{video_id}-{frame_id}-{line_index}",
                dataset="SSG-VQA",
                image_path=image_path,
                question=question,
                answer=answer,
                task_type=task_type,
                split=split,
                metadata={
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "source_file": str(text_file),
                    "raw_image": image_value,
                    **metadata,
                },
            )


def find_qa_root(raw_root: Path) -> Path:
    candidates = [
        raw_root / "qa_txt" / "ssg-qa",
        raw_root / "qa_txt",
        raw_root / "ssg-qa",
        raw_root,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.txt")):
            return candidate
    return raw_root


def parse_qa_line(line: str) -> tuple[str, str, dict] | None:
    line = line.strip()
    if not line or "|" not in line:
        return None
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 2:
        return None
    question, answer = parts[0], parts[1]
    if not question:
        return None
    metadata = {}
    if len(parts) > 2:
        metadata["raw_fields"] = parts[2:]
    if len(parts) > 3:
        metadata["question_family"] = parts[2]
        metadata["answer_type"] = parts[3]
    return question, answer, metadata


def candidate_frame_path(video_id: str, frame_id: str) -> str:
    return candidate_frame_paths(video_id, frame_id)[0]


def candidate_frame_paths(video_id: str, frame_id: str) -> list[str]:
    frame_number = frame_id.zfill(6)
    return [
        f"images/{video_id}/{frame_number}.jpg",
        f"images/{video_id}/{frame_number}.png",
        f"videos/{video_id}/{frame_number}.png",
        f"videos/{video_id}/{frame_number}.jpg",
    ]


def resolve_ssg_image(raw_root: Path, video_id: str, frame_id: str, text_file: Path) -> tuple[str, str]:
    candidates = candidate_frame_paths(video_id, frame_id)
    for image_value in candidates:
        image_path = resolve_image_path(raw_root, image_value, text_file)
        if Path(image_path).exists():
            return image_value, image_path
    image_value = candidates[0]
    return image_value, resolve_image_path(raw_root, image_value, text_file)


def infer_ssg_split(video_id: str) -> str:
    digits = "".join(char for char in video_id if char.isdigit())
    if not digits:
        return "train"
    number = int(digits)
    return "val" if number % 5 == 0 else "train"


def infer_ssg_task_type(question: str, metadata: dict) -> str:
    answer_type = str(metadata.get("answer_type", "")).lower()
    question_lower = question.lower()
    if answer_type == "count" or "how many" in question_lower or "what number" in question_lower:
        return "tool_count"
    if "tool" in question_lower or "instrument" in question_lower:
        return "tool_type"
    if "action" in question_lower:
        return "action"
    return infer_task_type(question)
