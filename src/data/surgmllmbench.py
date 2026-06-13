from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator

from src.data.io import iter_json_records
from src.data.schema import SurgicalSample, infer_task_type, resolve_image_path


def parse_surgmllmbench(raw_root: Path) -> Iterator[SurgicalSample]:
    json_files = sorted(
        path
        for pattern in ("*.json", "*.jsonl")
        for path in raw_root.rglob(pattern)
        if path.is_file()
    )

    for path in json_files:
        split = infer_split(path)
        for row_index, record in enumerate(iter_json_records(path)):
            parsed = parse_record(record, raw_root, path, split, row_index)
            if parsed is not None:
                yield parsed


def parse_record(
    record: dict,
    raw_root: Path,
    source_file: Path,
    split: str,
    row_index: int,
) -> SurgicalSample | None:
    image_value = record.get("image") or record.get("image_path") or record.get("img")
    question = record.get("question")
    answer = record.get("answer")

    conversations = record.get("conversations")
    if conversations and (question is None or answer is None):
        question, answer = extract_conversation_qa(conversations)

    if not image_value or not question or answer is None:
        return None

    sample_id = str(
        record.get("id")
        or record.get("question_id")
        or record.get("sample_id")
        or uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}:{row_index}:{image_value}:{question}")
    )
    image_path = resolve_image_path(raw_root, str(image_value), source_file)
    question = str(question).replace("<image>", "").strip()
    answer = str(answer).strip()
    task_type = str(record.get("task_type") or infer_task_type(question, str(source_file)))

    return SurgicalSample(
        sample_id=sample_id,
        dataset="SurgMLLMBench",
        image_path=image_path,
        question=question,
        answer=answer,
        task_type=task_type,
        split=split,
        metadata={
            "source_file": str(source_file),
            "raw_image": str(image_value),
        },
    )


def extract_conversation_qa(conversations: list[dict]) -> tuple[str | None, str | None]:
    question = None
    answer = None
    for message in conversations:
        role = str(message.get("from") or message.get("role") or "").lower()
        value = message.get("value") or message.get("content")
        if value is None:
            continue
        if role in {"human", "user"} and question is None:
            question = str(value)
        elif role in {"gpt", "assistant", "model"} and answer is None:
            answer = str(value)
    return question, answer


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "test" in parts or "test" in name:
        return "test"
    if "val" in parts or "validation" in parts or "val" in name:
        return "val"
    return "train"
