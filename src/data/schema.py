from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TASK_TYPES = {
    "vqa",
    "phase",
    "tool_count",
    "tool_type",
    "action",
    "segmentation",
    "triplet",
}


@dataclass(frozen=True)
class SurgicalSample:
    sample_id: str
    dataset: str
    image_path: str
    question: str
    answer: str
    task_type: str
    split: str
    metadata: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        item = asdict(self)
        if item["metadata"] is None:
            item["metadata"] = {}
        return item

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "SurgicalSample":
        return cls(
            sample_id=str(item["sample_id"]),
            dataset=str(item["dataset"]),
            image_path=str(item["image_path"]),
            question=str(item["question"]),
            answer=str(item["answer"]),
            task_type=str(item["task_type"]),
            split=str(item["split"]),
            metadata=dict(item.get("metadata") or {}),
        )


def resolve_image_path(raw_root: Path, image_value: str, source_file: Path | None = None) -> str:
    image_path = Path(image_value)
    if image_path.is_absolute():
        return str(image_path)

    candidates = []
    if source_file is not None:
        candidates.append(source_file.parent / image_path)
    candidates.append(raw_root / image_path)
    candidates.append(raw_root / "images" / image_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def infer_task_type(question: str, source_path: str = "") -> str:
    text = f"{question} {source_path}".lower()
    if "phase" in text or "workflow" in text:
        return "phase"
    if "count" in text or "number of" in text or "how many" in text:
        return "tool_count"
    if "instrument" in text or "tool" in text:
        if "action" in text or "perform" in text or "doing" in text:
            return "action"
        return "tool_type"
    if "verb" in text or "action" in text or "target" in text or "triplet" in text:
        return "action"
    if "segment" in text or "mask" in text or "anatomy" in text:
        return "segmentation"
    return "vqa"
