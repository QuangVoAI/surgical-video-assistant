from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from src.data.schema import SurgicalSample, resolve_image_path


def parse_cholec_labels(raw_root: Path) -> Iterator[SurgicalSample]:
    label_files = sorted(
        list(raw_root.rglob("*.csv")) + list(raw_root.rglob("*.jsonl")) + list(raw_root.rglob("*.json"))
    )
    for label_file in label_files:
        for row_index, row in enumerate(iter_label_rows(label_file)):
            yield from row_to_samples(row, raw_root, label_file, row_index)


def iter_label_rows(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return

        payload = json.load(handle)
        if isinstance(payload, list):
            yield from payload
        elif isinstance(payload, dict):
            for key in ("data", "annotations", "samples", "records"):
                if isinstance(payload.get(key), list):
                    yield from payload[key]
                    return
            yield payload


def row_to_samples(row: dict, raw_root: Path, source_file: Path, row_index: int) -> Iterator[SurgicalSample]:
    image_value = first_present(row, ["image", "image_path", "frame_path", "frame", "filename"])
    if not image_value:
        return

    split = str(row.get("split") or ("test" if "test" in source_file.name.lower() else "train"))
    image_path = resolve_image_path(raw_root, str(image_value), source_file)
    base_id = str(row.get("id") or row.get("sample_id") or f"{source_file.stem}-{row_index}")

    phase = first_present(row, ["phase", "phase_label", "surgical_phase"])
    if phase:
        yield make_sample(base_id, image_path, split, "phase", "What surgical phase is shown?", str(phase), source_file)

    instruments = first_present(row, ["instrument", "instruments", "tool", "tools"])
    if instruments:
        yield make_sample(
            base_id,
            image_path,
            split,
            "tool_type",
            "Which surgical instruments are visible?",
            normalize_list_answer(instruments),
            source_file,
        )

    verb = first_present(row, ["verb", "action", "instrument_action"])
    if verb:
        yield make_sample(
            base_id,
            image_path,
            split,
            "action",
            "What action is the instrument performing?",
            str(verb),
            source_file,
        )

    target = first_present(row, ["target", "anatomy_target"])
    if instruments and verb and target:
        answer = f"{normalize_list_answer(instruments)} | {verb} | {target}"
        yield make_sample(
            base_id,
            image_path,
            split,
            "triplet",
            "What is the surgical action triplet <instrument, verb, target>?",
            answer,
            source_file,
        )


def first_present(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def normalize_list_answer(value: str) -> str:
    return ", ".join(part.strip() for part in str(value).replace(";", ",").split(",") if part.strip())


def make_sample(
    base_id: str,
    image_path: str,
    split: str,
    task_type: str,
    question: str,
    answer: str,
    source_file: Path,
) -> SurgicalSample:
    return SurgicalSample(
        sample_id=f"{base_id}-{task_type}",
        dataset="Cholec",
        image_path=image_path,
        question=question,
        answer=answer,
        task_type=task_type,
        split=split,
        metadata={"source_file": str(source_file)},
    )
