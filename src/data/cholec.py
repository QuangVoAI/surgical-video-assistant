from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from src.data.schema import SurgicalSample, resolve_image_path
from src.models.prompts import PHASE_CHOICES


CHOLECT50_CROSSVAL_FOLDS = {
    1: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111],
    2: [80, 32, 5, 15, 40, 47, 26, 48, 70, 96],
    3: [31, 57, 36, 18, 52, 68, 10, 8, 73, 103],
    4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
    5: [78, 43, 62, 35, 74, 1, 56, 4, 13, 92],
}

CHOLECT50_PHASE_CANONICAL = {
    "preparation": "Preparation",
    "calot triangle dissection": "Calot Triangle Dissection",
    "carlot triangle dissection": "Calot Triangle Dissection",
    "calot-triangle-dissection": "Calot Triangle Dissection",
    "carlot-triangle-dissection": "Calot Triangle Dissection",
    "clipping and cutting": "Clipping and Cutting",
    "clipping-and-cutting": "Clipping and Cutting",
    "gallbladder dissection": "Gallbladder Dissection",
    "gallbladder-dissection": "Gallbladder Dissection",
    "gallbladder packaging": "Gallbladder Packaging",
    "gallbladder-packaging": "Gallbladder Packaging",
    "cleaning and coagulation": "Cleaning and Coagulation",
    "cleaning-and-coagulation": "Cleaning and Coagulation",
    "gallbladder extraction": "Gallbladder Extraction",
    "gallbladder-extraction": "Gallbladder Extraction",
    "gallbladder retraction": "Gallbladder Extraction",
    "gallbladder-retraction": "Gallbladder Extraction",
}


def parse_cholec_labels(raw_root: Path) -> Iterator[SurgicalSample]:
    if (raw_root / "labels").is_dir() and (raw_root / "videos").is_dir():
        yield from parse_native_cholect50(raw_root)
        return

    label_files = sorted(
        list(raw_root.rglob("*.csv")) + list(raw_root.rglob("*.jsonl")) + list(raw_root.rglob("*.json"))
    )
    for label_file in label_files:
        for row_index, row in enumerate(iter_label_rows(label_file)):
            yield from row_to_samples(row, raw_root, label_file, row_index)


def parse_native_cholect50(raw_root: Path, test_fold: int = 1) -> Iterator[SurgicalSample]:
    split_by_video = build_cholect50_splits(test_fold)
    for label_file in sorted((raw_root / "labels").glob("VID*.json")):
        payload = json.loads(label_file.read_text(encoding="utf-8"))
        video_name = label_file.stem
        video_id = int(str(payload.get("video") or video_name.replace("VID", "")).lstrip("0") or 0)
        split = split_by_video.get(video_id, "train")
        categories = normalize_categories(payload.get("categories", {}))
        annotations = payload.get("annotations", {})

        for frame_key, frame_annotations in sorted(annotations.items(), key=lambda item: int(item[0])):
            image_path = raw_root / "videos" / video_name / f"{int(frame_key):06d}.png"
            frame_labels = collect_native_frame_labels(frame_annotations, categories)
            base_id = f"cholect50-{video_name}-{int(frame_key):06d}"
            metadata = {
                "source_file": str(label_file),
                "video_id": video_name,
                "frame_id": int(frame_key),
                "native_video_id": video_id,
            }

            if frame_labels["phase"]:
                phase_metadata = dict(metadata)
                phase_metadata["answer_space"] = PHASE_CHOICES
                yield make_sample(
                    base_id,
                    str(image_path),
                    split,
                    "phase",
                    "What surgical phase is shown?",
                    frame_labels["phase"],
                    label_file,
                    phase_metadata,
                )
            if frame_labels["instruments"]:
                yield make_sample(
                    base_id,
                    str(image_path),
                    split,
                    "tool_type",
                    "Which surgical instruments are visible?",
                    ", ".join(frame_labels["instruments"]),
                    label_file,
                    metadata,
                )
            if frame_labels["verbs"]:
                yield make_sample(
                    base_id,
                    str(image_path),
                    split,
                    "action",
                    "What surgical actions are visible?",
                    ", ".join(frame_labels["verbs"]),
                    label_file,
                    metadata,
                )
            if frame_labels["triplets"]:
                yield make_sample(
                    base_id,
                    str(image_path),
                    split,
                    "triplet",
                    "What surgical action triplets <instrument, verb, target> are visible?",
                    ", ".join(frame_labels["triplets"]),
                    label_file,
                    metadata,
                )


def build_cholect50_splits(test_fold: int) -> dict[int, str]:
    test_videos = set(CHOLECT50_CROSSVAL_FOLDS[test_fold])
    training_pool = [
        video_id
        for fold, video_ids in CHOLECT50_CROSSVAL_FOLDS.items()
        if fold != test_fold
        for video_id in video_ids
    ]
    val_videos = set(training_pool[-5:])
    split_by_video = {video_id: "test" for video_id in test_videos}
    split_by_video.update({video_id: "val" for video_id in val_videos})
    for video_id in training_pool:
        split_by_video.setdefault(video_id, "train")
    return split_by_video


def normalize_categories(categories: Any) -> dict[str, dict[str, str]]:
    if isinstance(categories, list):
        merged: dict[str, dict[str, str]] = {}
        for item in categories:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, dict):
                        merged[key] = {str(k): str(v) for k, v in value.items()}
        return merged
    if isinstance(categories, dict):
        return {
            str(key): {str(k): str(v) for k, v in value.items()}
            for key, value in categories.items()
            if isinstance(value, dict)
        }
    return {}


def collect_native_frame_labels(frame_annotations: Any, categories: dict[str, dict[str, str]]) -> dict[str, Any]:
    instruments: list[str] = []
    verbs: list[str] = []
    targets: list[str] = []
    triplets: list[str] = []
    phase = ""

    if not isinstance(frame_annotations, list):
        return {
            "phase": phase,
            "instruments": instruments,
            "verbs": verbs,
            "targets": targets,
            "triplets": triplets,
        }

    for annotation in frame_annotations:
        if not isinstance(annotation, list) or len(annotation) < 15:
            continue
        triplet_id = parse_label_id(annotation[0])
        instrument_id = first_valid_id(annotation[1:7])
        verb_id = parse_label_id(annotation[7])
        target_id = first_valid_id(annotation[8:14])
        phase_id = parse_label_id(annotation[14])

        instrument = category_name(categories, "instrument", instrument_id)
        verb = category_name(categories, "verb", verb_id)
        target = category_name(categories, "target", target_id)
        triplet = category_name(categories, "triplet", triplet_id)
        frame_phase = category_name(categories, "phase", phase_id)

        if frame_phase and not phase:
            phase = frame_phase
        append_unique(instruments, instrument)
        append_unique(verbs, verb)
        append_unique(targets, target)
        if triplet:
            append_unique(triplets, triplet)
        elif instrument and verb and target:
            append_unique(triplets, f"<{instrument}, {verb}, {target}>")

    return {
        "phase": phase,
        "instruments": instruments,
        "verbs": verbs,
        "targets": targets,
        "triplets": triplets,
    }


def first_valid_id(values: list[Any]) -> int | None:
    for value in values:
        label_id = parse_label_id(value)
        if label_id is not None:
            return label_id
    return None


def parse_label_id(value: Any) -> int | None:
    try:
        label_id = int(value)
    except (TypeError, ValueError):
        return None
    return label_id if label_id >= 0 else None


def category_name(categories: dict[str, dict[str, str]], task: str, label_id: int | None) -> str:
    if label_id is None:
        return ""
    task_categories = categories.get(task, {})
    value = task_categories.get(str(label_id), str(label_id))
    if task == "phase":
        return canonicalize_phase_label(value)
    return value


def append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def canonicalize_phase_label(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", " ").replace("/", " ")
    normalized = " ".join(normalized.split())
    return CHOLECT50_PHASE_CANONICAL.get(normalized, str(value).strip())


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
    metadata: dict[str, Any] | None = None,
) -> SurgicalSample:
    sample_metadata = {"source_file": str(source_file)}
    if metadata:
        sample_metadata.update(metadata)
    return SurgicalSample(
        sample_id=f"{base_id}-{task_type}",
        dataset="Cholec",
        image_path=image_path,
        question=question,
        answer=answer,
        task_type=task_type,
        split=split,
        metadata=sample_metadata,
    )
