from __future__ import annotations

import re
from collections import defaultdict

from src.data.schema import SurgicalSample
from src.eval.metrics import normalize_text, parse_labels


CLASSIFICATION_TASKS = {"phase", "tool_type", "action", "triplet"}


def build_answer_spaces(samples: list[SurgicalSample]) -> dict[str, list[str]]:
    by_task: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.task_type not in CLASSIFICATION_TASKS:
            continue
        if sample.task_type == "tool_type":
            by_task[sample.task_type].update(
                label for label in parse_labels(sample.answer) if label
            )
        else:
            by_task[sample.task_type].add(sample.answer.strip())
    return {task: sorted(values) for task, values in by_task.items()}


def attach_answer_spaces(
    samples: list[SurgicalSample],
    answer_spaces: dict[str, list[str]],
) -> list[SurgicalSample]:
    enriched: list[SurgicalSample] = []
    for sample in samples:
        metadata = dict(sample.metadata or {})
        if sample.task_type in answer_spaces:
            metadata["answer_space"] = answer_spaces[sample.task_type]
        enriched.append(
            SurgicalSample(
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                image_path=sample.image_path,
                question=sample.question,
                answer=sample.answer,
                task_type=sample.task_type,
                split=sample.split,
                metadata=metadata,
            )
        )
    return enriched


def constrain_prediction(
    task_type: str,
    prediction: str,
    answer_spaces: dict[str, list[str]],
) -> str:
    if task_type not in answer_spaces:
        return prediction.strip()

    if task_type == "tool_type":
        allowed = answer_spaces[task_type]
        normalized_to_original = {normalize_text(item): item for item in allowed}
        matched = []
        for token in ordered_prediction_labels(prediction):
            best = pick_best_label(token, allowed)
            if best is not None:
                matched.append(normalized_to_original[normalize_text(best)])
        deduped = []
        seen = set()
        for item in matched:
            key = normalize_text(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return ", ".join(deduped) if deduped else prediction.strip()

    best = pick_best_label(prediction, answer_spaces[task_type])
    return best if best is not None else prediction.strip()


def ordered_prediction_labels(text: str) -> list[str]:
    return [normalize_text(part) for part in re.split(r"[,;|/]+", str(text)) if normalize_text(part)]


def pick_best_label(prediction: str, candidates: list[str]) -> str | None:
    pred = normalize_text(prediction)
    if not pred:
        return None
    exact = [candidate for candidate in candidates if normalize_text(candidate) == pred]
    if exact:
        return exact[0]

    contained = [
        candidate
        for candidate in candidates
        if pred in normalize_text(candidate) or normalize_text(candidate) in pred
    ]
    if contained:
        return sorted(contained, key=lambda item: (abs(len(normalize_text(item)) - len(pred)), len(item)))[0]
    return None
