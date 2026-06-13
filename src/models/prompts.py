from __future__ import annotations

from collections import defaultdict
from random import Random

from src.data.schema import SurgicalSample


PHASE_CHOICES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping and Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning and Coagulation",
    "Gallbladder Retraction",
]


def system_prompt() -> str:
    return (
        "You are a surgical frame understanding assistant for research use only. "
        "Answer concisely from the visible image. Do not provide clinical advice."
    )


def build_prompt(
    sample: SurgicalSample,
    examples: list[SurgicalSample] | None = None,
    include_system: bool = True,
) -> str:
    system = system_prompt()
    task_instruction = instruction_for_task(sample.task_type, sample.metadata or {})
    few_shot = ""
    if examples:
        rendered = []
        for example in examples:
            rendered.append(f"Question: {example.question}\nAnswer: {example.answer}")
        few_shot = "\n\nExamples:\n" + "\n\n".join(rendered)
    body = f"{task_instruction}{few_shot}\n\nQuestion: {sample.question}\nAnswer:"
    if include_system:
        return f"{system}\n\n{body}"
    return body


def instruction_for_task(task_type: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    answer_space = metadata.get("answer_space") or []
    answer_space_text = ""
    if answer_space:
        answer_space_text = " Valid labels: " + "; ".join(str(item) for item in answer_space[:30]) + "."
    if task_type == "phase":
        choices = answer_space or PHASE_CHOICES
        rendered = "; ".join(choices)
        return f"Task: identify the surgical phase. If applicable, choose one of: {rendered}."
    if task_type == "tool_count":
        return "Task: count visible surgical instruments. Reply with a number only when possible."
    if task_type == "tool_type":
        return "Task: list visible surgical instruments. Use comma-separated tool names." + answer_space_text
    if task_type in {"action", "triplet"}:
        return "Task: identify the surgical instrument action. Prefer the dataset vocabulary if it is clear." + answer_space_text
    if task_type == "segmentation":
        return "Task: answer about visible segmented instruments or anatomy. Keep the answer short."
    return "Task: answer the visual question about this surgical frame."


def select_few_shot_examples(
    train_samples: list[SurgicalSample],
    target_samples: list[SurgicalSample],
    k: int,
    seed: int = 13,
) -> dict[str, list[SurgicalSample]]:
    rng = Random(seed)
    by_task: dict[str, list[SurgicalSample]] = defaultdict(list)
    for sample in train_samples:
        if sample.split == "train":
            by_task[sample.task_type].append(sample)

    selected: dict[str, list[SurgicalSample]] = {}
    for sample in target_samples:
        candidates = [candidate for candidate in by_task.get(sample.task_type, []) if candidate.sample_id != sample.sample_id]
        rng.shuffle(candidates)
        selected[sample.sample_id] = candidates[:k]
    return selected
