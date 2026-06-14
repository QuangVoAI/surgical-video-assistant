from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from typing import Iterable


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans({mark: " " for mark in string.punctuation}))
    text = re.sub(r"\s+", " ", text)
    text = text.replace("carlot triangle dissection", "calot triangle dissection")
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_labels(text: str) -> set[str]:
    raw_parts = re.split(r"[,;|/]+", str(text))
    return {normalize_text(part) for part in raw_parts if normalize_text(part)}


def multilabel_scores(rows: Iterable[dict]) -> dict[str, float]:
    tp = fp = fn = exact = total = 0
    per_label: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        total += 1
        pred = parse_labels(row["prediction"])
        gold = parse_labels(row["ground_truth"])
        exact += int(pred == gold)
        for label in pred | gold:
            if label in pred and label in gold:
                tp += 1
                per_label[label]["tp"] += 1
            elif label in pred:
                fp += 1
                per_label[label]["fp"] += 1
            else:
                fn += 1
                per_label[label]["fn"] += 1

    micro_precision = safe_div(tp, tp + fp)
    micro_recall = safe_div(tp, tp + fn)
    micro_f1 = harmonic(micro_precision, micro_recall)
    macro_f1_values = []
    for counts in per_label.values():
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        macro_f1_values.append(harmonic(precision, recall))

    return {
        "exact_match": safe_div(exact, total),
        "micro_f1": micro_f1,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
    }


def classification_scores(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    total = len(rows)
    correct = sum(exact_match(row["prediction"], row["ground_truth"]) for row in rows)
    labels = sorted({normalize_text(row["ground_truth"]) for row in rows} | {normalize_text(row["prediction"]) for row in rows})
    per_label = {}
    confusion: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        pred = normalize_text(row["prediction"])
        gold = normalize_text(row["ground_truth"])
        confusion[gold][pred] += 1

    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            pred = normalize_text(row["prediction"])
            gold = normalize_text(row["ground_truth"])
            if pred == label and gold == label:
                tp += 1
            elif pred == label:
                fp += 1
            elif gold == label:
                fn += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": harmonic(precision, recall),
        }

    return {
        "accuracy": safe_div(correct, total),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(per_label) if per_label else 0.0,
        "per_label": per_label,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }


def count_scores(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    exact = 0
    absolute_errors = []
    for row in rows:
        pred = extract_first_number(row["prediction"])
        gold = extract_first_number(row["ground_truth"])
        if pred is None or gold is None:
            absolute_errors.append(math.nan)
            continue
        exact += int(pred == gold)
        absolute_errors.append(abs(pred - gold))
    numeric_errors = [value for value in absolute_errors if not math.isnan(value)]
    return {
        "accuracy": safe_div(exact, len(rows)),
        "mae": sum(numeric_errors) / len(numeric_errors) if numeric_errors else math.nan,
        "numeric_coverage": safe_div(len(numeric_errors), len(rows)),
    }


def extract_first_number(text: str) -> int | None:
    match = re.search(r"\d+", str(text))
    if not match:
        return None
    return int(match.group(0))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
