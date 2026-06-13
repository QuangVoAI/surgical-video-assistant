from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from src.data.schema import SurgicalSample


def iter_json_records(path: Path) -> Iterator[dict]:
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
        else:
            raise ValueError(f"Unsupported JSON payload in {path}")


def write_jsonl(samples: Iterable[SurgicalSample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_samples(path: Path) -> list[SurgicalSample]:
    return [SurgicalSample.from_json(item) for item in iter_json_records(path)]
