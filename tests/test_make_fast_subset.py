from types import SimpleNamespace

from scripts.make_fast_subset import select_balanced_by_answer


def test_select_balanced_by_answer_spreads_labels() -> None:
    samples = []
    for label, count in (("A", 10), ("B", 2), ("C", 2)):
        for index in range(count):
            samples.append(SimpleNamespace(answer=label))

    selected = select_balanced_by_answer(samples, limit=6, rng=__import__("random").Random(13))
    counts = {}
    for sample in selected:
        counts[sample.answer] = counts.get(sample.answer, 0) + 1

    assert len(selected) == 6
    assert counts["A"] == 2
    assert counts["B"] == 2
    assert counts["C"] == 2
