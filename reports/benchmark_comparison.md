# Benchmark Comparison

This table is only valid when the dataset split and metric protocol match the cited paper.

| Task | Dataset | Paper baseline | Metric | Paper score | Gemma 4 12B | Gemma 4 26B-A4B | Citation |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| phase | Cholec80 | TeCNO | accuracy | TBD | TBD | TBD | https://arxiv.org/abs/2003.10751 |
| triplet | CholecT45/50 | Rendezvous | mAP | TBD | TBD | TBD | https://arxiv.org/abs/2109.03223 |
| triplet | CholecT45/50 | Rendezvous in Time | mAP | TBD | TBD | TBD | https://arxiv.org/abs/2211.16963 |
| triplet | CholecT45/50 | DiffTriplet | mAP | TBD | TBD | TBD | https://arxiv.org/abs/2406.13210 |
| vqa | SurgMLLMBench | SurgVLM / SurgMLLMBench baseline | exact_match_or_task_metric | TBD | TBD | TBD | https://arxiv.org/abs/2511.21339 |

## Notes

- Gemma 4 large model in this repo is `google/gemma-4-26B-A4B-it`, which is the official 26B-A4B MoE variant.
- Fill the `metric_value` fields from the exact experiment setting you cite before using this table in a report.
