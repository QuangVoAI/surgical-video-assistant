from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_REPORTS = Path.home() / "Downloads" / "reports"

PHASE_ORDER = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping and Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning and Coagulation",
    "Gallbladder Extraction",
]


def main() -> None:
    st.set_page_config(page_title="Surgical Frame QA Demo", layout="wide")
    add_styles()

    st.title("Surgical Frame QA Demo")
    st.caption("Gemma 4 12B zero-shot vs LoRA, with post-hoc calibration from saved candidate scores.")

    metrics = load_metrics()
    cases = load_demo_cases()

    with st.sidebar:
        st.header("Demo Controls")
        case_labels = [case_label(case) for case in cases]
        selected_index = st.selectbox("Frame case", range(len(cases)), format_func=lambda i: case_labels[i])
        show_scores = st.toggle("Show candidate losses", value=True)
        st.divider()
        st.write("Evaluation set")
        st.write("4,239 held-out phase samples")
        st.write("7 phase labels")

    show_metric_overview(metrics)
    st.divider()

    case = cases[selected_index]
    show_case(case, show_scores)

    st.divider()
    show_calibration_note(metrics)


def load_metrics() -> dict:
    paths = {
        "zeroshot_topk": PROJECT_ROOT / "reports/eval_full_phase_zeroshot_fixed/topk_zeroshot_fixed.json",
        "lora_topk": PROJECT_ROOT / "reports/eval_full_phase_lora_fixed/topk_lora_fixed.json",
        "calibration": PROJECT_ROOT / "reports/calibration/calibration_summary.json",
    }
    return {name: read_json(path) for name, path in paths.items()}


def load_demo_cases() -> list[dict]:
    local_path = PROJECT_ROOT / "reports/demo_cases.json"
    if local_path.exists():
        return read_json(local_path)

    zeroshot_path = DOWNLOAD_REPORTS / "eval_full_phase_zeroshot_fixed/predictions_gemma4_12b_zeroshot_phase_full_fixed.jsonl"
    lora_path = DOWNLOAD_REPORTS / "eval_full_phase_lora_fixed/predictions_lora12b_phase_full_fixed.jsonl"
    if not zeroshot_path.exists() or not lora_path.exists():
        st.error("Missing demo cases and full prediction files. Expected reports/demo_cases.json or prediction JSONL files.")
        st.stop()

    return build_cases_from_predictions(zeroshot_path, lora_path)


def read_json(path: Path):
    if not path.exists():
        st.error(f"Missing required file: {path}")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


def build_cases_from_predictions(zeroshot_path: Path, lora_path: Path) -> list[dict]:
    zero = read_jsonl_by_id(zeroshot_path)
    lora = read_jsonl_by_id(lora_path)
    cases = []
    for sample_id, lora_row in lora.items():
        if sample_id not in zero:
            continue
        zero_row = zero[sample_id]
        cases.append(
            {
                "sample_id": sample_id,
                "video_id": (lora_row.get("metadata") or {}).get("video_id"),
                "frame_id": (lora_row.get("metadata") or {}).get("frame_id"),
                "image_path": lora_row.get("image_path"),
                "question": lora_row.get("question"),
                "ground_truth": lora_row.get("ground_truth"),
                "zeroshot_prediction": zero_row.get("prediction"),
                "lora_prediction": lora_row.get("prediction"),
                "calibrated_prediction": lora_row.get("prediction"),
                "zeroshot_top_scores": zero_row.get("top_scores", [])[:5],
                "lora_top_scores": lora_row.get("top_scores", [])[:5],
                "lora_calibrated_top_scores": lora_row.get("top_scores", [])[:5],
            }
        )
        if len(cases) >= 12:
            break
    return cases


def read_jsonl_by_id(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def show_metric_overview(metrics: dict) -> None:
    zero = metrics["zeroshot_topk"]
    lora = metrics["lora_topk"]
    calibration = metrics["calibration"]["lora_fixed"]
    lora_after = calibration["after_heldout"]

    st.subheader("Full Phase Evaluation")
    cols = st.columns(3)
    metric_card(cols[0], "Zero-shot Top-1", percent(zero["top1_acc"]), "Collapses to majority class")
    metric_card(cols[1], "LoRA Top-5", percent(lora["top5_acc"]), "Correct label often appears in top candidates")
    metric_card(cols[2], "LoRA + Calibration Top-1", percent(lora_after["top1_acc"]), "Post-hoc reranking, no GPU retraining")

    rows = [
        metric_row("Gemma 4 12B zero-shot", zero, macro_f1=0.0792812096715284),
        metric_row("Gemma 4 12B + LoRA", lora, macro_f1=0.10650782919690482),
        {
            "Setting": "LoRA + calibration",
            "Top-1": percent(lora_after["top1_acc"]),
            "Top-3": percent(lora_after["top3_acc"]),
            "Top-5": percent(lora_after["top5_acc"]),
            "MRR": f"{lora_after['mrr']:.3f}",
            "Macro F1": percent(lora_after["macro_f1"]),
        },
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def metric_row(name: str, topk: dict, macro_f1: float) -> dict:
    return {
        "Setting": name,
        "Top-1": percent(topk["top1_acc"]),
        "Top-3": percent(topk["top3_acc"]),
        "Top-5": percent(topk["top5_acc"]),
        "MRR": f"{topk['mrr']:.3f}",
        "Macro F1": percent(macro_f1),
    }


def metric_card(column, label: str, value: str, caption: str) -> None:
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_case(case: dict, show_scores: bool) -> None:
    st.subheader("Frame-Level QA Case")
    left, right = st.columns([1, 1.45])

    with left:
        image_path = resolve_image_path(case.get("image_path", ""))
        if image_path and image_path.exists():
            st.image(str(image_path), caption=f"{case.get('video_id')} frame {case.get('frame_id')}", use_container_width=True)
        else:
            st.markdown(
                f"""
                <div class="image-placeholder">
                  <div class="placeholder-title">Frame preview unavailable</div>
                  <div>{case.get('image_path', 'No image path')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("Question")
        st.info(case.get("question", "What surgical phase is shown?"))

    with right:
        status_row(case)
        cols = st.columns(3)
        answer_card(cols[0], "Zero-shot", case.get("zeroshot_prediction"), case.get("ground_truth"))
        answer_card(cols[1], "LoRA", case.get("lora_prediction"), case.get("ground_truth"))
        answer_card(cols[2], "Calibrated", case.get("calibrated_prediction"), case.get("ground_truth"))

        if show_scores:
            st.write("Candidate ranking")
            score_cols = st.columns(3)
            score_table(score_cols[0], "Zero-shot top-5", case.get("zeroshot_top_scores", []))
            score_table(score_cols[1], "LoRA top-5", case.get("lora_top_scores", []))
            score_table(score_cols[2], "LoRA calibrated top-5", case.get("lora_calibrated_top_scores", []), calibrated=True)


def status_row(case: dict) -> None:
    st.markdown(
        f"""
        <div class="case-strip">
          <div><span>Ground truth</span><strong>{case.get('ground_truth')}</strong></div>
          <div><span>Video</span><strong>{case.get('video_id', 'unknown')}</strong></div>
          <div><span>Frame</span><strong>{case.get('frame_id', 'unknown')}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def answer_card(column, label: str, prediction: str | None, ground_truth: str | None) -> None:
    correct = normalize(prediction) == normalize(ground_truth)
    class_name = "answer-ok" if correct else "answer-miss"
    verdict = "Correct" if correct else "Needs reranking"
    with column:
        st.markdown(
            f"""
            <div class="answer-card {class_name}">
              <div class="answer-label">{label}</div>
              <div class="answer-text">{prediction or 'N/A'}</div>
              <div class="answer-verdict">{verdict}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def score_table(column, title: str, scores: list[dict], calibrated: bool = False) -> None:
    with column:
        st.markdown(f"**{title}**")
        rows = []
        for index, score in enumerate(scores, start=1):
            value = score.get("calibrated_loss") if calibrated else score.get("loss")
            rows.append(
                {
                    "Rank": index,
                    "Answer": score.get("answer"),
                    "Loss": "" if value is None else f"{float(value):.3f}",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)


def show_calibration_note(metrics: dict) -> None:
    calibration = metrics["calibration"]["lora_fixed"]
    before = calibration["before_heldout"]
    after = calibration["after_heldout"]
    st.subheader("Calibration Finding")
    st.markdown(
        f"""
        Calibration is a post-processing step that adjusts candidate scores before choosing the final Top-1 answer.
        In this run, LoRA Top-1 improved from **{percent(before["top1_acc"])}** to
        **{percent(after["top1_acc"])}** on the held-out split, while Top-3 moved from
        **{percent(before["top3_acc"])}** to **{percent(after["top3_acc"])}**.
        """
    )
    st.caption("This reranks saved candidate losses only. It does not retrain Gemma and does not use GPU.")


def resolve_image_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidates = [
        PROJECT_ROOT / raw_path,
        Path.home() / "Downloads" / raw_path,
        Path("/workspace/surgical-video-assistant") / raw_path,
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def case_label(case: dict) -> str:
    video = case.get("video_id", "video")
    frame = case.get("frame_id", "frame")
    gt = case.get("ground_truth", "unknown")
    return f"{video} / {frame} / {gt}"


def normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def add_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fa;
        }
        .metric-card, .answer-card, .image-placeholder, .case-strip {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .metric-card {
            min-height: 128px;
            padding: 18px 20px;
        }
        .metric-label, .answer-label, .case-strip span {
            color: #5e6675;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .metric-value {
            color: #172033;
            font-size: 34px;
            font-weight: 750;
            line-height: 1.15;
            margin-top: 8px;
        }
        .metric-caption, .answer-verdict {
            color: #6b7280;
            font-size: 14px;
            margin-top: 8px;
        }
        .case-strip {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 12px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .case-strip div {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .case-strip strong {
            color: #172033;
            font-size: 16px;
        }
        .answer-card {
            min-height: 150px;
            padding: 16px;
            border-left-width: 5px;
        }
        .answer-ok {
            border-left-color: #238636;
        }
        .answer-miss {
            border-left-color: #d29922;
        }
        .answer-text {
            color: #172033;
            font-size: 20px;
            font-weight: 700;
            line-height: 1.25;
            margin-top: 12px;
            overflow-wrap: anywhere;
        }
        .image-placeholder {
            min-height: 260px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 24px;
            color: #5e6675;
            overflow-wrap: anywhere;
        }
        .placeholder-title {
            color: #172033;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
