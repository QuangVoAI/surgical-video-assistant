from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st
import yaml

from src.data.schema import SurgicalSample
from src.models.prompts import build_prompt
from src.models.providers import build_provider


TASK_OPTIONS = {
    "Free-form VQA": ("vqa", "What is visible in this surgical frame?"),
    "Phase recognition": ("phase", "What surgical phase is shown?"),
    "Tool recognition": ("tool_type", "Which surgical instruments are visible?"),
    "Tool count": ("tool_count", "How many surgical instruments are visible?"),
    "Action recognition": ("action", "What action is the instrument performing?"),
}


@st.cache_resource
def cached_provider(config_path: str):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return build_provider(config["model"])


def main() -> None:
    st.set_page_config(page_title="Surgical Frame Assistant", layout="wide")
    st.title("Surgical Frame Assistant")
    st.caption("Research and education only. Not clinical decision support.")

    with st.sidebar:
        config_path = st.text_input("Inference config", value="configs/mock_zero_shot.yaml")
        task_label = st.selectbox("Task", list(TASK_OPTIONS))
        custom_question = st.text_area("Question", value=TASK_OPTIONS[task_label][1], height=100)
        run = st.button("Analyze frame", type="primary")

    uploaded = st.file_uploader("Upload a surgical frame", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        st.image(uploaded, caption="Uploaded frame", use_container_width=True)

    if run:
        if not uploaded:
            st.error("Upload an image first.")
            return
        try:
            provider = cached_provider(config_path)
            image_path = save_uploaded_file(uploaded)
            task_type, _ = TASK_OPTIONS[task_label]
            sample = SurgicalSample(
                sample_id=str(uuid.uuid4()),
                dataset="demo",
                image_path=str(image_path),
                question=custom_question.strip() or TASK_OPTIONS[task_label][1],
                answer="",
                task_type=task_type,
                split="demo",
                metadata={},
            )
            prompt = build_prompt(sample)
            answer = provider.generate(sample, prompt)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Heuristic confidence")
            st.write(confidence_hint(task_type, answer))
        except Exception as exc:
            st.exception(exc)


def save_uploaded_file(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".png"
    tmp_dir = Path(tempfile.gettempdir()) / "surgical_frame_assistant"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return path


def confidence_hint(task_type: str, answer: str) -> str:
    clean = answer.strip()
    if not clean:
        return "Low: empty answer."
    if task_type == "tool_count":
        return "Higher" if any(char.isdigit() for char in clean) else "Low: no numeric count detected."
    if task_type in {"phase", "tool_type", "action"}:
        return "Medium: answer has expected text format. Confirm with benchmark metrics."
    return "Medium: free-form answer requires human review."


if __name__ == "__main__":
    main()
