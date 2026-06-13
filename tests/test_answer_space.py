from src.data.schema import SurgicalSample
from src.models.answer_space import build_answer_spaces, constrain_prediction


def test_build_answer_space_and_constrain_phase_prediction() -> None:
    samples = [
        SurgicalSample(
            sample_id="1",
            dataset="demo",
            image_path="frame.jpg",
            question="What phase?",
            answer="Preparation",
            task_type="phase",
            split="train",
            metadata={},
        ),
        SurgicalSample(
            sample_id="2",
            dataset="demo",
            image_path="frame.jpg",
            question="What phase?",
            answer="Clipping and Cutting",
            task_type="phase",
            split="train",
            metadata={},
        ),
    ]
    spaces = build_answer_spaces(samples)
    assert "phase" in spaces
    assert constrain_prediction("phase", "the phase is preparation", spaces) == "Preparation"


def test_constrain_tool_prediction_from_allowed_labels() -> None:
    spaces = {"tool_type": ["grasper", "hook", "clipper"]}
    assert constrain_prediction("tool_type", "grasper, unknown, hook", spaces) == "grasper, hook"
