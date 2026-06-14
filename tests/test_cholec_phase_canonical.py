from src.data.cholec import canonicalize_phase_label


def test_canonicalize_phase_label_variants() -> None:
    assert canonicalize_phase_label("carlot-triangle-dissection") == "Calot Triangle Dissection"
    assert canonicalize_phase_label("clipping-and-cutting") == "Clipping and Cutting"
    assert canonicalize_phase_label("gallbladder-retraction") == "Gallbladder Extraction"
