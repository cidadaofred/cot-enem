from cot_enem.runtime.diagnostics import _cached_huggingface_models_gb


def test_cached_model_measurement_counts_only_selected_repositories(tmp_path):
    selected = tmp_path / "hub" / "models--org--selected" / "blobs"
    unrelated = tmp_path / "hub" / "models--org--unrelated" / "blobs"
    selected.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    (selected / "weights").write_bytes(b"x" * 1024)
    (unrelated / "weights").write_bytes(b"x" * 4096)

    cached_gb = _cached_huggingface_models_gb(tmp_path, ["org/selected"])

    assert cached_gb == 1024 / 1024**3
