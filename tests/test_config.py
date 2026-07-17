import os

import pytest

from cot_enem.config import load_env_file


def test_load_env_file_preserves_existing_variables(tmp_path, monkeypatch):
    source = tmp_path / ".env"
    source.write_text(
        "# baseline\nLLM_API_KEY=local\nLLM_MODEL=\"model-a\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "from-process")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    loaded = load_env_file(source)
    assert os.environ["LLM_API_KEY"] == "from-process"
    assert os.environ["LLM_MODEL"] == "model-a"
    assert loaded == {"LLM_MODEL": "model-a"}


def test_load_env_file_rejects_invalid_lines(tmp_path):
    source = tmp_path / ".env"
    source.write_text("INVALID LINE", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid environment entry"):
        load_env_file(source)
