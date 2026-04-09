"""Tests for cortex/config.py — defaults and env-var overrides."""

import importlib


def _reload():
    import cortex.config as cfg
    importlib.reload(cfg)
    return cfg


def test_all_defaults_are_non_empty_strings():
    import cortex.config as cfg

    for attr in (
        "PERSIST_DIR",
        "DATA_DIR",
        "MODEL_PATH_GPU",
        "MODEL_PATH_CPU",
        "RL_FEEDBACK_DIR",
        "IMAGE_CACHE_DIR",
        "TFIDF_MODEL_DIR",
    ):
        value = getattr(cfg, attr)
        assert isinstance(value, str), f"{attr} is not a str"
        assert value, f"{attr} is empty"


def test_persist_dir_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_PERSIST_DIR", "/custom/db")
    cfg = _reload()
    assert cfg.PERSIST_DIR == "/custom/db"


def test_data_dir_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_DATA_DIR", "/custom/docs")
    cfg = _reload()
    assert cfg.DATA_DIR == "/custom/docs"


def test_model_gpu_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_GPU", "/models/custom.gguf")
    cfg = _reload()
    assert cfg.MODEL_PATH_GPU == "/models/custom.gguf"


def test_model_cpu_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_CPU", "/models/cpu-model")
    cfg = _reload()
    assert cfg.MODEL_PATH_CPU == "/models/cpu-model"


def test_rl_feedback_dir_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_RL_FEEDBACK_DIR", "/tmp/rl")
    cfg = _reload()
    assert cfg.RL_FEEDBACK_DIR == "/tmp/rl"


def test_image_cache_dir_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_IMAGE_CACHE_DIR", "/tmp/images")
    cfg = _reload()
    assert cfg.IMAGE_CACHE_DIR == "/tmp/images"


def test_tfidf_model_dir_env_override(monkeypatch):
    monkeypatch.setenv("CORTEX_TFIDF_MODEL_DIR", "/models/tfidf")
    cfg = _reload()
    assert cfg.TFIDF_MODEL_DIR == "/models/tfidf"
