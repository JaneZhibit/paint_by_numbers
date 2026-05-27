"""Hugging Face: only local cache, no network. Import before transformers."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def local_files_only_from_config(config: dict | None) -> bool:
    if config is None:
        return True
    return config.get("ml_models", {}).get("local_files_only", True)


def create_hf_pipeline(task: str, model: str, config: dict | None = None, **kwargs):
    """Load a transformers pipeline from the local HF cache only."""
    from transformers import pipeline

    local_only = kwargs.pop("local_files_only", local_files_only_from_config(config))
    return pipeline(task, model=model, local_files_only=local_only, **kwargs)
