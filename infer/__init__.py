"""infer/__init__.py - Inference modules"""
from .api import APIModelInfer
from .local import LocalInfer


def create_infer(mode: str = "api", **kwargs):
    """Factory: mode=api → APIModelInfer, mode=local → LocalInfer"""
    if mode == "api":
        return APIModelInfer(**kwargs)
    elif mode == "local":
        return LocalInfer(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
