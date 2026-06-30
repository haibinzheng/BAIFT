"""src/models/__init__.py"""
from .api import APIModel
from .base import BaseModel
from .local import LocalModel


def create_model(model_name: str, mode: str = "api", **kwargs) -> BaseModel:
    """Factory: mode='api' → APIModel, mode='local' → LocalModel"""
    if mode == "api":
        return APIModel(model_name, **kwargs)
    elif mode == "local":
        return LocalModel(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}, use 'api' or 'local'")
