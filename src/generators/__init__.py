"""src/generators/__init__.py"""
from .baift import BAIFTGenerator
from .caffe import CAFFEGenerator
from .metal import METALGenerator
from .roleplay import RolePlayGenerator

GENERATORS = {
    "baift": BAIFTGenerator,
    "caffe": CAFFEGenerator,
    "metal": METALGenerator,
    "llm_roleplay": RolePlayGenerator,
}


def create_generator(method: str, model=None, **kwargs):
    cls = GENERATORS.get(method.lower())
    if cls is None:
        raise ValueError(f"Unknown method: {method}, choose from {list(GENERATORS.keys())}")
    return cls(model=model, **kwargs)
