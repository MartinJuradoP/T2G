# triples/__init__.py
from .schemas import TripleIR, DocumentTriples
from .dep_triples import DepTripleExtractor, DepTripleConfig, run_on_file

__all__ = [
    "TripleIR",
    "DocumentTriples",
    "DepTripleExtractor",
    "DepTripleConfig",
    "run_on_file",
]
