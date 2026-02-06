"""
Registry for Continual Learning Methods
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseCLMethod

# Global registry for CL methods
CL_METHODS: Dict[str, Type["BaseCLMethod"]] = {}


def register_cl_method(name: str):
    """Decorator to register a CL method class."""
    def decorator(cls):
        CL_METHODS[name] = cls
        return cls
    return decorator


def get_cl_method(name: str) -> Type["BaseCLMethod"]:
    """Get a CL method class by name."""
    if name not in CL_METHODS:
        available = list(CL_METHODS.keys())
        raise ValueError(f"Unknown CL method: {name}. Available methods: {available}")
    return CL_METHODS[name]

