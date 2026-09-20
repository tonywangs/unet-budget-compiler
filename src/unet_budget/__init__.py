"""Dependency-free architecture selection and code generation."""

__version__ = "0.1.0"

from .compiler import SpecError, compile_spec

__all__ = ["SpecError", "compile_spec", "__version__"]
