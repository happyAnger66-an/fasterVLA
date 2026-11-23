"""Dataclass config loading with unknown-key rejection."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def from_mapping(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass from a mapping; unknown or missing required keys fail hard."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"expected dict, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")

    required = {
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"missing required keys for {cls.__name__}: {sorted(missing)}")

    return cls(**{k: data[k] for k in known if k in data})  # type: ignore[call-arg]


def load_dataclass_yaml(cls: type[T], path: str | Path) -> T:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return from_mapping(cls, raw)
