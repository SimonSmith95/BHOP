"""
Phase 1 — Data Models
Defines all shared dataclasses and enums used across the application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional


class ParameterType(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    CATEGORICAL = "CATEGORICAL"


@dataclass
class AllowedSubRange:
    """A single continuous allowed interval [low, high] for INT or FLOAT params."""
    low: float
    high: float

    def to_dict(self) -> dict:
        return {"low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, d: dict) -> AllowedSubRange:
        return cls(low=float(d["low"]), high=float(d["high"]))

    def __repr__(self) -> str:
        return f"[{self.low}, {self.high}]"


@dataclass
class ParameterConfig:
    """Full configuration for a single experiment parameter."""
    name: str
    ptype: ParameterType
    enabled: bool = True

    # ── INT / FLOAT fields ─────────────────────────────────────────────────
    full_min: float = 0.0
    full_max: float = 1.0
    allowed_subranges: List[AllowedSubRange] = field(default_factory=list)
    step: Optional[int] = None          # INT only; None = step of 1

    # ── CATEGORICAL / BOOL fields ──────────────────────────────────────────
    all_choices: List[str] = field(default_factory=list)      # all values seen in CSV
    allowed_choices: List[str] = field(default_factory=list)  # subset the user allows

    # ── BOOL field ─────────────────────────────────────────────────────────
    fixed_value: Optional[bool] = None  # None = optimize; True/False = held constant

    def __post_init__(self):
        # Default allowed_subranges to the full range if not provided
        if self.ptype in (ParameterType.INT, ParameterType.FLOAT):
            if not self.allowed_subranges:
                self.allowed_subranges = [AllowedSubRange(self.full_min, self.full_max)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ptype": self.ptype.value,
            "enabled": self.enabled,
            "full_min": self.full_min,
            "full_max": self.full_max,
            "allowed_subranges": [r.to_dict() for r in self.allowed_subranges],
            "step": self.step,
            "all_choices": self.all_choices,
            "allowed_choices": self.allowed_choices,
            "fixed_value": self.fixed_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParameterConfig:
        return cls(
            name=d["name"],
            ptype=ParameterType(d["ptype"]),
            enabled=d.get("enabled", True),
            full_min=float(d.get("full_min", 0.0)),
            full_max=float(d.get("full_max", 1.0)),
            allowed_subranges=[AllowedSubRange.from_dict(r) for r in d.get("allowed_subranges", [])],
            step=d.get("step"),
            all_choices=d.get("all_choices", []),
            allowed_choices=d.get("allowed_choices", []),
            fixed_value=d.get("fixed_value"),
        )


@dataclass
class ObjectiveConfig:
    """Configuration for one optimization objective (result column)."""
    column_name: str
    direction: Literal["minimize", "maximize"] = "minimize"

    def to_dict(self) -> dict:
        return {"column_name": self.column_name, "direction": self.direction}

    @classmethod
    def from_dict(cls, d: dict) -> ObjectiveConfig:
        return cls(column_name=d["column_name"], direction=d["direction"])


@dataclass
class StudyConfig:
    """Full configuration for an Optuna study session."""
    parameters: List[ParameterConfig] = field(default_factory=list)
    objectives: List[ObjectiveConfig] = field(default_factory=list)
    batch_size: int = 1
    n_batches: int = 10
    sampler_name: Literal["TPE", "NSGAII", "Random"] = "TPE"

    def to_dict(self) -> dict:
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "objectives": [o.to_dict() for o in self.objectives],
            "batch_size": self.batch_size,
            "n_batches": self.n_batches,
            "sampler_name": self.sampler_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StudyConfig:
        return cls(
            parameters=[ParameterConfig.from_dict(p) for p in d.get("parameters", [])],
            objectives=[ObjectiveConfig.from_dict(o) for o in d.get("objectives", [])],
            batch_size=d.get("batch_size", 1),
            n_batches=d.get("n_batches", 10),
            sampler_name=d.get("sampler_name", "TPE"),
        )
