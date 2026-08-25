"""
Phase 1 — Data Models
Defines all shared dataclasses and enums used across the application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Tuple


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
class ParameterConstraint:
    """
    A user-defined constraint relating one or more parameters.

    Examples
    --------
    Compositional equality (fractions must sum to 1):
        expression = "CsPbI + FAPbI + MAPbI"
        operator   = "="
        target     = 1.0
        residual_param = "MAPbI"   # auto-computed; removed from Optuna's search space

    Inequality (processing budget):
        expression = "temp * time"
        operator   = "<="
        target     = 50000.0
        residual_param = ""         # not applicable for inequalities

    The expression is a safe arithmetic formula that may contain:
        parameter names, numeric literals, +  -  *  /  **  ( )
    It is evaluated with the restricted ``_eval_expr`` helper — no
    arbitrary Python code is executed.
    """
    name: str                                      # display name, e.g. "Composition"
    expression: str                                # formula, e.g. "CsPbI + FAPbI + MAPbI"
    operator: Literal["=", "<=", ">="]             # constraint type
    target: float = 1.0                            # right-hand-side value
    residual_param: str = ""                       # for "=": which param is auto-computed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "expression": self.expression,
            "operator": self.operator,
            "target": self.target,
            "residual_param": self.residual_param,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParameterConstraint:
        return cls(
            name=d["name"],
            expression=d["expression"],
            operator=d.get("operator", "="),
            target=float(d.get("target", 1.0)),
            residual_param=d.get("residual_param", ""),
        )

    def validate_expression(self, param_names: List[str]) -> List[str]:
        """
        Return a list of error strings.  Empty list = valid.
        Checks that the expression parses correctly and only references
        known parameter names.
        """
        import ast
        errors: List[str] = []
        if not self.expression.strip():
            errors.append("Expression is empty.")
            return errors
        try:
            tree = ast.parse(self.expression.strip(), mode="eval")
        except SyntaxError as exc:
            errors.append(f"Syntax error in expression: {exc}")
            return errors
        # Walk the AST and allow only safe node types
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
            ast.Constant, ast.Name, ast.Load,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                errors.append(
                    f"Unsupported operation in expression: {type(node).__name__}. "
                    "Only +  −  *  /  **  and parameter names are allowed."
                )
                return errors
        # Check all Name nodes are known parameter names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in param_names:
                errors.append(
                    f"Unknown parameter '{node.id}' in expression. "
                    f"Known parameters: {param_names}"
                )
        if self.operator == "=" and self.residual_param:
            if self.residual_param not in param_names:
                errors.append(
                    f"Residual parameter '{self.residual_param}' is not a known parameter."
                )
        return errors

    @staticmethod
    def eval_expr(expression: str, param_values: dict) -> float:
        """
        Safely evaluate an arithmetic expression with the given parameter values.
        Only +  −  *  /  **  and numeric constants are permitted.
        Raises ValueError on any error.
        """
        import ast, operator as _op

        _BINOPS = {
            ast.Add:  _op.add,
            ast.Sub:  _op.sub,
            ast.Mult: _op.mul,
            ast.Div:  _op.truediv,
            ast.Pow:  _op.pow,
        }
        _UNOPS = {
            ast.USub: _op.neg,
            ast.UAdd: _op.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant):
                return float(node.n if hasattr(node, 'n') else node.value)
            if isinstance(node, ast.Name):
                if node.id not in param_values:
                    raise ValueError(f"Unknown parameter '{node.id}'")
                return float(param_values[node.id])
            if isinstance(node, ast.BinOp):
                op_fn = _BINOPS.get(type(node.op))
                if op_fn is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op_fn(_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                op_fn = _UNOPS.get(type(node.op))
                if op_fn is None:
                    raise ValueError(f"Unsupported unary operator")
                return op_fn(_eval(node.operand))
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
            return _eval(tree.body)
        except Exception as exc:
            raise ValueError(f"Cannot evaluate '{expression}': {exc}") from exc

    def is_satisfied(self, param_values: dict) -> Tuple[bool, float]:
        """
        Evaluate the constraint against *param_values*.

        Returns
        -------
        (satisfied, violation)
            satisfied : True if the constraint holds.
            violation : signed violation amount (positive = infeasible).
                        For "=": |actual − target|
                        For "<=": actual − target  (positive = violated)
                        For ">=": target − actual  (positive = violated)
        """
        try:
            actual = self.eval_expr(self.expression, param_values)
        except ValueError:
            return False, float("inf")

        if self.operator == "=":
            diff = abs(actual - self.target)
            return diff < 1e-9, diff
        elif self.operator == "<=":
            v = actual - self.target
            return v <= 0, v
        else:  # ">="
            v = self.target - actual
            return v <= 0, v


@dataclass
class StudyConfig:
    """Full configuration for an Optuna study session."""
    parameters: List[ParameterConfig] = field(default_factory=list)
    objectives: List[ObjectiveConfig] = field(default_factory=list)
    constraints: List[ParameterConstraint] = field(default_factory=list)
    batch_size: int = 1
    n_batches: int = 10
    sampler_name: Literal["TPE", "NSGAII", "Random"] = "TPE"

    def to_dict(self) -> dict:
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "batch_size": self.batch_size,
            "n_batches": self.n_batches,
            "sampler_name": self.sampler_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StudyConfig:
        return cls(
            parameters=[ParameterConfig.from_dict(p) for p in d.get("parameters", [])],
            objectives=[ObjectiveConfig.from_dict(o) for o in d.get("objectives", [])],
            constraints=[ParameterConstraint.from_dict(c) for c in d.get("constraints", [])],
            batch_size=d.get("batch_size", 1),
            n_batches=d.get("n_batches", 10),
            sampler_name=d.get("sampler_name", "TPE"),
        )
