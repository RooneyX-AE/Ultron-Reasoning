"""Composable verifiers for verifiable reasoning tasks."""
import ast
import math
import re


def exact_match(expected: str, actual: str) -> float:
    return float(expected.strip() == actual.strip())


def numeric_match(expected: float, actual: str, tolerance: float = 1e-6) -> float:
    try:
        value = float(actual.strip().replace(",", ""))
    except ValueError:
        return 0.0
    return float(math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance))


def extract_final_answer(text: str):
    match = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.I)
    return match.group(1).strip() if match else text.strip().splitlines()[-1].strip()


def safe_arithmetic(expression: str) -> float:
    """Evaluate a restricted arithmetic expression without builtins/functions."""
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Constant, ast.UnaryOp, ast.BinOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("unsupported expression")
    return float(eval(compile(tree, "<safe-arithmetic>", "eval"), {"__builtins__": {}}, {}))
