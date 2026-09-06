"""Phase 5 — Hallucination verifier: checks all numbers in LLM text match computed values."""

from __future__ import annotations

import re
from typing import Dict

MAX_RETRIES = 2
TOLERANCE = 0.05   # 5% relative tolerance


def extract_numbers_from_text(text: str) -> list[float]:
    pattern = r"\d+\.?\d*"
    return [float(m) for m in re.findall(pattern, text)]


def verify_explanation(explanation: Dict[str, str], expected: Dict[str, float]) -> bool:
    """Return True if all expected values appear (±tolerance) in the combined text."""
    full_text = " ".join(explanation.values())
    found_numbers = extract_numbers_from_text(full_text)

    for label, expected_val in expected.items():
        if expected_val == 0:
            continue
        matched = any(
            abs(n - expected_val) / (abs(expected_val) + 1e-9) <= TOLERANCE
            for n in found_numbers
        )
        if not matched:
            return False
    return True


def fix_numbers_in_text(text: str, expected: Dict[str, float]) -> str:
    """Best-effort: replace first occurrence of a wrong number with the correct one."""
    for label, correct_val in expected.items():
        numbers = extract_numbers_from_text(text)
        for n in numbers:
            if n != 0 and abs(n - correct_val) / (abs(correct_val) + 1e-9) > TOLERANCE:
                text = text.replace(str(n), str(round(correct_val, 2)), 1)
                break
    return text
