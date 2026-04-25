from __future__ import annotations

from typing import Optional, Set


VALID_MODEL_TYPES: Set[str] = {
    "original",
    "enhanced_cnn_lstm",
    "resnet_lstm",
    "transformer",
    "trained_model",
}


def sanitize_model_type(model_type: Optional[str]) -> Optional[str]:
    """
    Ensure model_type is one of the supported architecture identifiers.

    This is used as a security boundary (prevents path traversal) and as an
    API contract validator (consistent behavior across endpoints/services).
    """
    if model_type is None:
        return None
    normalized = str(model_type).strip()
    if normalized in VALID_MODEL_TYPES:
        return normalized
    raise ValueError(f"Invalid model_type. Must be one of: {sorted(VALID_MODEL_TYPES)}")

