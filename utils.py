from __future__ import annotations

import base64
import json
from typing import Dict, Tuple



def validate_weight_inputs(weights_pct: Dict[str, float]) -> Tuple[bool, str]:
    total = round(sum(weights_pct.values()), 4)
    if not weights_pct:
        return False, "Enter at least one weight."
    if any(v < 0 for v in weights_pct.values()):
        return False, "Weights cannot be negative."
    if abs(total - 100.0) > 0.01:
        return False, f"Weights currently sum to {total:.2f}%. They must sum to 100%."
    return True, f"Weights sum correctly to {total:.2f}%."



def normalize_weights_from_percent(weights_pct: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights_pct.values())
    if total == 0:
        raise ValueError("Weight total cannot be zero.")
    return {k: v / total for k, v in weights_pct.items()}



def validate_date_range(date_range) -> Tuple[bool, str]:
    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        return False, "Please select both a start date and an end date."
    start_date, end_date = date_range
    if start_date >= end_date:
        return False, "Start date must be earlier than end date."
    return True, "Date range looks valid."



def format_pct(value: float) -> str:
    return f"{value:.2%}"



def safe_json_download_link(data: dict, label: str, filename: str) -> str:
    payload = json.dumps(data, indent=2).encode("utf-8")
    b64 = base64.b64encode(payload).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">{label}</a>'
