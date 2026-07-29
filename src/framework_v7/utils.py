from __future__ import annotations

"""General helper functions for FRAMEWORK V7.

The current project keeps most helpers in domain-specific modules such as
``profiling`` and ``paths``. This module exists as a stable import point for
future generic utilities that do not belong to a specific business layer.
"""

import pandas as pd


def format_count(value: int | float) -> str:
    """Format a numeric count for UI or CLI output.

    Args:
        value: Numeric value to format.

    Returns:
        String with thousands separators.
    """

    return f"{value:,.0f}"


def format_metric_date(value: object) -> str:
    """Format date-like values for Streamlit metric widgets.

    Args:
        value: Date-like value, pandas timestamp, or null-like value.

    Returns:
        ISO date string, or "-" when the value cannot be parsed.
    """

    if pd.isna(value):
        return "-"

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "-"

    return timestamp.date().isoformat()
