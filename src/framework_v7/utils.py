"""General UI helpers for FRAMEWORK V7."""

from __future__ import annotations

import pandas as pd


def format_metric_date(value: object) -> str:
    """Format a date-like value for Streamlit metric widgets.

    Args:
        value: Date-like object, string or pandas timestamp.

    Returns:
        Human-readable date in ``YYYY-MM-DD`` format. Returns ``"-"`` when the
        value cannot be parsed as a valid date.
    """

    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return "-"
    return str(date_value.date())
