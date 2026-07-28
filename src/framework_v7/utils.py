from __future__ import annotations

"""General helper functions for FRAMEWORK V7.

The current project keeps most helpers in domain-specific modules such as
``profiling`` and ``paths``. This module exists as a stable import point for
future generic utilities that do not belong to a specific business layer.
"""


def format_count(value: int | float) -> str:
    """Format a numeric count for UI or CLI output.

    Args:
        value: Numeric value to format.

    Returns:
        String with thousands separators.
    """

    return f"{value:,.0f}"
