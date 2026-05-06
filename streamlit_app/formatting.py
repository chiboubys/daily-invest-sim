from __future__ import annotations


def format_currency(value: float) -> str:
    return f"EUR {value:,.2f}"


def format_pct(value: float) -> str:
    return f"{value:,.2f}%"
