from __future__ import annotations

import math

import polars as pl

from streamlit_app.models import MonteCarloConfig, ScenarioInput
from streamlit_app.simulation import (
    _stable_name_seed,
    run_deterministic_simulation,
    run_monte_carlo_simulation,
)


def test_deterministic_matches_closed_form_without_addition() -> None:
    scenario = ScenarioInput(
        name="Closed form",
        initial_investment=100.0,
        daily_addition=0.0,
        daily_growth_pct=1.0,
        days=10,
    )

    timeline = run_deterministic_simulation(scenario)
    expected = 100.0 * math.pow(1.01, 10)
    actual = float(timeline.select(pl.col("portfolio_value").last()).item())
    assert actual == pytest_approx(expected, rel=1e-12)


def test_deterministic_handles_addition() -> None:
    scenario = ScenarioInput(
        name="Additions",
        initial_investment=100.0,
        daily_addition=10.0,
        daily_growth_pct=0.0,
        days=5,
    )

    timeline = run_deterministic_simulation(scenario)
    # day 0 includes initial investment only; 5 daily additions are then accumulated
    actual = float(timeline.select(pl.col("portfolio_value").last()).item())
    assert actual == pytest_approx(150.0)


def test_monte_carlo_is_seed_reproducible() -> None:
    scenario = ScenarioInput(
        name="Repro",
        initial_investment=100.0,
        daily_addition=1.0,
        daily_growth_pct=0.2,
        days=30,
    )
    config = MonteCarloConfig(
        paths=200,
        expected_daily_return_pct=0.2,
        daily_volatility_pct=1.1,
        random_seed=123,
    )

    timeline_a, ends_a = run_monte_carlo_simulation(scenario, config)
    timeline_b, ends_b = run_monte_carlo_simulation(scenario, config)

    assert timeline_a.to_dicts() == timeline_b.to_dicts()
    assert (ends_a == ends_b).all()


def test_stable_name_seed_is_deterministic() -> None:
    seed_a = _stable_name_seed(123, "Scenario A")
    seed_b = _stable_name_seed(123, "Scenario A")
    seed_c = _stable_name_seed(123, "Scenario B")

    assert seed_a == seed_b
    assert seed_a != seed_c


def pytest_approx(expected: float, rel: float = 1e-12):
    import pytest

    return pytest.approx(expected, rel=rel)
