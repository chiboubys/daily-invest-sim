from __future__ import annotations

import polars as pl

from streamlit_app.analytics import make_comparison_timeline, summarize_all
from streamlit_app.models import AppConfig, MonteCarloConfig, ScenarioInput, SimulationMode
from streamlit_app.simulation import run_scenario


def _config() -> AppConfig:
    return AppConfig(
        mode=SimulationMode.DETERMINISTIC,
        trading_days_per_year=218,
        monte_carlo=MonteCarloConfig(
            paths=200,
            expected_daily_return_pct=0.2,
            daily_volatility_pct=1.2,
            random_seed=42,
        ),
    )


def test_summarize_all_orders_by_final_value() -> None:
    slow = ScenarioInput(
        name="Slow",
        initial_investment=100.0,
        daily_addition=0.0,
        daily_growth_pct=0.1,
        days=10,
    )
    fast = ScenarioInput(
        name="Fast",
        initial_investment=100.0,
        daily_addition=0.0,
        daily_growth_pct=0.3,
        days=10,
    )

    artifacts = {
        slow.name: run_scenario(slow, None),
        fast.name: run_scenario(fast, None),
    }

    summary = summarize_all([slow, fast], artifacts, _config())

    assert isinstance(summary, pl.DataFrame)
    assert summary.height == 2
    assert summary.row(0, named=True)["scenario"] == "Fast"


def test_make_comparison_timeline_contains_all_scenarios() -> None:
    scenario_a = ScenarioInput(
        name="A",
        initial_investment=100.0,
        daily_addition=2.0,
        daily_growth_pct=0.2,
        days=5,
    )
    scenario_b = ScenarioInput(
        name="B",
        initial_investment=100.0,
        daily_addition=3.0,
        daily_growth_pct=0.2,
        days=5,
    )

    artifacts = {
        scenario_a.name: run_scenario(scenario_a, None),
        scenario_b.name: run_scenario(scenario_b, None),
    }

    timeline = make_comparison_timeline([scenario_a, scenario_b], artifacts)

    assert isinstance(timeline, pl.DataFrame)
    assert timeline.height == 12
    names = set(timeline.get_column("scenario").to_list())
    assert names == {"A", "B"}
