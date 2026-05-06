from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
import polars as pl
import streamlit as st

from .models import MonteCarloConfig, ScenarioInput


@dataclass(slots=True, frozen=True)
class SimulationArtifacts:
    deterministic_timeline: pl.DataFrame
    monte_carlo_percentiles: pl.DataFrame | None
    monte_carlo_end_values: np.ndarray | None


def _assert_finite_array(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"Simulation generated non-finite values in '{name}'")


def _stable_name_seed(base_seed: int, scenario_name: str) -> int:
    digest = blake2b(scenario_name.encode("utf-8"), digest_size=8).digest()
    name_component = int.from_bytes(digest, byteorder="little", signed=False)
    return int((base_seed + name_component) % 2_147_483_647)


def _coerce_growth_multiplier(daily_growth_pct: float) -> float:
    multiplier = 1.0 + daily_growth_pct / 100.0
    if multiplier <= 0.0:
        raise ValueError("Daily growth percentage leads to invalid multiplier <= 0")
    return multiplier


@st.cache_data
def run_deterministic_simulation(scenario: ScenarioInput) -> pl.DataFrame:
    days = scenario.days
    day_idx = np.arange(days + 1, dtype=np.int32)
    contributions = np.zeros(days + 1, dtype=np.float64)
    
    # Vectorized contributions: initial + daily_addition * day_number
    contributions = scenario.initial_investment + scenario.daily_addition * day_idx

    multiplier = _coerce_growth_multiplier(scenario.daily_growth_pct)

    # Vectorized portfolio values using geometric series
    # V_t = (V_0 + A) * m^t - A * (m^(t+1) - m) / (m - 1)
    # But we use iterative approach with numpy cumulative multiplication
    if days == 0:
        values = np.array([scenario.initial_investment], dtype=np.float64)
    else:
        # Vectorized approach: compute multiplied contributions
        powers = np.power(multiplier, day_idx)  # m^0, m^1, ..., m^days
        
        # For each day: V_t = C_0 * m^t + A * (m^t - 1) / (m - 1)
        if multiplier == 1.0:
            # Special case: no growth
            values = contributions.copy()
        else:
            values = scenario.initial_investment * powers + scenario.daily_addition * (powers - 1.0) / (multiplier - 1.0)

    growth_gain = values - contributions
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_pct = np.where(contributions > 0.0, growth_gain / contributions * 100.0, 0.0)

    _assert_finite_array("deterministic_values", values)
    _assert_finite_array("deterministic_contributions", contributions)
    _assert_finite_array("deterministic_gain_pct", gain_pct)

    timeline = pl.DataFrame(
        {
            "day": day_idx,
            "invested_capital": contributions,
            "portfolio_value": values,
            "growth_gain": growth_gain,
            "gain_pct": gain_pct,
        }
    )
    return timeline


@st.cache_data
def run_monte_carlo_simulation(
    scenario: ScenarioInput,
    config: MonteCarloConfig,
) -> tuple[pl.DataFrame, np.ndarray]:
    if scenario.days == 0:
        start = np.array([scenario.initial_investment], dtype=np.float64)
        summary = pl.DataFrame(
            {
                "day": np.array([0], dtype=np.int32),
                "p_low": start,
                "p50": start,
                "p_high": start,
            }
        )
        return summary, start

    seed = _stable_name_seed(config.random_seed, scenario.name)
    rng = np.random.default_rng(seed)

    mean = config.expected_daily_return_pct / 100.0
    std = config.daily_volatility_pct / 100.0
    daily_returns = rng.normal(loc=mean, scale=std, size=(config.paths, scenario.days)).astype(
        np.float64
    )

    values = np.empty((config.paths, scenario.days + 1), dtype=np.float64)
    values[:, 0] = scenario.initial_investment

    for day in range(1, scenario.days + 1):
        prev = values[:, day - 1] + scenario.daily_addition
        values[:, day] = np.maximum(prev * (1.0 + daily_returns[:, day - 1]), 0.0)

    qs = np.quantile(values, [config.low_quantile, 0.5, config.high_quantile], axis=0)
    _assert_finite_array("monte_carlo_values", values)
    _assert_finite_array("monte_carlo_quantiles", qs)
    _assert_finite_array("monte_carlo_end_values", values[:, -1])

    timeline = pl.DataFrame(
        {
            "day": np.arange(scenario.days + 1, dtype=np.int32),
            "p_low": qs[0],
            "p50": qs[1],
            "p_high": qs[2],
        }
    )
    return timeline, values[:, -1]


def run_scenario(
    scenario: ScenarioInput,
    monte_carlo: MonteCarloConfig | None,
) -> SimulationArtifacts:
    deterministic = run_deterministic_simulation(scenario)
    if monte_carlo is None:
        return SimulationArtifacts(
            deterministic_timeline=deterministic,
            monte_carlo_percentiles=None,
            monte_carlo_end_values=None,
        )

    percentiles, end_values = run_monte_carlo_simulation(scenario, monte_carlo)
    return SimulationArtifacts(
        deterministic_timeline=deterministic,
        monte_carlo_percentiles=percentiles,
        monte_carlo_end_values=end_values,
    )
