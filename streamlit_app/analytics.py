from __future__ import annotations

import math
from collections.abc import Iterable

import polars as pl

from .models import AppConfig, ScenarioInput, ScenarioSummary
from .simulation import SimulationArtifacts


def build_summary(
    scenario: ScenarioInput,
    artifacts: SimulationArtifacts,
    config: AppConfig,
) -> ScenarioSummary:
    last_row = artifacts.deterministic_timeline.tail(1).to_dicts()[0]
    invested = float(last_row["invested_capital"])
    final_value = float(last_row["portfolio_value"])
    absolute_gain = final_value - invested
    gain_pct = (absolute_gain / invested * 100.0) if invested > 0 else 0.0

    years = scenario.days / max(config.trading_days_per_year, 1)
    annualized = 0.0
    if years > 0 and invested > 0 and final_value > 0:
        annualized = (math.pow(final_value / invested, 1.0 / years) - 1.0) * 100.0

    return ScenarioSummary(
        scenario=scenario.name,
        invested_capital=invested,
        final_value=final_value,
        absolute_gain=absolute_gain,
        gain_pct=gain_pct,
        annualized_return_pct=annualized,
    )


def summarize_all(
    scenarios: Iterable[ScenarioInput],
    artifacts_by_name: dict[str, SimulationArtifacts],
    config: AppConfig,
) -> pl.DataFrame:
    rows = []
    for scenario in scenarios:
        artifacts = artifacts_by_name[scenario.name]
        rows.append(build_summary(scenario, artifacts, config).model_dump())

    frame = pl.DataFrame(rows)
    if frame.is_empty():
        return frame

    return frame.sort("final_value", descending=True)


def make_comparison_timeline(
    scenarios: Iterable[ScenarioInput],
    artifacts_by_name: dict[str, SimulationArtifacts],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for scenario in scenarios:
        timeline = artifacts_by_name[scenario.name].deterministic_timeline.with_columns(
            pl.lit(scenario.name).alias("scenario")
        )
        frames.append(timeline)
    if not frames:
        return pl.DataFrame(
            schema={
                "day": pl.Int32,
                "invested_capital": pl.Float64,
                "portfolio_value": pl.Float64,
                "growth_gain": pl.Float64,
                "gain_pct": pl.Float64,
                "scenario": pl.String,
            }
        )
    return pl.concat(frames, how="vertical")
