from __future__ import annotations

from typing import cast

import polars as pl
import streamlit as st

from .models import AppConfig, ScenarioInput
from .simulation import SimulationArtifacts


def _default_scenarios() -> list[ScenarioInput]:
    return [
        ScenarioInput(
            name="Baseline",
            initial_investment=1000.0,
            daily_addition=10.0,
            daily_growth_pct=0.25,
            days=218,
        ),
        ScenarioInput(
            name="Aggressive",
            initial_investment=1000.0,
            daily_addition=15.0,
            daily_growth_pct=0.35,
            days=218,
        ),
    ]


def initialize_state() -> None:
    if "app_config" not in st.session_state:
        st.session_state.app_config = AppConfig()
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = _default_scenarios()
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = {}
    if "summary" not in st.session_state:
        st.session_state.summary = None


def get_config() -> AppConfig:
    return cast(AppConfig, st.session_state.app_config)


def set_config(config: AppConfig) -> None:
    st.session_state.app_config = config


def get_scenarios() -> list[ScenarioInput]:
    return cast(list[ScenarioInput], st.session_state.scenarios)


def set_scenarios(scenarios: list[ScenarioInput]) -> None:
    st.session_state.scenarios = scenarios


def set_results(
    artifacts: dict[str, SimulationArtifacts],
    summary: pl.DataFrame,
) -> None:
    st.session_state.artifacts = artifacts
    st.session_state.summary = summary


def get_results() -> tuple[dict[str, SimulationArtifacts], pl.DataFrame | None]:
    return (
        cast(dict[str, SimulationArtifacts], st.session_state.artifacts),
        cast(pl.DataFrame | None, st.session_state.summary),
    )
