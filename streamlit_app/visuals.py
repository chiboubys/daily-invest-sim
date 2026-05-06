from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_evolution_chart(timeline: pd.DataFrame) -> go.Figure:
    fig = px.line(
        timeline,
        x="day",
        y="portfolio_value",
        color="scenario",
        template="plotly_white",
        title="Portfolio Evolution by Scenario",
    )
    fig.update_layout(legend_title_text="Scenario", margin={"l": 20, "r": 20, "t": 50, "b": 20})
    fig.update_yaxes(title_text="Portfolio Value")
    return fig


def gain_decomposition_chart(single_timeline: pd.DataFrame, scenario_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=single_timeline["day"],
            y=single_timeline["invested_capital"],
            mode="lines",
            name="Invested Capital",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=single_timeline["day"],
            y=single_timeline["portfolio_value"],
            mode="lines",
            name="Portfolio Value",
            fill="tonexty",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Gain Decomposition - {scenario_name}",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def monte_carlo_band_chart(percentiles: pd.DataFrame, scenario_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=percentiles["day"],
            y=percentiles["p_low"],
            mode="lines",
            line={"color": "rgba(10, 122, 100, 0.2)"},
            name="Low quantile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentiles["day"],
            y=percentiles["p_high"],
            mode="lines",
            fill="tonexty",
            line={"color": "rgba(10, 122, 100, 0.2)"},
            name="High quantile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentiles["day"],
            y=percentiles["p50"],
            mode="lines",
            line={"color": "rgba(0, 112, 243, 1)"},
            name="Median",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Monte Carlo Confidence Band - {scenario_name}",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def monte_carlo_distribution_histogram(
    end_values: np.ndarray,
    scenario_name: str,
) -> go.Figure:
    fig = px.histogram(
        x=end_values,
        nbins=50,
        template="plotly_white",
        title=f"Monte Carlo End Value Distribution - {scenario_name}",
    )
    fig.update_xaxes(title_text="End Portfolio Value")
    fig.update_yaxes(title_text="Frequency")
    return fig
