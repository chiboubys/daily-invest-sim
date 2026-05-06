from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from streamlit_app.analytics import make_comparison_timeline, summarize_all
from streamlit_app.formatting import format_currency, format_pct
from streamlit_app.models import AppConfig, MonteCarloConfig, ScenarioInput, SimulationMode
from streamlit_app.simulation import SimulationArtifacts, run_scenario
from streamlit_app.state import (
    get_config,
    get_results,
    get_scenarios,
    initialize_state,
    set_config,
    set_results,
    set_scenarios,
)
from streamlit_app.visuals import (
    gain_decomposition_chart,
    line_evolution_chart,
    monte_carlo_band_chart,
    monte_carlo_distribution_histogram,
)


def _scenario_editor_table(scenarios: list[ScenarioInput]) -> pd.DataFrame:
    rows = [
        {
            "name": s.name,
            "initial_investment": s.initial_investment,
            "daily_addition": s.daily_addition,
            "daily_growth_pct": s.daily_growth_pct,
            "days": s.days,
        }
        for s in scenarios
    ]
    return pd.DataFrame(rows)


def _parse_scenarios(frame: pd.DataFrame) -> list[ScenarioInput]:
    parsed: list[ScenarioInput] = []
    seen: set[str] = set()

    for _, row in frame.iterrows():
        if pd.isna(row["name"]):
            continue

        scenario = ScenarioInput(
            name=str(row["name"]),
            initial_investment=float(row["initial_investment"]),
            daily_addition=float(row["daily_addition"]),
            daily_growth_pct=float(row["daily_growth_pct"]),
            days=int(row["days"]),
        )

        if scenario.name in seen:
            raise ValueError(f"Scenario name '{scenario.name}' is duplicated")

        seen.add(scenario.name)
        parsed.append(scenario)

    if not parsed:
        raise ValueError("At least one valid scenario is required")

    return parsed


def _render_config_form(config: AppConfig) -> AppConfig:
    with st.form("global_config"):
        st.subheader("Simulation Configuration")
        mode = st.segmented_control(
            "Simulation Mode",
            options=[SimulationMode.DETERMINISTIC, SimulationMode.MONTE_CARLO],
            default=config.mode,
            format_func=lambda x: x.value,
        )

        col1, col2 = st.columns(2)
        with col1:
            trading_days = st.number_input(
                "Trading days per year",
                min_value=1,
                max_value=366,
                value=config.trading_days_per_year,
                step=1,
            )
        with col2:
            mc_paths = st.number_input(
                "Monte Carlo paths",
                min_value=100,
                max_value=50_000,
                value=config.monte_carlo.paths,
                step=100,
                disabled=(mode == SimulationMode.DETERMINISTIC),
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            mc_return = st.number_input(
                "Expected daily return %",
                min_value=-50.0,
                max_value=50.0,
                value=float(config.monte_carlo.expected_daily_return_pct),
                step=0.05,
                disabled=(mode == SimulationMode.DETERMINISTIC),
            )
        with c2:
            mc_vol = st.number_input(
                "Daily volatility %",
                min_value=0.0,
                max_value=50.0,
                value=float(config.monte_carlo.daily_volatility_pct),
                step=0.05,
                disabled=(mode == SimulationMode.DETERMINISTIC),
            )
        with c3:
            mc_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=config.monte_carlo.random_seed,
                step=1,
                disabled=(mode == SimulationMode.DETERMINISTIC),
            )

        saved = st.form_submit_button("Apply Configuration", use_container_width=True)

    if not saved:
        return config

    selected_mode = mode or config.mode

    updated = AppConfig(
        mode=selected_mode,
        trading_days_per_year=int(trading_days),
        monte_carlo=MonteCarloConfig(
            paths=int(mc_paths),
            expected_daily_return_pct=float(mc_return),
            daily_volatility_pct=float(mc_vol),
            random_seed=int(mc_seed),
        ),
    )
    st.success("Configuration updated")
    return updated


def page_scenario_builder() -> None:
    st.title("Investment Scenario Builder")
    st.caption("Define one or many scenarios, then run deterministic or Monte Carlo simulation.")

    config = get_config()
    scenarios = get_scenarios()

    updated_config = _render_config_form(config)
    set_config(updated_config)

    st.subheader("Scenarios")
    edited = st.data_editor(
        _scenario_editor_table(scenarios),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Scenario"),
            "initial_investment": st.column_config.NumberColumn(
                "Initial Investment", min_value=0.0
            ),
            "daily_addition": st.column_config.NumberColumn("Daily Addition", min_value=0.0),
            "daily_growth_pct": st.column_config.NumberColumn("Daily Growth %", min_value=-99.0),
            "days": st.column_config.NumberColumn("Days", min_value=0, max_value=365_000, step=1),
        },
        key="scenario_editor",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        apply_button = st.button("Validate Scenarios", use_container_width=True)
    with col_b:
        run_button = st.button("Run Simulation", type="primary", use_container_width=True)

    if apply_button or run_button:
        try:
            parsed = _parse_scenarios(edited)
            set_scenarios(parsed)
            st.success(f"Validated {len(parsed)} scenarios")
        except (ValidationError, ValueError) as exc:
            st.error(f"Validation failed: {exc}")
            return

    if run_button:
        active = get_scenarios()
        current_config = get_config()

        artifacts: dict[str, SimulationArtifacts] = {}
        with st.status("Running simulations...", expanded=True) as status:
            for scenario in active:
                status.write(f"Simulating {scenario.name}")
                mc = (
                    current_config.monte_carlo
                    if current_config.mode == SimulationMode.MONTE_CARLO
                    else None
                )
                try:
                    artifacts[scenario.name] = run_scenario(scenario, mc)
                except ValueError as exc:
                    st.error(f"Scenario '{scenario.name}' failed: {exc}")
            status.update(label="Simulation completed", state="complete", expanded=False)

        if not artifacts:
            st.error("No scenario could be simulated. Please review inputs and retry.")
            return

        summary = summarize_all(active, artifacts, current_config)
        set_results(artifacts, summary)
        st.success("Results are ready in the Results and Comparison pages")


def page_results() -> None:
    st.title("Simulation Results")
    artifacts, summary = get_results()
    scenarios = get_scenarios()

    if summary is None or not artifacts:
        st.info("Run simulations from Scenario Builder first.")
        return

    summary_df = summary
    top = summary_df.row(0, named=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Final Value", format_currency(float(top["final_value"])))
    k2.metric("Best Gain", format_currency(float(top["absolute_gain"])))
    k3.metric("Best Gain %", format_pct(float(top["gain_pct"])))
    k4.metric("Best Annualized", format_pct(float(top["annualized_return_pct"])))

    comparison_timeline = make_comparison_timeline(scenarios, artifacts).to_pandas()
    st.plotly_chart(line_evolution_chart(comparison_timeline), use_container_width=True)

    selected = st.selectbox("Scenario details", options=[s.name for s in scenarios])
    selected_artifacts = artifacts[selected]

    st.plotly_chart(
        gain_decomposition_chart(selected_artifacts.deterministic_timeline.to_pandas(), selected),
        use_container_width=True,
    )

    if selected_artifacts.monte_carlo_percentiles is not None:
        st.plotly_chart(
            monte_carlo_band_chart(
                selected_artifacts.monte_carlo_percentiles.to_pandas(), selected
            ),
            use_container_width=True,
        )

    if selected_artifacts.monte_carlo_end_values is not None:
        st.plotly_chart(
            monte_carlo_distribution_histogram(selected_artifacts.monte_carlo_end_values, selected),
            use_container_width=True,
        )

    st.subheader("Summary")
    st.dataframe(summary_df.to_pandas(), use_container_width=True, hide_index=True)


def page_comparison() -> None:
    st.title("Scenario Comparison")
    artifacts, summary = get_results()

    if summary is None or not artifacts:
        st.info("Run simulations from Scenario Builder first.")
        return

    summary_df = summary.with_row_index(name="rank", offset=1).select(
        [
            "rank",
            "scenario",
            "invested_capital",
            "final_value",
            "absolute_gain",
            "gain_pct",
            "annualized_return_pct",
        ]
    )

    st.dataframe(summary_df.to_pandas(), use_container_width=True, hide_index=True)

    csv_data = summary_df.write_csv().encode("utf-8")
    st.download_button(
        label="Download comparison as CSV",
        data=csv_data,
        file_name="scenario_comparison.csv",
        mime="text/csv",
    )


def page_methodology() -> None:
    st.title("Methodology")
    st.markdown(
        """
### Deterministic model

Each day follows:

- contribution first
- then compounding growth

Formula:

$V_t = (V_{t-1} + A) \times (1 + r)$

Where:

- $V_t$: portfolio value at day $t$
- $A$: daily addition
- $r$: daily growth rate

### Monte Carlo model

Daily return is sampled from a normal distribution and applied with the same contribution order.

- configurable expected daily return
- configurable daily volatility
- reproducible seed
- percentile confidence bands and endpoint distribution

### Notes

- This dashboard is educational and analytic, not financial advice.
- Markets can deviate significantly from normal assumptions.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Investment Evolution Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()

    pages = st.navigation(
        {
            "Simulation": [
                st.Page(page_scenario_builder, title="Scenario Builder", icon="🧪"),
                st.Page(page_results, title="Results", icon="📈"),
                st.Page(page_comparison, title="Comparison", icon="⚖️"),
                st.Page(page_methodology, title="Methodology", icon="📘"),
            ]
        }
    )
    pages.run()


if __name__ == "__main__":
    main()
