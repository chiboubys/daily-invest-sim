from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from streamlit_app.models import ScenarioInput
from streamlit_app.simulation import run_deterministic_simulation


@given(
    initial=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    addition=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    growth=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    days=st.integers(min_value=1, max_value=500),
)
def test_non_negative_growth_and_addition_are_monotonic(initial, addition, growth, days) -> None:
    scenario = ScenarioInput(
        name="Property scenario",
        initial_investment=initial,
        daily_addition=addition,
        daily_growth_pct=growth,
        days=days,
    )

    timeline = run_deterministic_simulation(scenario)
    values = timeline.get_column("portfolio_value").to_numpy()
    assert (values[1:] >= values[:-1]).all()
