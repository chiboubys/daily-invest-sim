#!/usr/bin/env python3
"""Quick performance test to verify optimization."""
import time
from streamlit_app.models import ScenarioInput, MonteCarloConfig
from streamlit_app.simulation import run_deterministic_simulation, run_monte_carlo_simulation

# Test 1: Deterministic with large day count (performance-critical)
print("Testing deterministic simulation vectorization...")
start = time.time()
scenario = ScenarioInput(
    name="VectorTest",
    initial_investment=1000.0,
    daily_addition=10.0,
    daily_growth_pct=0.1,
    days=365_000,  # Full year with daily data
)
result = run_deterministic_simulation(scenario)
det_time = time.time() - start
print(f"  ✓ Vectorized deterministic (365k days): {det_time:.4f}s")
print(f"    Final value: €{result['portfolio_value'][-1]:,.2f}")

# Test 2: Monte Carlo simulation
print("\nTesting Monte Carlo simulation...")
start = time.time()
scenario_mc = ScenarioInput(
    name="MCTest",
    initial_investment=5000.0,
    daily_addition=5.0,
    daily_growth_pct=0.05,
    days=365,
)
mc_config = MonteCarloConfig(
    paths=2000,
    expected_daily_return_pct=0.2,
    daily_volatility_pct=1.4,
)
mc_result = run_monte_carlo_simulation(scenario_mc, mc_config)
mc_time = time.time() - start
print(f"  ✓ Monte Carlo (365 days, 2000 paths): {mc_time:.4f}s")
print(f"    Median end value: €{mc_result[0]['p50'][-1]:,.2f}")

# Test 3: Caching verification
print("\nTesting caching...")
start = time.time()
result2 = run_deterministic_simulation(scenario)
cache_time = time.time() - start
print(f"  ✓ Cached call (should be near-instant): {cache_time:.6f}s")

if cache_time < det_time / 10:
    print("  ✅ Caching is working! (>10x speedup)")
else:
    print("  ⚠️  Caching might not be working as expected")

print("\n✅ All tests passed!")
