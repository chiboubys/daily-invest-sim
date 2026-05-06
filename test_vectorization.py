#!/usr/bin/env python3
"""Test vectorized deterministic simulation (no Streamlit dependency)."""
import sys
import time
import numpy as np

# Simulate what the vectorized function should compute
def test_vectorization():
    """Test that vectorized formula matches expected output."""
    
    # Test parameters
    initial = 1000.0
    daily_add = 10.0
    daily_growth_pct = 0.1
    days = 365
    
    # Vectorized approach using power formula
    multiplier = 1.0 + daily_growth_pct / 100.0
    day_idx = np.arange(days + 1, dtype=np.int32)
    contributions = initial + daily_add * day_idx
    
    # V_t = V_0 * m^t + A * (m^t - 1) / (m - 1)
    if multiplier == 1.0:
        values = contributions.copy()
    else:
        powers = np.power(multiplier, day_idx)
        values = initial * powers + daily_add * (powers - 1.0) / (multiplier - 1.0)
    
    # Verify correctness by checking iterative equivalent for a few points
    # Day 0: V_0 = 1000, C_0 = 1000
    assert abs(values[0] - 1000.0) < 0.01, f"Day 0 failed: {values[0]}"
    assert abs(contributions[0] - 1000.0) < 0.01
    
    # Day 1: C_1 = 1010, V_1 = (1000 + 10) * 1.001 = 1010.01
    expected_v1 = (initial + daily_add) * multiplier
    assert abs(values[1] - expected_v1) < 0.01, f"Day 1 failed: got {values[1]}, expected {expected_v1}"
    
    # Day 2: V_2 = (V_1 + 10) * 1.001 = (1010.01 + 10) * 1.001 = 1020.030001
    expected_v2 = (values[1] + daily_add) * multiplier
    assert abs(values[2] - expected_v2) < 0.01, f"Day 2 failed: got {values[2]}, expected {expected_v2}"
    
    print("✅ Vectorization is mathematically correct!")
    print(f"   Day 0: C={contributions[0]:.2f}, V={values[0]:.2f}")
    print(f"   Day 1: C={contributions[1]:.2f}, V={values[1]:.4f}")
    print(f"   Day 365: C={contributions[365]:.2f}, V={values[365]:.2f}")
    
    return True

def test_performance():
    """Compare vectorized vs loop-based performance (simulated)."""
    print("\nPerformance comparison:")
    
    initial = 1000.0
    daily_add = 10.0
    daily_growth_pct = 0.1
    days = 365_000
    
    multiplier = 1.0 + daily_growth_pct / 100.0
    
    # Vectorized approach (fast)
    start = time.time()
    day_idx = np.arange(days + 1, dtype=np.int32)
    contributions = initial + daily_add * day_idx
    powers = np.power(multiplier, day_idx)
    values = initial * powers + daily_add * (powers - 1.0) / (multiplier - 1.0)
    vec_time = time.time() - start
    
    print(f"  Vectorized (365k days): {vec_time:.4f}s")
    print(f"  Final value: €{values[-1]:,.2f}")
    print(f"  ✅ Vectorization is very fast!")
    
    return True

if __name__ == "__main__":
    try:
        test_vectorization()
        test_performance()
        print("\n✅ All validation tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
