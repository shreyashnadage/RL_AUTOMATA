import pytest
import time
import numpy as np
from engine import SyntheticSwapEngine

def test_engine_initialization_and_shapes():
    # Parameters
    base_rate = 0.045
    a = 0.1
    sigma = 0.01
    notional = 10000000.0
    maturity_years = 5

    engine = SyntheticSwapEngine(
        base_rate=base_rate,
        a=a,
        sigma=sigma,
        notional=notional,
        maturity_years=maturity_years
    )

    num_paths = 10
    num_days = 30

    res = engine.generate_trajectory_batch(num_paths=num_paths, num_days=num_days)

    # Check keys
    assert "time_grid" in res
    assert "short_rates" in res
    assert "mtm_profiles" in res
    assert "exposure_profiles" in res

    # Check dimensions
    assert res["time_grid"].shape == (num_days,)
    assert res["short_rates"].shape == (num_paths, num_days)
    assert res["mtm_profiles"].shape == (num_paths, num_days)
    assert res["exposure_profiles"].shape == (num_paths, num_days)

def test_mathematical_sanity_at_t0():
    # At t=0, the MtM value of the par swap must be approximately 0.0
    base_rate = 0.045
    a = 0.1
    sigma = 0.01
    notional = 10000000.0
    maturity_years = 5

    engine = SyntheticSwapEngine(
        base_rate=base_rate,
        a=a,
        sigma=sigma,
        notional=notional,
        maturity_years=maturity_years
    )

    # We evaluate for 10 paths and 1 day (just t=0)
    res = engine.generate_trajectory_batch(num_paths=10, num_days=1)
    
    # Check that t=0 MtM is very close to 0
    mtm_t0 = res["mtm_profiles"][:, 0]
    np.testing.assert_allclose(mtm_t0, 0.0, atol=1e-4)

def test_positivity_constraint():
    # All values in exposure_profiles must be >= 0.0
    base_rate = 0.045
    a = 0.1
    sigma = 0.01
    notional = 10000000.0
    maturity_years = 5

    engine = SyntheticSwapEngine(
        base_rate=base_rate,
        a=a,
        sigma=sigma,
        notional=notional,
        maturity_years=maturity_years
    )

    res = engine.generate_trajectory_batch(num_paths=50, num_days=30)
    exposure = res["exposure_profiles"]
    
    assert np.all(exposure >= 0.0)
    
    # Check that exposure is exactly max(mtm, 0)
    expected_exposure = np.maximum(res["mtm_profiles"], 0.0)
    np.testing.assert_allclose(exposure, expected_exposure, atol=1e-12)

def test_reproducibility():
    # Running with the same seed twice must yield identical values
    base_rate = 0.045
    a = 0.1
    sigma = 0.01
    notional = 10000000.0
    maturity_years = 5

    engine = SyntheticSwapEngine(
        base_rate=base_rate,
        a=a,
        sigma=sigma,
        notional=notional,
        maturity_years=maturity_years
    )

    res1 = engine.generate_trajectory_batch(num_paths=20, num_days=15, seed=100)
    res2 = engine.generate_trajectory_batch(num_paths=20, num_days=15, seed=100)
    res3 = engine.generate_trajectory_batch(num_paths=20, num_days=15, seed=200)

    # res1 and res2 must be identical
    np.testing.assert_array_equal(res1["short_rates"], res2["short_rates"])
    np.testing.assert_array_equal(res1["mtm_profiles"], res2["mtm_profiles"])
    np.testing.assert_array_equal(res1["exposure_profiles"], res2["exposure_profiles"])

    # res1 and res3 must NOT be identical (different seeds)
    assert not np.array_equal(res1["short_rates"], res3["short_rates"])

def test_performance_requirement():
    # Batch run of 100 paths and 30 days must evaluate in under 2.0 seconds
    base_rate = 0.045
    a = 0.1
    sigma = 0.01
    notional = 10000000.0
    maturity_years = 5

    engine = SyntheticSwapEngine(
        base_rate=base_rate,
        a=a,
        sigma=sigma,
        notional=notional,
        maturity_years=maturity_years
    )

    start_time = time.time()
    res = engine.generate_trajectory_batch(num_paths=100, num_days=30)
    elapsed = time.time() - start_time

    print(f"\nExecution speed: {elapsed:.4f} seconds")
    assert elapsed < 2.0
