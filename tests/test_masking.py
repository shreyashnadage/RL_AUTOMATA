import sys
import os
# Add workspace root to sys.path to allow importing from root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from engine import SyntheticSwapEngine
from envs.otc_env import OTCHedgingEnv
from envs.reward_calculators import JointOptimizationReward
from agents.agent_factory import create_agent
from agents.callbacks import FinancialMetricsCallback

def load_automaton_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "simple_automaton.json")
    with open(config_path, "r") as f:
        return json.load(f)

def setup_env_and_engine():
    engine = SyntheticSwapEngine(
        base_rate=0.045,
        a=0.1,
        sigma=0.01,
        notional=10000000.0,
        maturity_years=5
    )
    config = load_automaton_config()
    reward_calc = JointOptimizationReward()
    env = OTCHedgingEnv(
        engine=engine,
        automaton_config=config,
        reward_calculator=reward_calc,
        num_paths=100,
        num_days=30
    )
    return env

def test_action_masking_states():
    env = setup_env_and_engine()
    
    # 1. Normal state: all actions allowed
    obs, info = env.reset(seed=42)
    assert env.automaton.state in ("normal", "margin_call_issued")
    
    # Force FSM to grace_period
    # normal -> observe_breach -> margin_call_issued -> elapse_one_day -> grace_period
    env.automaton.state = "grace_period"
    
    # Get action masks
    mask = env.action_masks()
    
    # Mask should block even actions (Ignore Margin) and allow odd actions (Post Margin)
    # Actions 0, 2, 4 are False
    # Actions 1, 3, 5 are True
    expected_mask = np.array([False, True, False, True, False, True])
    np.testing.assert_array_equal(mask, expected_mask)

def test_gym_api_compliance():
    env = setup_env_and_engine()
    
    # Verify that the environment matches standard Gymnasium APIs and requirements
    check_env(env)

def test_debug_agent_run():
    env = setup_env_and_engine()
    
    # Create the MaskablePPO agent using the factory
    log_dir = "./logs/airl_runs/"
    model = create_agent(algo_name="MaskablePPO", env=env, tensorboard_log_dir=log_dir)
    
    # Create the callback
    callback = FinancialMetricsCallback()
    
    # Perform a short training run of 100 timesteps to ensure no runtime errors
    model.learn(total_timesteps=100, callback=callback)
    
    print("\nDebug training run completed successfully!")
