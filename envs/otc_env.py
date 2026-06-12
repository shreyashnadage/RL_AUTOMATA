import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from engine import SyntheticSwapEngine
from csa_automaton import CSALegalContract
from envs.reward_calculators import BaseRewardCalculator, JointOptimizationReward

class OTCHedgingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 engine: SyntheticSwapEngine, 
                 automaton_config: dict, 
                 reward_calculator: BaseRewardCalculator = None, 
                 num_paths: int = 100, 
                 num_days: int = 30, 
                 initial_cash: float = 1000000.0,
                 tx_cost_coeff: float = 0.0001):
        super(OTCHedgingEnv, self).__init__()
        
        self.engine = engine
        self.automaton_config = automaton_config
        self.reward_calculator = reward_calculator or JointOptimizationReward()
        self.num_paths = num_paths
        self.num_days = num_days
        self.initial_cash = initial_cash
        self.tx_cost_coeff = tx_cost_coeff

        # Cache for pre-simulated trajectories
        self.trajectories = None

        # Space Definitions
        # Action space: Discrete(6)
        # Action mapping:
        # 0: Hedge 0%, Ignore Margin
        # 1: Hedge 0%, Post Margin
        # 2: Hedge 50%, Ignore Margin
        # 3: Hedge 50%, Post Margin
        # 4: Hedge 100%, Ignore Margin
        # 5: Hedge 100%, Post Margin
        self.action_space = spaces.Discrete(6)

        # Observation space: Concatenated vector
        # [Current MtM, Simulated Short Rate, Cash Balance, Margin Required] + [One-Hot Encoded FSM State]
        self.state_to_idx = {
            "normal": 0,
            "margin_call_issued": 1,
            "grace_period": 2,
            "default": 3
        }
        
        # 4 continuous features + 4 one-hot features
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize Automaton FSM
        self.automaton = CSALegalContract(self.automaton_config)

    def _get_obs(self) -> np.ndarray:
        day_idx = min(self.current_day, self.num_days - 1)
        V_curr = self.trajectories["mtm_profiles"][self.path_idx, day_idx]
        r_curr = self.trajectories["short_rates"][self.path_idx, day_idx]
        margin_required = max(V_curr, 0.0)

        # One-hot encoded automaton state
        one_hot = np.zeros(4, dtype=np.float32)
        state_name = self.automaton.state
        one_hot[self.state_to_idx[state_name]] = 1.0

        obs = np.array([V_curr, r_curr, self.cash_balance, margin_required], dtype=np.float32)
        return np.concatenate([obs, one_hot])

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array of shape (6,) indicating which actions are valid.
        If in grace_period, ignoring margin call (even actions) is invalid/blocked.
        """
        mask = np.ones(6, dtype=bool)
        if self.automaton.state == "grace_period":
            # Actions 0, 2, 4 correspond to Ignore Margin
            mask[0] = False
            mask[2] = False
            mask[4] = False
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate batch of paths if not already generated
        if self.trajectories is None:
            self.trajectories = self.engine.generate_trajectory_batch(
                num_paths=self.num_paths,
                num_days=self.num_days
            )

        # Select a path randomly using Gymnasium's generator
        self.path_idx = self.np_random.integers(0, self.num_paths)
        self.current_day = 0

        # Reset state parameters
        self.cash_balance = self.initial_cash
        self.margin_posted = 0.0
        self.h_curr = 0.0
        self.h_prev = 0.0

        # Reset automaton to initial state
        self.automaton = CSALegalContract(self.automaton_config)

        # Running trackers for callbacks
        self.min_cash = self.cash_balance
        self.rule_violations = 0
        self.margin_calls = 0
        self.unhedged_risk_sum = 0.0
        self.state_counts = {k: 0 for k in self.state_to_idx.keys()}
        self.state_counts[self.automaton.state] += 1

        # Initial check for margin breach on day 0
        V_0 = self.trajectories["mtm_profiles"][self.path_idx, 0]
        margin_required = max(V_0, 0.0)
        if margin_required > self.margin_posted and self.automaton.state == "normal":
            if self.automaton.safe_trigger("observe_breach"):
                self.margin_calls += 1

        obs = self._get_obs()
        info = {
            "path_idx": self.path_idx,
            "automaton_state": self.automaton.state
        }
        return obs, info

    def step(self, action: int):
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Track rule violations (if the agent attempts an action that is currently masked)
        mask = self.action_masks()
        if not mask[action]:
            self.rule_violations += 1

        # Decode action
        # Hedging ratio choice
        if action in (0, 1):
            h_new = 0.0
        elif action in (2, 3):
            h_new = 0.5
        else:
            h_new = 1.0

        # Margin decision
        post_margin = (action % 2 == 1)

        # Current market parameters before step transitions
        V_prev = self.trajectories["mtm_profiles"][self.path_idx, self.current_day]
        margin_required_prev = max(V_prev, 0.0)

        # Apply action to FSM and update Cash Balance
        self.h_prev = self.h_curr
        self.h_curr = h_new

        if post_margin:
            # Transition FSM back to normal if a call was active
            if self.automaton.state in ("margin_call_issued", "grace_period"):
                self.automaton.safe_trigger("receive_full_margin")
            
            # Post full variation margin
            margin_payment = margin_required_prev - self.margin_posted
            self.cash_balance -= margin_payment
            self.margin_posted = margin_required_prev
        else:
            # FSM transition for ignore margin
            if self.automaton.state == "margin_call_issued":
                self.automaton.safe_trigger("elapse_one_day")
            elif self.automaton.state == "grace_period":
                self.automaton.safe_trigger("miss_deadline")

        # Roll valuation clock forward by 1 day
        self.current_day += 1

        # Check if episode is completed
        terminated = (self.automaton.state == "default")
        truncated = (self.current_day >= self.num_days)

        # Read new market parameters
        if not (terminated or truncated):
            V_curr = self.trajectories["mtm_profiles"][self.path_idx, self.current_day]
            delta_V = V_curr - V_prev
            
            # Update cash balance with hedging PnL and transaction cost
            tx_cost = self.tx_cost_coeff * abs(self.h_curr - self.h_prev) * self.engine.notional
            delta_pnl = (1.0 - self.h_prev) * delta_V
            self.cash_balance += delta_pnl - tx_cost

            # Check for new breaches on the advanced day
            margin_required_curr = max(V_curr, 0.0)
            if margin_required_curr > self.margin_posted and self.automaton.state == "normal":
                if self.automaton.safe_trigger("observe_breach"):
                    self.margin_calls += 1
        else:
            # For final step, delta_V is 0
            delta_V = 0.0

        # Update trackers
        self.min_cash = min(self.min_cash, self.cash_balance)
        self.unhedged_risk_sum += (1.0 - self.h_curr) * abs(delta_V)
        self.state_counts[self.automaton.state] += 1

        # Calculate reward
        V_curr = self.trajectories["mtm_profiles"][self.path_idx, min(self.current_day, self.num_days - 1)]
        margin_required = max(V_curr, 0.0)
        
        is_default = (self.automaton.state == "default")
        reward = self.reward_calculator.calculate_reward(
            delta_V=delta_V,
            h_prev=self.h_prev,
            h_curr=self.h_curr,
            notional=self.engine.notional,
            is_default=is_default,
            margin_required=margin_required,
            margin_posted=self.margin_posted
        )

        obs = self._get_obs()
        info = {
            "path_idx": self.path_idx,
            "automaton_state": self.automaton.state,
            "cash_balance": self.cash_balance,
            "margin_posted": self.margin_posted,
            "margin_required": margin_required,
            "hedging_ratio": self.h_curr
        }

        # If the episode ends, construct and insert summary metrics for callback
        if terminated or truncated:
            total_steps = self.current_day
            mean_unhedged_risk = self.unhedged_risk_sum / total_steps if total_steps > 0 else 0.0
            
            # Normalize occupancy counts to fractions
            total_occupancies = sum(self.state_counts.values())
            state_occupancy = {}
            for state_name, count in self.state_counts.items():
                state_occupancy[state_name] = count / total_occupancies if total_occupancies > 0 else 0.0
                
            info["episode_metrics"] = {
                "peak_liquidity": max(0.0, -self.min_cash),
                "rule_violations": self.rule_violations,
                "mean_unhedged_risk": mean_unhedged_risk,
                "state_occupancy": state_occupancy,
                "margin_calls": self.margin_calls,
                "defaults": 1 if self.automaton.state == "default" else 0
            }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Day: {self.current_day}, State: {self.automaton.state}, Cash: {self.cash_balance:.2f}")
