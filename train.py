import os
import json
import argparse
from engine import SyntheticSwapEngine
from envs.otc_env import OTCHedgingEnv
from envs.reward_calculators import JointOptimizationReward, PureHedgingReward, PureLiquidityReward
from agents.agent_factory import create_agent
from agents.callbacks import FinancialMetricsCallback

def load_automaton_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="AIRL OTC Derivatives RL Training CLI")
    parser.add_argument("--algo", type=str, default="MaskablePPO", choices=["MaskablePPO", "PPO"],
                        help="RL Algorithm to use (MaskablePPO or PPO)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total number of training timesteps")
    parser.add_argument("--tb_log_dir", type=str, default="./logs/tb/",
                        help="TensorBoard logs directory")
    parser.add_argument("--reward", type=str, default="joint", choices=["joint", "hedging", "liquidity"],
                        help="Reward function strategy")
    parser.add_argument("--paths", type=int, default=100,
                        help="Number of pre-simulated market paths")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days per episode/path")
    
    args = parser.parse_args()

    # 1. Initialize the QuantLib Stochastic Simulation Engine
    print("Initializing Synthetic Swap Pricing Engine (QuantLib)...")
    engine = SyntheticSwapEngine(
        base_rate=0.045,      # 4.5% flat starting rate
        a=0.1,                # Hull-White speed of mean reversion
        sigma=0.01,           # Volatility parameter
        notional=10000000.0,  # $10M notional
        maturity_years=5      # 5-year swap maturity
    )

    # 2. Select Reward Strategy
    if args.reward == "joint":
        reward_calculator = JointOptimizationReward()
    elif args.reward == "hedging":
        reward_calculator = PureHedgingReward()
    else:
        reward_calculator = PureLiquidityReward()

    # 3. Load Legal Automaton config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "simple_automaton.json")
    automaton_config = load_automaton_config(config_path)

    # 4. Create the Gym Environment
    print(f"Creating OTCHedgingEnv with {args.paths} paths and {args.days} days...")
    env = OTCHedgingEnv(
        engine=engine,
        automaton_config=automaton_config,
        reward_calculator=reward_calculator,
        num_paths=args.paths,
        num_days=args.days
    )

    # 5. Create RL Agent via the Agent Factory
    print(f"Creating RL Agent: {args.algo}...")
    model = create_agent(
        algo_name=args.algo,
        env=env,
        tensorboard_log_dir=args.tb_log_dir
    )

    # 6. Initialize Callbacks for Custom Financial & FSM Metrics
    callback = FinancialMetricsCallback()

    # 7. Start Training Loop
    print(f"Starting training for {args.timesteps} timesteps. Logs will be saved to {args.tb_log_dir}...")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    
    # Save the trained policy
    model_path = f"airl_{args.algo.lower()}_model"
    model.save(model_path)
    print(f"Training completed successfully! Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
