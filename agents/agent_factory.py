from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker

def create_agent(algo_name: str, env, tensorboard_log_dir: str):
    """
    Factory function to initialize a Stable-Baselines3 / sb3-contrib agent.
    
    Args:
        algo_name: "MaskablePPO" or "PPO"
        env: The Gymnasium environment instance
        tensorboard_log_dir: Local path to save TensorBoard logs
    """
    if algo_name == "MaskablePPO":
        # Wrap environment to compute and supply the action masks to MaskablePPO
        masked_env = ActionMasker(env, action_mask_fn=lambda e: e.unwrapped.action_masks())
        return MaskablePPO("MlpPolicy", masked_env, tensorboard_log=tensorboard_log_dir)
    elif algo_name == "PPO":
        # Baseline comparison agent (No action masking, relies strictly on rewards/penalties)
        return PPO("MlpPolicy", env, tensorboard_log=tensorboard_log_dir)
    else:
        raise ValueError(f"Unsupported algorithm '{algo_name}'")
