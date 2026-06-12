from stable_baselines3.common.callbacks import BaseCallback

class FinancialMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(FinancialMetricsCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # check if environment info contains our custom episode metrics
        # (which are populated by OTCHedgingEnv when an episode terminates/truncates)
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode_metrics" in info:
                    self.episode_count += 1
                    metrics = info["episode_metrics"]
                    
                    # Record financial metrics to the Stable-Baselines3 Logger
                    # These will be written to TensorBoard automatically at rollout end
                    self.logger.record("financial/mean_peak_liquidity", metrics["peak_liquidity"])
                    self.logger.record("financial/rule_violations", metrics["rule_violations"])
                    self.logger.record("financial/mean_unhedged_risk", metrics["mean_unhedged_risk"])
                    
                    # Record state occupancy metrics
                    state_occupancies = metrics["state_occupancy"]
                    for state_name, fraction in state_occupancies.items():
                        self.logger.record(f"automaton/state_occupancy_{state_name}", fraction)
                        
        return True
