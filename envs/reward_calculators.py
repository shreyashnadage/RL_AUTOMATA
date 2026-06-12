from abc import ABC, abstractmethod

class BaseRewardCalculator(ABC):
    @abstractmethod
    def calculate_reward(self, 
                         delta_V: float, 
                         h_prev: float, 
                         h_curr: float, 
                         notional: float, 
                         is_default: bool, 
                         margin_required: float, 
                         margin_posted: float) -> float:
        """
        Calculates the step reward based on the strategy.
        
        Args:
            delta_V: Change in swap Mark-to-Market value (V_t - V_{t-1}).
            h_prev: Hedging ratio from the previous step.
            h_curr: New hedging ratio chosen in this step.
            notional: Swap notional value.
            is_default: True if the contract state machine has transitioned to default.
            margin_required: Current required variation margin.
            margin_posted: Current posted variation margin.
        """
        pass

class JointOptimizationReward(BaseRewardCalculator):
    def __init__(self, 
                 lambda_risk: float = 0.5, 
                 lambda_liq: float = 0.1, 
                 tx_cost_coeff: float = 0.0001, 
                 default_penalty: float = 100000.0):
        self.lambda_risk = lambda_risk
        self.lambda_liq = lambda_liq
        self.tx_cost_coeff = tx_cost_coeff
        self.default_penalty = default_penalty

    def calculate_reward(self, 
                         delta_V: float, 
                         h_prev: float, 
                         h_curr: float, 
                         notional: float, 
                         is_default: bool, 
                         margin_required: float, 
                         margin_posted: float) -> float:
        if is_default:
            return -self.default_penalty

        # Delta PnL component
        tx_cost = self.tx_cost_coeff * abs(h_curr - h_prev) * notional
        # Swap PnL is delta_V, hedge PnL is -h_prev * delta_V
        delta_pnl = (1.0 - h_prev) * delta_V - tx_cost

        # Unhedged risk component (unhedged portion of current price movement)
        unhedged_risk = (1.0 - h_curr) * abs(delta_V)

        # Margin shortfall component
        margin_shortfall = max(margin_required - margin_posted, 0.0)

        reward = delta_pnl - self.lambda_risk * unhedged_risk - self.lambda_liq * margin_shortfall
        return reward

class PureHedgingReward(BaseRewardCalculator):
    def __init__(self, 
                 lambda_risk: float = 0.5, 
                 tx_cost_coeff: float = 0.0001, 
                 default_penalty: float = 100000.0):
        self.lambda_risk = lambda_risk
        self.tx_cost_coeff = tx_cost_coeff
        self.default_penalty = default_penalty

    def calculate_reward(self, 
                         delta_V: float, 
                         h_prev: float, 
                         h_curr: float, 
                         notional: float, 
                         is_default: bool, 
                         margin_required: float, 
                         margin_posted: float) -> float:
        if is_default:
            return -self.default_penalty

        tx_cost = self.tx_cost_coeff * abs(h_curr - h_prev) * notional
        delta_pnl = (1.0 - h_prev) * delta_V - tx_cost
        unhedged_risk = (1.0 - h_curr) * abs(delta_V)

        reward = delta_pnl - self.lambda_risk * unhedged_risk
        return reward

class PureLiquidityReward(BaseRewardCalculator):
    def __init__(self, 
                 lambda_liq: float = 0.1, 
                 default_penalty: float = 100000.0):
        self.lambda_liq = lambda_liq
        self.default_penalty = default_penalty

    def calculate_reward(self, 
                         delta_V: float, 
                         h_prev: float, 
                         h_curr: float, 
                         notional: float, 
                         is_default: bool, 
                         margin_required: float, 
                         margin_posted: float) -> float:
        if is_default:
            return -self.default_penalty

        margin_shortfall = max(margin_required - margin_posted, 0.0)
        reward = -self.lambda_liq * margin_shortfall
        return reward
