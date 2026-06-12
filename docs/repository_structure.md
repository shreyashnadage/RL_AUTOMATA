# Repository Structure and Architecture

This document provides a detailed breakdown of how the repository is structured, from the fundamental building blocks to the RL environment, agent, reward functions, and the integration of QuantLib and the FSM.

---

## 1. High-Level Directory Structure
The repository is designed using the **Strategy Pattern** to keep pricing, state logic, environment rules, and agents decoupled:

*   **`engine.py`**: The stochastic simulation and valuation engine using QuantLib.
*   **`csa_automaton.py` & `simple_automaton.json`**: The legal Finite State Machine (FSM) modeling the Credit Support Annex (CSA) rules.
*   **`envs/`**:
    *   [otc_env.py](file:///d:/RL_AUTOMATA/envs/otc_env.py): The Gymnasium environment (`OTCHedgingEnv`) tying everything together.
    *   [reward_calculators.py](file:///d:/RL_AUTOMATA/envs/reward_calculators.py): Implementations of different multi-objective reward strategies.
*   **`agents/`**:
    *   [agent_factory.py](file:///d:/RL_AUTOMATA/agents/agent_factory.py): Spawns either `MaskablePPO` (which utilizes action masking) or standard `PPO`.
    *   [callbacks.py](file:///d:/RL_AUTOMATA/agents/callbacks.py): Collects and logs domain-specific metrics (liquidity deficit, unhedged risk, FSM occupancies) to TensorBoard.

---

## 2. Core Building Blocks

### A. Stochastic Simulator & Valuation Engine (`engine.py`)
This component models the market dynamics using the **Hull-White 1-Factor (HW1F)** short-rate model and prices a plain vanilla interest rate swap using **QuantLib**.

1.  **Vectorized Short-Rate Path Simulation**:
    *   In `generate_trajectory_batch()`, daily short-rate ($r_t$) paths are generated using the analytical solution of the HW1F process:
        $$r_t = x_t + \alpha_t$$
        where $x_t$ is a mean-reverting process simulated via a loop over the time grid, and $\alpha_t$ is the analytical calibration function fitting the initial flat yield curve.
2.  **Pathwise Swap Re-pricing**:
    *   Instead of pricing on the fly during training (which is slow), paths are pre-priced in a batch during the environment's `reset()`.
    *   For each day and path, the engine calculates the analytical zero-coupon bond prices from the simulated $r_t$.
    *   It builds a new `ql.DiscountCurve` with these discount factors, links it to `self.pricing_curve_handle`, and calls `self.swap.NPV()` to extract the Mark-to-Market (MtM) value ($V_t$).
    *   The engine outputs the MtM profile ($V_t$) and the exposure profile ($\max(V_t, 0)$).

### B. CSA Legal Automaton (`csa_automaton.py` & `simple_automaton.json`)
The legal agreement is represented as a Finite State Machine using the `transitions` library in Python:

*   **States**: `normal`, `margin_call_issued`, `grace_period`, and `default`.
*   **Transitions**:
    *   `observe_breach` (from `normal` to `margin_call_issued` when margin is owed).
    *   `elapse_one_day` (from `margin_call_issued` to `grace_period`).
    *   `receive_full_margin` (from `margin_call_issued` or `grace_period` back to `normal`).
    *   `miss_deadline` (from `grace_period` to `default`).
*   **Execution**:
    *   The `CSALegalContract` handles transitions dynamically. The `safe_trigger()` method executes state transitions safely, returning `False` if an action is legally invalid in the current state instead of throwing an unhandled exception.

---

## 3. The Gymnasium Environment (`envs/otc_env.py`)

The environment matches the standard Gymnasium API and orchestrates the interaction between the market engine and the CSA contract FSM.

### A. Action Space
The environment has a **`Discrete(6)`** action space. The actions represent the cartesian product of two decisions (Hedging Ratio $\times$ Margin Posting):
*   **Action 0**: Hedge 0%, Ignore Margin Call
*   **Action 1**: Hedge 0%, Post Margin
*   **Action 2**: Hedge 50%, Ignore Margin Call
*   **Action 3**: Hedge 50%, Post Margin
*   **Action 4**: Hedge 100%, Ignore Margin Call
*   **Action 5**: Hedge 100%, Post Margin

### B. Observation Space
The observation space is a `Box` space of 8 continuous/one-hot values, concatenating continuous market metrics with the FSM state:
$$\text{Obs} = [ V_t,\, r_t,\, \text{Cash Balance},\, \text{Margin Required} ] \quad\mathbin{\Vert}\quad [ \text{One-Hot State} ]$$
The FSM state is mapped to a 4-dimensional one-hot vector: `normal` $[1,0,0,0]$, `margin_call_issued` $[0,1,0,0]$, `grace_period` $[0,0,1,0]$, and `default` $[0,0,0,1]$.

### C. Invalid Action Masking
This is the core mechanic preventing the agent from violating covenants:
*   The `action_masks()` method returns a boolean array of shape `(6,)` specifying which actions are legal.
*   If the FSM state is `grace_period`, the agent **must** post margin because failing to do so would trigger `miss_deadline` $\rightarrow$ `default`.
*   Thus, when in `grace_period`, actions that ignore the margin call (`0`, `2`, `4`) are masked out (`False`), forcing the agent to select from the legal margin-paying actions (`1`, `3`, `5`).

### D. State Updates inside `step()`
*   The step function decodes the selected action.
*   **Back-Office (Margin)**: If `post_margin` is chosen, the FSM transitions back to `normal` via `receive_full_margin`, and cash is deducted to cover the margin shortfall. Otherwise, the FSM progresses closer to default (e.g., `elapse_one_day` or `miss_deadline`).
*   **Front-Office (Hedging)**: The environment rolls the day forward. The agent's cash balance is updated with the hedging PnL:
    $$\Delta \text{PnL} = (1 - h_{t-1}) \cdot \Delta V_t - \text{Transaction Costs}$$
*   **Termination**: If the contract hits the `default` state, the episode is `terminated`. If the timeline reaches the final day (`num_days`), the episode is `truncated`.

---

## 4. Reward Functions (`envs/reward_calculators.py`)

The reward function evaluates the performance of the chosen action. Under the default **`JointOptimizationReward`**, the step reward is formulated as:
$$\text{Reward} = \Delta \text{PnL} - \lambda_{\text{risk}} \cdot \text{Unhedged Risk} - \lambda_{\text{liq}} \cdot \text{Margin Shortfall} - \text{Default Penalty}$$

*   **$\Delta$ PnL**: The portfolio cash change (hedged PnL minus transaction costs).
*   **Unhedged Risk**: Penalizes price exposure on the unhedged portion of the swap:
    $$\text{Unhedged Risk} = (1 - h_t) \cdot |\Delta V_t|$$
*   **Margin Shortfall**: Penalizes failing to keep up with margin obligations:
    $$\text{Margin Shortfall} = \max(\text{Margin Required} - \text{Margin Posted}, 0)$$
*   **Default Penalty**: If `is_default` is `True`, a massive negative constant penalty (e.g., $-100,000$) is returned immediately.

---

## 5. The RL Agent (`agents/agent_factory.py`)

The framework supports two training strategies:
1.  **Maskable PPO (`MaskablePPO`)**:
    *   Uses the `ActionMasker` wrapper on the environment. 
    *   During rollouts, the policy network zeroes out the action probabilities of illegal actions before computing the action distribution. This guarantees the agent never performs an action that leads to an unforced default during training.
2.  **Baseline PPO (`PPO`)**:
    *   A standard PPO model without masking. It must learn the legal constraints solely through trial-and-error by experiencing the large negative `default_penalty`.
