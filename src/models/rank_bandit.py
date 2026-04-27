"""
Multi-Armed Bandit for LoRA Rank Selection in C-CLIP continual learning.

Motivation (from CoDyRA paper):
  - Higher rank  → more plasticity (learns new task well) but more forgetting
  - Lower rank   → more stability (less forgetting) but limits adaptation
  - The sweet-spot rank is TASK-SPECIFIC — ideal for bandit exploration

Why MAB fits here:
  - Each task is an independent context (stateless across tasks)
  - The reward is observed AFTER training (delayed but well-defined)
  - We want to balance exploration (trying ranks we haven't seen much) with
    exploitation (reusing ranks that worked well historically)

Reward signal:
  Primary  : validation accuracy on the current task after training  (plasticity)
  Secondary: zero-shot retention on a held-out ref dataset           (stability)
  Combined : R = alpha * plasticity + (1 - alpha) * stability

  This composite reward operationalises the plasticity-stability tradeoff
  described in both CoDyRA and DoRA papers.

Algorithms implemented:
  - UCB1      : upper-confidence-bound, works well with limited pulls
  - Epsilon-Greedy : simple baseline, annealed epsilon
  - Thompson Sampling (Beta) : Bayesian, best when rewards ∈ [0,1]
"""

import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import os


class RankArm:
    """Tracks statistics for a single rank value (one arm of the bandit)."""

    def __init__(self, rank: int):
        self.rank = rank
        self.n_pulls: int = 0          # times this rank was selected
        self.total_reward: float = 0.0
        self.rewards: List[float] = []
        # For Thompson Sampling (Beta distribution parameters)
        self.alpha: float = 1.0        # successes + 1
        self.beta: float = 1.0         # failures  + 1

    @property
    def mean_reward(self) -> float:
        if self.n_pulls == 0:
            return 0.0
        return self.total_reward / self.n_pulls

    def update(self, reward: float):
        """Update arm statistics with a new observed reward (reward ∈ [0, 1])."""
        self.n_pulls += 1
        self.total_reward += reward
        self.rewards.append(reward)
        # Correct soft Beta update — both parameters always updated.
        # This gives proper Bayesian posterior for continuous rewards in [0,1].
        self.alpha += reward
        self.beta += (1.0 - reward)

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "n_pulls": self.n_pulls,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "alpha": self.alpha,
            "beta": self.beta,
            "rewards": self.rewards,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RankArm":
        arm = cls(d["rank"])
        arm.n_pulls = d["n_pulls"]
        arm.total_reward = d["total_reward"]
        arm.alpha = d["alpha"]
        arm.beta = d["beta"]
        arm.rewards = d.get("rewards", [])
        return arm

    def __repr__(self):
        return (f"RankArm(r={self.rank}, pulls={self.n_pulls}, "
                f"mean={self.mean_reward:.3f})")


class LoRARankBandit:
    """
    Multi-Armed Bandit that selects the LoRA rank for each continual learning task.

    Args:
        rank_choices   : list of candidate ranks, e.g. [4, 8, 16, 32]
        algorithm      : 'ucb1' | 'epsilon_greedy' | 'thompson'
        plasticity_w   : weight for task-accuracy in composite reward (0-1)
        stability_w    : weight for zero-shot retention in composite reward (0-1)
                         plasticity_w + stability_w should equal 1
        epsilon        : initial epsilon for epsilon-greedy
        epsilon_decay  : multiplicative decay per task
        ucb_c          : exploration constant for UCB1
        save_path      : if set, bandit state is persisted here across runs
    """

    def __init__(
        self,
        rank_choices: List[int] = [4, 8, 16, 32],
        algorithm: str = "ucb1",
        plasticity_w: float = 0.6,
        stability_w: float = 0.4,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.85,
        ucb_c: float = 2.0,
        save_path: Optional[str] = None,
    ):
        assert algorithm in ("ucb1", "epsilon_greedy", "thompson"), \
            f"Unknown algorithm: {algorithm}"
        assert abs(plasticity_w + stability_w - 1.0) < 1e-6, \
            "plasticity_w + stability_w must equal 1.0"

        self.rank_choices = sorted(rank_choices)
        self.algorithm = algorithm
        self.plasticity_w = plasticity_w
        self.stability_w = stability_w
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.ucb_c = ucb_c
        self.save_path = save_path

        self.arms: Dict[int, RankArm] = {r: RankArm(r) for r in rank_choices}
        self.task_history: List[Dict] = []   # log of (task, rank_chosen, reward)
        self.total_tasks: int = 0

        # If persisted state exists, load it
        if save_path and os.path.exists(save_path):
            self._load(save_path)
            print(f"[RankBandit] Loaded state from {save_path}")

    # ------------------------------------------------------------------ #
    #  Selection                                                           #
    # ------------------------------------------------------------------ #

    def select_rank(self, task_idx: int, task_name: str = "") -> int:
        """
        Choose a rank for the upcoming task.

        During the first len(rank_choices) tasks we do one forced pull per arm
        (round-robin initialization) to avoid cold-start bias.
        """
        n_unpulled = sum(1 for arm in self.arms.values() if arm.n_pulls == 0)
        if n_unpulled > 0:
            # Force-explore: pick first unpulled arm in rank_choices order.
            # This uses the config-specified ordering (e.g. [16, 8, 32, 4])
            # instead of dict insertion order, giving predictable assignment.
            unpulled = [r for r in self.rank_choices if self.arms[r].n_pulls == 0]
            rank = unpulled[0]
            strategy = f"force_explore"
        elif self.algorithm == "ucb1":
            rank, strategy = self._ucb1_select()
        elif self.algorithm == "epsilon_greedy":
            rank, strategy = self._epsilon_greedy_select()
        else:  # thompson
            rank, strategy = self._thompson_select()

        print(f"\n[RankBandit] Task {task_idx} ({task_name})")
        print(f"  Algorithm   : {self.algorithm}")
        print(f"  Strategy    : {strategy}")
        print(f"  Rank chosen : {rank}")
        self._print_arm_stats()

        return rank

    def _ucb1_select(self) -> Tuple[int, str]:
        """UCB1: rank = argmax[ mean + c * sqrt(ln(N) / n_i) ]"""
        total_pulls = sum(arm.n_pulls for arm in self.arms.values())
        ln_N = math.log(max(total_pulls, 1))

        scores = {}
        for r, arm in self.arms.items():
            exploit = arm.mean_reward
            explore = self.ucb_c * math.sqrt(ln_N / arm.n_pulls)
            scores[r] = exploit + explore

        best = max(scores, key=scores.__getitem__)
        return best, f"ucb1(score={scores[best]:.3f})"

    def _epsilon_greedy_select(self) -> Tuple[int, str]:
        """Epsilon-greedy with annealed epsilon."""
        if random.random() < self.epsilon:
            rank = random.choice(self.rank_choices)
            return rank, f"explore(eps={self.epsilon:.3f})"
        else:
            rank = max(self.arms, key=lambda r: self.arms[r].mean_reward)
            return rank, f"exploit(eps={self.epsilon:.3f})"

    def _thompson_select(self) -> Tuple[int, str]:
        """Thompson Sampling: sample from Beta(alpha, beta) for each arm."""
        samples = {r: np.random.beta(arm.alpha, arm.beta)
                   for r, arm in self.arms.items()}
        best = max(samples, key=samples.__getitem__)
        return best, f"thompson(sample={samples[best]:.3f})"

    # ------------------------------------------------------------------ #
    #  Reward                                                              #
    # ------------------------------------------------------------------ #

    def compute_reward(
        self,
        task_accuracy: float,          # accuracy on current task val set [0,1]
        baseline_accuracy: float,      # pretrained zero-shot on same set [0,1]
        zeroshot_retention: float,     # current zero-shot on ref set     [0,1]
        zeroshot_baseline: float,      # pretrained zero-shot on ref set  [0,1]
        prior_task_accs: Optional[List[float]] = None,   # per-task accs
        prior_task_baselines: Optional[List[float]] = None,  # per-task baselines
    ) -> float:
        """
        Composite reward balancing plasticity and stability.

        Uses WORST-CASE stability instead of average — this ensures that a
        catastrophic drop on even a single prior task (e.g., aircraft 69%→8%)
        is heavily penalised rather than masked by averaging.

        plasticity = how much we GAINED on the new task (relative gain)
        stability  = how well we RETAINED the WORST-performing prior task

        Both are normalised to [0,1] where 1 is perfect.
        """
        # Plasticity: normalised gain over pre-trained baseline
        # If we match baseline → 0.5, if we double → 1.0, if we drop → 0
        if baseline_accuracy > 0:
            gain = (task_accuracy - baseline_accuracy) / baseline_accuracy
            plasticity = max(0.0, min(1.0, 0.5 + 0.5 * gain))
        else:
            plasticity = task_accuracy  # fallback

        # Stability: use WORST-CASE retention across prior tasks
        # This prevents a single catastrophic drop from being masked
        if prior_task_accs and prior_task_baselines:
            # Per-task retention: how much of each prior task's pretrained accuracy we kept
            per_task_retention = []
            below_baseline_penalty = 0.0
            for acc, base in zip(prior_task_accs, prior_task_baselines):
                if base > 0:
                    retention = min(1.0, acc / base)
                else:
                    retention = 1.0
                per_task_retention.append(retention)
                # Extra penalty if a task drops BELOW pretrained baseline
                if acc < base:
                    below_baseline_penalty += (base - acc) / base

            # Worst-case: minimum retention across all prior tasks
            worst_retention = min(per_task_retention)
            avg_retention = sum(per_task_retention) / len(per_task_retention)
            # Blend: 70% worst-case + 30% average (so it's not purely adversarial)
            stability = 0.7 * worst_retention + 0.3 * avg_retention
            # Apply penalty for tasks below baseline (0.1 per task that dropped)
            stability = max(0.0, stability - 0.1 * below_baseline_penalty)
        elif zeroshot_baseline > 0:
            # Fallback to original average-based stability
            stability = min(1.0, zeroshot_retention / zeroshot_baseline)
        else:
            stability = 1.0

        reward = self.plasticity_w * plasticity + self.stability_w * stability

        print(f"[RankBandit] Reward breakdown:")
        print(f"  Task acc     : {task_accuracy:.3f} (baseline {baseline_accuracy:.3f})")
        print(f"  Zero-shot    : {zeroshot_retention:.3f} (baseline {zeroshot_baseline:.3f})")
        if prior_task_accs:
            print(f"  Per-task ret : {[f'{r:.3f}' for r in per_task_retention]}")
            print(f"  Worst-case   : {worst_retention:.3f}")
        print(f"  Plasticity   : {plasticity:.3f}  (w={self.plasticity_w})")
        print(f"  Stability    : {stability:.3f}  (w={self.stability_w})")
        print(f"  Total reward : {reward:.3f}")

        return reward

    def update(
        self,
        rank: int,
        reward: float,
        task_idx: int,
        task_name: str = "",
        extra_info: Optional[Dict] = None,
    ):
        """
        Feed the reward back into the chosen arm and update bandit state.
        Call this AFTER training and evaluation for the task.
        """
        self.arms[rank].update(reward)
        self.total_tasks += 1

        # Decay epsilon for epsilon-greedy
        if self.algorithm == "epsilon_greedy":
            self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

        record = {
            "task_idx": task_idx,
            "task_name": task_name,
            "rank_chosen": rank,
            "reward": reward,
            "extra": extra_info or {},
        }
        self.task_history.append(record)

        print(f"[RankBandit] Updated arm r={rank} -> mean={self.arms[rank].mean_reward:.3f}")

        if self.save_path:
            self._save(self.save_path)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def best_rank(self) -> int:
        """Return the rank with the highest empirical mean reward."""
        return max(self.arms, key=lambda r: self.arms[r].mean_reward)

    def _print_arm_stats(self):
        print("  Arm statistics:")
        for r in self.rank_choices:
            arm = self.arms[r]
            bar = "#" * int(arm.mean_reward * 20)
            print(f"    r={r:2d} | pulls={arm.n_pulls} | "
                  f"mean={arm.mean_reward:.3f} |{bar}|")

    def summary(self) -> Dict:
        return {
            "algorithm": self.algorithm,
            "total_tasks": self.total_tasks,
            "best_rank": self.best_rank(),
            "arms": {r: arm.to_dict() for r, arm in self.arms.items()},
            "task_history": self.task_history,
        }

    def _save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.total_tasks = data.get("total_tasks", 0)
        self.task_history = data.get("task_history", [])
        for r_str, arm_data in data.get("arms", {}).items():
            r = int(r_str)
            if r in self.arms:
                self.arms[r] = RankArm.from_dict(arm_data)

    def __repr__(self):
        return (f"LoRARankBandit(algo={self.algorithm}, "
                f"ranks={self.rank_choices}, tasks_seen={self.total_tasks})")
