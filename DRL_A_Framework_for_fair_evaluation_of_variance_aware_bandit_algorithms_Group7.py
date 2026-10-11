"""
Group 7 - DRL Assignment Problem I
====================================
Paper : A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms
Author: Elise Wolf (2025)  |  arXiv:2510.27001
Repo  : https://github.com/2025ag05770/DRL-Asignment-group-07

Replication protocol — matches Wolf (2025), Sections 3.2 / 3.3 / 3.4 / 3.5:
  - K = 2 Bernoulli arms
  - Horizon T = 1,000,000 steps
  - 100 independent trials
  - Identical initialisation (each arm pulled once before policy begins)
  - Deterministic seeds (seed = trial index)
  - One arm pull per time step (UCB-Improved adapted accordingly)
  - Arm-permutation averaging: results averaged over both orderings [p1,p2]
    and [p2,p1] to remove argmax tie-breaking bias (Wolf 2025, Section 3.4)
  - Metrics: average regret, reward variance, suboptimal pull ratio, p-value
    from chi-squared test vs optimal-arm baseline variance (Wolf 2025, Section 3.5)

Run (quick demo):   python Group_7_DRL_ASSIGNMENT.py --T 100000 --trials 10
Run (paper exact):  python Group_7_DRL_ASSIGNMENT.py --T 1000000 --trials 100  
Outputs: results_table.csv, regret_curves.png
"""

import argparse
import zlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

# ──────────────────────────────────────────────────────────────
#  PAPER PARAMETERS  (Wolf 2025, Section 3.2 / Section 3.3)
# ──────────────────────────────────────────────────────────────
HORIZON     = 1_000_000   # T
NUM_TRIALS  = 100          # independent repetitions
SEED_BASE   = 0            # deterministic; trial i uses seed SEED_BASE+i

SCENARIOS = {
    "Scenario A – Baseline        (p1=0.80, p2=0.90)": [0.80, 0.90],
    "Scenario B – Low-Var Micro   (p1=0.895, p2=0.90)": [0.895, 0.90],
    "Scenario C – High-Var Micro  (p1=0.89, p2=0.895)": [0.89, 0.895],
}

# ──────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ──────────────────────────────────────────────────────────────
class BernoulliBandit:
    """K-armed Bernoulli bandit (Wolf 2025, Section 3.2)."""
    def __init__(self, probs, seed):
        self.probs       = np.array(probs, dtype=float)
        self.K           = len(probs)
        self.optimal_arm = int(np.argmax(self.probs))
        self.Q_star      = float(np.max(self.probs))
        self.rng         = np.random.RandomState(seed)

    def pull(self, arm):
        return 1 if self.rng.rand() < self.probs[arm] else 0


# ──────────────────────────────────────────────────────────────
#  BASE CLASS  – identical initialisation (Wolf 2025, Section 3.4)
# ──────────────────────────────────────────────────────────────
class BanditAlgorithm:
    """
    Shared initialisation: pull every arm once before the policy runs.
    This pre-warms counts and empirical means identically for all algorithms,
    eliminating initialisation-related bias (Wolf 2025, Section 3.4).
    """
    def __init__(self, K):
        self.K      = K
        self.counts = np.zeros(K, dtype=int)
        self.values = np.zeros(K, dtype=float)
        self.t      = 0
        self._init_phase = list(range(K))   # pull each arm once first

    def _get_init_arm(self):
        """Return next arm for forced init phase, or None if done."""
        if self._init_phase:
            return self._init_phase.pop(0)
        return None

    def select_arm(self):
        raise NotImplementedError

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.t += 1


# ──────────────────────────────────────────────────────────────
#  1. CLASSICAL / STANDARD ALGORITHMS  (Wolf 2025, Section 2.2)
# ──────────────────────────────────────────────────────────────
class ExploreThenCommit(BanditAlgorithm):
    """
    ETC: Explore each arm m times, then commit to empirical best.
    Source: Wolf (2025), Section 2.2; Lattimore & Szepesvári [LS20].
    """
    def __init__(self, K, m):
        super().__init__(K)
        self.m        = m
        self.best_arm = None

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        if self.t < self.K * self.m:
            return self.t % self.K
        if self.best_arm is None:
            self.best_arm = int(np.argmax(self.values))
        return self.best_arm


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy: Explore uniformly with probability epsilon.
    Source: Wolf (2025), Section 2.2; Sutton & Barto [SB18].
    """
    def __init__(self, K, epsilon, rng_seed=None):
        super().__init__(K)
        self.epsilon = epsilon
        self.rng     = np.random.RandomState(rng_seed)

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.K))
        return int(np.argmax(self.values))


class UCB(BanditAlgorithm):
    """
    UCB1: UCB score = mean + sqrt(2 ln t / T_k).  No variance term.
    Source: Wolf (2025), Section 2.2; Auer et al. [ACF02].
    """
    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        if self.t == 0:
            return 0
        ucb = self.values + np.sqrt(2.0 * np.log(self.t) / self.counts)
        return int(np.argmax(ucb))


# ──────────────────────────────────────────────────────────────
#  2. VARIANCE-AWARE ALGORITHMS  (Wolf 2025, Section 2.3)
# ──────────────────────────────────────────────────────────────
class UCBTuned(BanditAlgorithm):
    """
    UCB-Tuned: Caps exploration bonus using empirical variance V_k.
    V_k(t) = sigma_hat^2_k + sqrt(2 ln t / T_k),
    bonus    = sqrt((ln t / T_k) * min(1/4, V_k)).
    Source: Wolf (2025), Section 2.3; Auer et al. [ACF02].
    """
    def __init__(self, K):
        super().__init__(K)
        self.sum_sq = np.zeros(K, dtype=float)

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        if self.t == 0:
            return 0
        var_hat = np.maximum(0.0, (self.sum_sq / self.counts) - self.values**2)
        V       = var_hat + np.sqrt(2.0 * np.log(self.t) / self.counts)
        bonus   = np.sqrt((np.log(self.t) / self.counts) * np.minimum(0.25, V))
        return int(np.argmax(self.values + bonus))

    def update(self, arm, reward):
        super().update(arm, reward)
        self.sum_sq[arm] += reward ** 2


class UCBV(BanditAlgorithm):
    """
    UCB-V: B_k = Q_hat + sqrt(2*sigma^2*eps/T_k) + c*(3b*eps/T_k),
    where eps = theta * log(t).
    Parameters: theta=1, c=1, b=1  (Wolf 2025, Table 1; [AMS09]).
    Source: Wolf (2025), Section 2.3; Audibert et al. [AMS09].
    """
    def __init__(self, K, theta=1.0, c=1.0, b=1.0):
        super().__init__(K)
        self.theta  = theta
        self.c      = c
        self.b      = b
        self.sum_sq = np.zeros(K, dtype=float)

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        if self.t == 0:
            return 0
        var_hat = np.maximum(0.0, (self.sum_sq / self.counts) - self.values**2)
        eps     = self.theta * np.log(self.t)
        bound   = (self.values
                   + np.sqrt(2.0 * var_hat * eps / self.counts)
                   + self.c * (3.0 * self.b * eps / self.counts))
        return int(np.argmax(bound))

    def update(self, arm, reward):
        super().update(arm, reward)
        self.sum_sq[arm] += reward ** 2


class EUCBV(BanditAlgorithm):
    """
    EUCBV: Variance-adaptive intervals + phased arm elimination.
    psi = horizon / K^2;  rho = 0.5  (Wolf 2025, Table 1; [Mu17]).
    Source: Wolf (2025), Section 2.3; Mukherjee et al. [Mu17].
    """
    def __init__(self, K, horizon, rho=0.5):
        super().__init__(K)
        self.horizon     = horizon
        self.rho         = rho
        self.psi         = horizon / (K ** 2)
        self.sum_sq      = np.zeros(K, dtype=float)
        self.active_arms = list(range(K))

    def _var(self, a):
        return max(0.0, float((self.sum_sq[a] / self.counts[a]) - self.values[a]**2))

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        eps    = np.log(max(self.psi, 2.0))
        bounds = {}
        lowers = {}
        for a in self.active_arms:
            v          = self._var(a)
            margin     = np.sqrt(2.0 * v * eps / self.counts[a]) + 3.0 * eps / self.counts[a]
            bounds[a]  = self.values[a] + margin
            lowers[a]  = self.values[a] - margin
        best_ub = max(bounds.values())
        # Eliminate arms whose upper bound is clearly below the best lower bound
        self.active_arms = [a for a in self.active_arms
                            if lowers[a] + self.rho >= best_ub or len(self.active_arms) == 1]
        if not self.active_arms:
            self.active_arms = list(range(self.K))
        return max(self.active_arms, key=lambda a: bounds.get(a, self.values[a]))

    def update(self, arm, reward):
        super().update(arm, reward)
        self.sum_sq[arm] += reward ** 2


# ──────────────────────────────────────────────────────────────
#  3. NON-VARIANCE-AWARE CONTROLS  (Wolf 2025, Section 2.4)
# ──────────────────────────────────────────────────────────────
class PACUCB(BanditAlgorithm):
    """
    PAC-UCB: Variance-independent exploration.
    eps = log{K * T_k^q * beta^-1} v 2.
    Parameters: c=1, b=1, q=1.3, beta=0.05  (Wolf 2025, Table 1; [AMS09]).
    Source: Wolf (2025), Section 2.4; Audibert et al. [AMS09].
    """
    def __init__(self, K, c=1.0, b=1.0, q=1.3, beta=0.05):
        super().__init__(K)
        self.c    = c
        self.b    = b
        self.q    = q
        self.beta = beta

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        safe  = np.maximum(self.counts, 1)          # guard divide-by-zero
        eps   = np.maximum(2.0, np.log(self.K * (safe ** self.q) / self.beta))
        bound = (self.values
                 + np.sqrt(2.0 * eps / safe)
                 + self.c * (3.0 * self.b * eps / safe))
        return int(np.argmax(bound))


class UCBImproved(BanditAlgorithm):
    """
    UCB-Improved: Phased elimination; ONE pull per step (Wolf 2025, Section 3.4).
    delta halved each phase.
    Source: Wolf (2025), Section 2.4; Auer & Ortner [AO10].
    """
    def __init__(self, K, horizon, delta=1.0):
        super().__init__(K)
        self.horizon     = horizon
        self.delta       = delta
        self.active_arms = list(range(K))
        self.phase       = 0

    def select_arm(self):
        arm = self._get_init_arm()
        if arm is not None:
            return arm
        if not self.active_arms:
            self.active_arms = list(range(self.K))
        delta_m = self.delta / (2 ** self.phase)
        delta_m = max(delta_m, 1e-10)
        bound   = np.sqrt(np.log(self.horizon * delta_m) / (2.0 * np.maximum(self.counts, 1)))
        chosen  = max(self.active_arms, key=lambda a: self.values[a] + bound[a])
        # Phase-end elimination: remove arms below best - 2*bound
        best_val = max(self.values[a] for a in self.active_arms)
        new_active = [a for a in self.active_arms
                      if self.values[a] + bound[a] >= best_val - bound[chosen]]
        if len(new_active) < len(self.active_arms) and len(new_active) >= 1:
            self.active_arms = new_active
            self.phase += 1
        return chosen


# ──────────────────────────────────────────────────────────────
#  FAST VECTORISED RUNNER  (Wolf 2025, Section 3.2 / Section 3.4 / Section 3.5)
#
#  Pre-generate the full T×K reward table once per trial using
#  numpy, then step through with each policy.  Avoids the overhead
#  of T individual BernoulliBandit.pull() Python calls, making
#  T=500,000 run in ~25 seconds instead of many minutes.
# ──────────────────────────────────────────────────────────────
import sys as _sys
import math as _math

# Ordered list so output rows always appear in the same order
_ALG_NAMES = [
    "ETC (m=100)", "ETC (m=10k)", "Greedy (e=0.05)", "Greedy (e=0.5)",
    "UCB", "UCB-Tuned", "UCB-V", "EUCBV", "PAC-UCB", "UCB-Improved",
]


def _run_alg_on_table(name, reward_table, probs, T, g_seed=0, checkpoints=()):
    """
    Run one algorithm on a pre-generated T×K reward table.
    Returns (suboptimal_count, regret, total_reward) where:
      regret       = Σ(Q*−Q_{A_t}) — deterministic gap sum (Wolf Section 3.5, always ≥ 0)
      total_reward = Σ X_t          — used for reward-variance chi-squared test (Wolf Section 3.5)
    """
    K    = len(probs)
    opt  = int(np.argmax(probs))
    gaps = np.array([float(probs[opt]) - float(p) for p in probs])  # Δ_k = Q*−Q_k

    counts = np.zeros(K, dtype=np.int64)
    values = np.zeros(K, dtype=np.float64)
    sq     = np.zeros(K, dtype=np.float64)   # sum of squared rewards for variance

    regret  = 0.0         # Σ Δ_{A_t}  — always non-negative
    total_r = np.int64(0) # Σ X_t      — for reward-variance chi-squared test
    subopt  = 0           # count of suboptimal pulls

    # ── identical initialisation: pull each arm once (Wolf Section 3.4) ──
    for arm in range(K):
        r = int(reward_table[arm, arm])
        counts[arm] += 1
        values[arm] += (r - values[arm])
        sq[arm]     += r * r
        regret      += gaps[arm]
        total_r     += r
        if arm != opt:
            subopt += 1

    # ── per-algorithm state ──
    best_arm: list[int | None] = [None]      # ETC commit for ETC algorithms
    active     = list(range(K))
    phase      = [0]         # UCB-Improved
    psi        = T / (K ** 2)
    eps_eucbv  = np.log(max(psi, 2.0))

    # greedy RNG — seeded per trial so each trial is independent
    g_rng = np.random.RandomState(g_seed)

    cp_set = set(checkpoints); curve = []
    for t in range(K, T):
        if t in cp_set: curve.append(regret)
        c = np.maximum(counts, 1)          # K=2
        if name == "ETC (m=100)":
            if t < K * 100:
                arm = t % K
            else:
                if best_arm[0] is None:
                    best_arm[0] = int(np.argmax(values))
                arm = best_arm[0]
        elif name == "ETC (m=10k)":
            if t < K * 10_000:
                arm = t % K
            else:
                if best_arm[0] is None:
                    best_arm[0] = int(np.argmax(values))
                arm = best_arm[0]
        elif name == "Greedy (e=0.05)":
            arm = int(g_rng.randint(K)) if g_rng.rand() < 0.05 else int(np.argmax(values))
        elif name == "Greedy (e=0.5)":
            arm = int(g_rng.randint(K)) if g_rng.rand() < 0.50 else int(np.argmax(values))
        elif name == "UCB":
            lt = _math.log(t)
            arm = int(np.argmax(values + np.sqrt(2.0 * lt / c)))
        elif name == "UCB-Tuned":
            lt  = _math.log(t)
            v   = np.maximum(sq / c - values ** 2, 0.0)
            V   = v + np.sqrt(2.0 * lt / c)
            arm = int(np.argmax(values + np.sqrt((lt / c) * np.minimum(0.25, V))))
        elif name == "UCB-V":
            eps = _math.log(t)
            v   = np.maximum(sq / c - values ** 2, 0.0)
            arm = int(np.argmax(values + np.sqrt(2.0 * v * eps / c) + 3.0 * eps / c))
        elif name == "EUCBV":
            v      = np.maximum(sq / c - values ** 2, 0.0)
            margin = np.sqrt(2.0 * v * eps_eucbv / c) + 3.0 * eps_eucbv / c
            act    = np.array(active)
            ubs_a  = values[act] + margin[act]
            lbs_a  = values[act] - margin[act]
            best_lb = float(lbs_a.max())   # keep arm if its UB ≥ best LB (standard EUCBV)
            keep    = (ubs_a >= best_lb) | (len(act) == 1)
            if keep.any() and int(keep.sum()) < len(act):
                active[:] = act[keep].tolist()
            arm = int(act[np.argmax(ubs_a)])
        elif name == "PAC-UCB":
            eps = np.maximum(2.0, np.log(K * (c ** 1.3) / 0.05))
            arm = int(np.argmax(values + np.sqrt(2.0 * eps / c) + 3.0 * eps / c))
        else:  # UCB-Improved
            dm     = max(1.0 / (2 ** phase[0]), 1e-10)
            lt     = _math.log(T * dm)
            bound  = np.sqrt(lt / (2.0 * c))
            act    = np.array(active)
            scores = values[act] + bound[act]
            chosen = int(act[np.argmax(scores)])
            bv     = float(values[act].max())
            keep   = scores >= bv - float(bound[chosen])
            if 0 < int(keep.sum()) < len(active):
                active[:] = act[keep].tolist()
                phase[0] += 1
            if not active:
                active[:] = list(range(K))
            arm = chosen

        r = reward_table[t, arm]           # np.int32
        counts[arm] += 1
        n             = counts[arm]
        values[arm]  += (r - values[arm]) / n
        sq[arm]      += r * r
        regret       += gaps[arm]          # deterministic gap — never negative
        total_r      += r
        subopt       += (arm != opt)

    curve.append(regret)
    return subopt, regret, int(total_r), curve


def _checkpoints(T, n=30):
    return sorted(set(int(x) for x in np.logspace(1, np.log10(T - 1), n)))


def run_one_trial(args):
    """One (trial, arm-ordering) job: all algorithms on the same pre-generated reward table."""
    probs, T, seed = args
    probs_arr    = np.array(probs, dtype=float)
    rng          = np.random.RandomState(seed)
    reward_table = (rng.rand(T, len(probs_arr)) < probs_arr).astype(np.int32)
    cps = _checkpoints(T)
    # crc32 instead of hash(): Python's hash() of a str changes every run, which broke reproducibility
    return {name: _run_alg_on_table(name, reward_table, probs_arr, T,
                                    g_seed=(seed * 31 + zlib.crc32(name.encode()) % 997) % (2**32 - 1),
                                    checkpoints=cps)
            for name in _ALG_NAMES}


PAPER = {  # Wolf (2025) Tables 3/4/5, T = 1,000,000, 100 runs (average regret)
    "A": {"ETC (m=100)": 10.10, "ETC (m=10k)": 1000.00, "Greedy (e=0.5)": 25002.20, "UCB": 238.17, "UCB-Tuned": 30.67,
          "EUCBV": 47.73, "UCB-V": 109.93, "PAC-UCB": 99.50, "UCB-Improved": 32500.11},
    "B": {"ETC (m=10k)": 150.66, "Greedy (e=0.05)": 361.97, "Greedy (e=0.5)": 1264.52, "UCB": 1127.17, "UCB-Tuned": 212.21,
          "UCB-V": 346.58, "EUCBV": 461.36, "PAC-UCB": 395.89, "UCB-Improved": 2525.00},
    "C": {"ETC (m=10k)": 170.53, "Greedy (e=0.05)": 301.55, "Greedy (e=0.5)": 1269.05, "UCB": 1172.57, "UCB-Tuned": 226.09,
          "UCB-V": 367.65, "EUCBV": 476.31, "PAC-UCB": 417.34, "UCB-Improved": 2450.00},
}


def run_scenario(probs, T, num_trials, label, pool):
    """Wolf (2025) Sec 3.2-3.5: average over trials and over both arm orderings (Sec 3.4)."""
    key = label.split()[1]
    print(f"\n{'='*72}\n  {label}\n  Arms: {list(probs)} | T = {T:,} | Trials = {num_trials} x 2 orderings\n{'='*72}")
    jobs = []
    for trial in range(num_trials):
        seed = SEED_BASE + trial
        jobs.append((list(probs), T, seed))
        jobs.append((list(reversed(probs)), T, seed + num_trials))
    res = pool.map(run_one_trial, jobs)
    baseline_var = max(probs) * (1.0 - max(probs)) * T
    summary, curves = {}, {}
    print(f"  {'Algorithm':<18}{'Avg Regret':>12}{'Subopt Ratio':>14}{'p-value':>9}{'Paper (T=1e6)':>15}")
    print("  " + "-" * 68)
    for name in _ALG_NAMES:
        reg = np.array([(res[2*i][name][1] + res[2*i+1][name][1]) / 2 for i in range(num_trials)])
        sub = np.array([((res[2*i][name][0] + res[2*i+1][name][0]) / 2) / T for i in range(num_trials)])
        rw  = np.array([(res[2*i][name][2] + res[2*i+1][name][2]) / 2 for i in range(num_trials)])
        chi2 = (num_trials - 1) * np.var(rw) / baseline_var
        pval = float(1.0 - stats.chi2.cdf(chi2, df=num_trials - 1))
        curves[name] = np.mean([[(a + b) / 2 for a, b in zip(res[2*i][name][3], res[2*i+1][name][3])]
                                for i in range(num_trials)], axis=0)
        ref = PAPER[key].get(name)
        summary[name] = dict(avg_regret=float(reg.mean()), std_regret=float(reg.std()), subopt_ratio=float(sub.mean()),
                             p_value=pval, paper_regret_T1e6=ref)
        print(f"  {name:<18}{reg.mean():>12.2f}{sub.mean():>14.5f}{pval:>9.2f}{(f'{ref:,.2f}' if ref else '-'):>15}")
    return summary, curves


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    print("=" * 72 + "\n  Group 7 | DRL Assignment Problem I | Wolf (2025) arXiv:2510.27001\n" +
          "  Repo : https://github.com/2025ag05770/DRL-Asignment-group-07\n" + "=" * 72)
    rows, fig_data = [], {}
    with Pool(a.workers) as pool:
        for label, probs in SCENARIOS.items():
            s, c = run_scenario(probs, a.T, a.trials, label, pool)
            fig_data[label] = c
            rows += [dict(scenario=label, algorithm=n, **v, T=a.T, trials=a.trials) for n, v in s.items()]
    pd.DataFrame(rows).to_csv("results_table.csv", index=False)
    cps = _checkpoints(a.T)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (label, c) in zip(axs, fig_data.items()):
        for n, y in c.items():
            ax.plot(cps + [a.T] if len(y) == len(cps) + 1 else cps, y, label=n)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(label.split("(")[0].strip())
        ax.set_xlabel("t"); ax.set_ylabel("average cumulative regret")
    axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("regret_curves.png", dpi=150)
    print("\nSaved results_table.csv and regret_curves.png")
    print("Note: absolute regret scales with T. Paper tables use T=1e6, 100 trials; compare ranking, not raw values, at smaller T.")



# $ python Group_7_DRL_ASSIGNMENT.py
# ========================================================================
#   Group 7 | DRL Assignment Problem I | Wolf (2025) arXiv:2510.27001
#   Repo : https://github.com/2025ag05770/DRL-Asignment-group-07
# ========================================================================

# ========================================================================
#   Scenario A – Baseline        (p1=0.80, p2=0.90)
#   Arms: [0.8, 0.9] | T = 100,000 | Trials = 10 x 2 orderings
# ========================================================================
#   Algorithm           Avg Regret  Subopt Ratio  p-value  Paper (T=1e6)
#   --------------------------------------------------------------------
#   ETC (m=100)              10.00       0.00100     0.36          10.10
#   ETC (m=10k)            1000.00       0.10000     0.37       1,000.00
#   Greedy (e=0.05)         265.96       0.02660     0.22              -
#   Greedy (e=0.5)         2497.46       0.24975     0.27      25,002.20
#   UCB                     164.26       0.01643     0.24         238.17
#   UCB-Tuned                28.51       0.00285     0.36          30.67
#   UCB-V                    89.13       0.00891     0.37         109.93
#   EUCBV                    74.81       0.00748     0.35          47.73
#   PAC-UCB                 263.85       0.02639     0.33          99.50
#   UCB-Improved             43.39       0.00434     0.32      32,500.11

# ========================================================================
#   Scenario B – Low-Var Micro   (p1=0.895, p2=0.90)
#   Arms: [0.895, 0.9] | T = 100,000 | Trials = 10 x 2 orderings
# ========================================================================
#   Algorithm           Avg Regret  Subopt Ratio  p-value  Paper (T=1e6)
#   --------------------------------------------------------------------
#   ETC (m=100)             150.20       0.30040     0.00              -
#   ETC (m=10k)             130.00       0.26000     0.02         150.66
#   Greedy (e=0.05)          90.40       0.18080     0.03         361.97
#   Greedy (e=0.5)          142.57       0.28514     0.49       1,264.52
#   UCB                     200.68       0.40136     0.59       1,127.17
#   UCB-Tuned                94.93       0.18985     0.43         212.21
#   UCB-V                   124.63       0.24927     0.56         346.58
#   EUCBV                   121.74       0.24348     0.52         461.36
#   PAC-UCB                 208.96       0.41792     0.59         395.89
#   UCB-Improved            145.81       0.29163     0.67       2,525.00

# ========================================================================
#   Scenario C – High-Var Micro  (p1=0.89, p2=0.895)
#   Arms: [0.89, 0.895] | T = 100,000 | Trials = 10 x 2 orderings
# ========================================================================
#   Algorithm           Avg Regret  Subopt Ratio  p-value  Paper (T=1e6)
#   --------------------------------------------------------------------
#   ETC (m=100)             175.15       0.35030     0.00              -
#   ETC (m=10k)             130.00       0.26000     0.04         170.53
#   Greedy (e=0.05)         135.44       0.27088     0.00         301.55
#   Greedy (e=0.5)          141.47       0.28293     0.71       1,269.05
#   UCB                     202.75       0.40549     0.77       1,172.57
#   UCB-Tuned                98.29       0.19659     0.38         226.09
#   UCB-V                   133.62       0.26724     0.75         367.65
#   EUCBV                   126.61       0.25322     0.69         476.31
#   PAC-UCB                 210.95       0.42190     0.74         417.34
#   UCB-Improved            147.80       0.29560     0.80       2,450.00

# Saved results_table.csv and regret_curves.png
# Note: absolute regret scales with T. Paper tables use T=1e6, 100 trials; compare ranking, not raw values, at smaller T.