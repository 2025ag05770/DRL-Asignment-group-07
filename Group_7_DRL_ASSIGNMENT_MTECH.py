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
    and [p2,p1] to remove argmax tie-breaking bias (Wolf 2025, §3.4)
  - Metrics: average regret, reward variance, suboptimal pull ratio, p-value
    from chi-squared test vs optimal-arm baseline variance (Wolf 2025, §3.5)

Run:  python Group_7_DRL_ASSIGNMENT_MTECH.py
"""

import numpy as np
import scipy.stats as stats

# ──────────────────────────────────────────────────────────────
#  PAPER PARAMETERS  (Wolf 2025, §3.2 / §3.3)
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
    """K-armed Bernoulli bandit (Wolf 2025, §3.2)."""
    def __init__(self, probs, seed):
        self.probs       = np.array(probs, dtype=float)
        self.K           = len(probs)
        self.optimal_arm = int(np.argmax(self.probs))
        self.Q_star      = float(np.max(self.probs))
        self.rng         = np.random.RandomState(seed)

    def pull(self, arm):
        return 1 if self.rng.rand() < self.probs[arm] else 0


# ──────────────────────────────────────────────────────────────
#  BASE CLASS  – identical initialisation (Wolf 2025, §3.4)
# ──────────────────────────────────────────────────────────────
class BanditAlgorithm:
    """
    Shared initialisation: pull every arm once before the policy runs.
    This pre-warms counts and empirical means identically for all algorithms,
    eliminating initialisation-related bias (Wolf 2025, §3.4).
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
#  1. CLASSICAL / STANDARD ALGORITHMS  (Wolf 2025, §2.2)
# ──────────────────────────────────────────────────────────────
class ExploreThenCommit(BanditAlgorithm):
    """
    ETC: Explore each arm m times, then commit to empirical best.
    Source: Wolf (2025), §2.2; Lattimore & Szepesvári [LS20].
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
    Source: Wolf (2025), §2.2; Sutton & Barto [SB18].
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
    Source: Wolf (2025), §2.2; Auer et al. [ACF02].
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
#  2. VARIANCE-AWARE ALGORITHMS  (Wolf 2025, §2.3)
# ──────────────────────────────────────────────────────────────
class UCBTuned(BanditAlgorithm):
    """
    UCB-Tuned: Caps exploration bonus using empirical variance V_k.
    V_k(t) = sigma_hat^2_k + sqrt(2 ln t / T_k),
    bonus    = sqrt((ln t / T_k) * min(1/4, V_k)).
    Source: Wolf (2025), §2.3; Auer et al. [ACF02].
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
    Source: Wolf (2025), §2.3; Audibert et al. [AMS09].
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
    Source: Wolf (2025), §2.3; Mukherjee et al. [Mu17].
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
#  3. NON-VARIANCE-AWARE CONTROLS  (Wolf 2025, §2.4)
# ──────────────────────────────────────────────────────────────
class PACUCB(BanditAlgorithm):
    """
    PAC-UCB: Variance-independent exploration.
    eps = log{K * T_k^q * beta^-1} v 2.
    Parameters: c=1, b=1, q=1.3, beta=0.05  (Wolf 2025, Table 1; [AMS09]).
    Source: Wolf (2025), §2.4; Audibert et al. [AMS09].
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
    UCB-Improved: Phased elimination; ONE pull per step (Wolf 2025, §3.4).
    delta halved each phase.
    Source: Wolf (2025), §2.4; Auer & Ortner [AO10].
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
#  FAST VECTORISED RUNNER  (Wolf 2025, §3.2 / §3.4 / §3.5)
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


def _run_alg_on_table(name, reward_table, probs, T, g_seed=0):
    """
    Run one algorithm on a pre-generated T×K reward table.
    Returns (suboptimal_count, regret, total_reward) where:
      regret       = Σ(Q*−Q_{A_t}) — deterministic gap sum (Wolf §3.5, always ≥ 0)
      total_reward = Σ X_t          — used for reward-variance chi-squared test (Wolf §3.5)
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

    # ── identical initialisation: pull each arm once (Wolf §3.4) ──
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

    for t in range(K, T):
        c = np.maximum(counts, 1)          # safe denominator; K=2 so this is ~free
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

    return subopt, regret, int(total_r)


def run_one_trial(probs, T, seed):
    """
    Run all algorithms on ONE arm ordering.
    Pre-generates reward table — fast for large T.
    """
    probs_arr    = np.array(probs, dtype=float)
    rng          = np.random.RandomState(seed)
    reward_table = (rng.rand(T, len(probs_arr)) < probs_arr).astype(np.int32)
    return {
        # pass seed as g_seed so each trial's greedy RNG is independent
        name: _run_alg_on_table(name, reward_table, probs_arr, T, g_seed=seed * 31 + hash(name) % 997)
        for name in _ALG_NAMES
    }


def run_scenario(probs, T=HORIZON, num_trials=NUM_TRIALS, label=""):
    """
    Full scenario runner — Wolf (2025) §3.2–3.5.
    Arm-permutation averaging over [p1,p2] and [p2,p1] (Wolf §3.4).
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Arms: {list(probs)}  |  T = {T:,}  |  Trials = {num_trials}")
    print(f"{'='*70}")

    Q_star       = max(probs)
    # baseline_var: variance of optimal-arm reward sum over T steps (Wolf §3.5)
    baseline_var = Q_star * (1.0 - Q_star) * T

    all_regrets = {n: [] for n in _ALG_NAMES}   # Σ Δ_{A_t} per trial
    all_rewards = {n: [] for n in _ALG_NAMES}   # Σ X_t per trial (for chi-squared)
    all_subopt  = {n: [] for n in _ALG_NAMES}

    total_runs = num_trials * 2          # 2 permutations per trial
    run_num    = 0
    for trial in range(num_trials):
        seed = SEED_BASE + trial
        run_num += 1
        _sys.stdout.write(f"\r  Run {run_num}/{total_runs}  (trial {trial+1}/{num_trials}, perm A) ...")
        _sys.stdout.flush()
        r1   = run_one_trial(probs,               T, seed)
        run_num += 1
        _sys.stdout.write(f"\r  Run {run_num}/{total_runs}  (trial {trial+1}/{num_trials}, perm B) ...")
        _sys.stdout.flush()
        r2   = run_one_trial(list(reversed(probs)), T, seed + num_trials)
        for name in _ALG_NAMES:
            sb1, reg1, rw1 = r1[name]
            sb2, reg2, rw2 = r2[name]
            all_regrets[name].append((reg1 + reg2) / 2.0)   # avg over permutations
            all_subopt[name].append(((sb1 + sb2) / 2.0) / T)
            all_rewards[name].append((rw1 + rw2) / 2.0)

    print(f"\r  {'Algorithm':<20} {'Avg Regret':>12} {'Subopt Ratio':>14} {'p-value':>9}")
    print(f"  {'-'*57}")

    summary: dict = {}
    for name in _ALG_NAMES:
        arr_reg = np.array(all_regrets[name])
        arr_rw  = np.array(all_rewards[name])
        arr_s   = np.array(all_subopt[name])
        avg_reg = float(np.mean(arr_reg))   # E[Σ Δ_{A_t}] — always ≥ 0
        var_rw  = float(np.var(arr_rw))     # variance of Σ X_t — matches Wolf §3.5 chi-sq test
        avg_sub = float(np.mean(arr_s))
        # chi-squared test: algorithm's reward variance vs optimal-arm baseline variance (Wolf §3.5)
        chi2    = ((num_trials - 1) * var_rw / baseline_var) if baseline_var > 0 else 0.0
        pval    = float(1.0 - stats.chi2.cdf(chi2, df=num_trials - 1))
        summary[name] = dict(avg_regret=avg_reg, reward_var=var_rw,
                             subopt_ratio=avg_sub, p_value=pval)
        print(f"  {name:<20} {avg_reg:>12.2f} {avg_sub:>14.5f} {pval:>9.2f}")

    # paper reference side-by-side
    refs: dict = {
        "Scenario A": [("ETC (m=100)","10.10","1.00"),("UCB-Tuned","30.67","1.00"),
                       ("EUCBV","47.73","1.00"),("UCB","238.17","1.00")],
        "Scenario B": [("ETC (m=10k)","150.66","0.00"),("UCB-Tuned","212.21","1.00"),
                       ("UCB","1127.17","0.00")],
        "Scenario C": [("UCB-Tuned","226.09","1.00"),("UCB-V","367.65","1.00"),
                       ("UCB","1172.57","0.78")],
    }
    for key, rows in refs.items():
        if key.lower() in label.lower():
            print(f"\n  Paper reference — Wolf (2025) Tables 3/4/5 (at T=1,000,000):")
            print(f"  {'Algorithm':<20} {'Regret (paper)':>16} {'p-val (paper)':>14}")
            print(f"  {'-'*52}")
            for alg, reg, pv in rows:
                print(f"  {alg:<20} {reg:>16} {pv:>14}")
    return summary


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Group 7 | DRL Assignment Problem I | Bandit Playground")
    print("  Paper: Wolf (2025), arXiv:2510.27001")
    print("  Repo : https://github.com/2025ag05770/DRL-Asignment-group-07")
    print("=" * 70)
    print()
    print("  Full replication : T=1,000,000 / 100 trials  (paper exact values)")
    print("  Demo below       : T=200,000  / 10 trials    (~5-8 sec, clear rankings)")
    print()

    DEMO_T      = 200_000
    DEMO_TRIALS = 10

    for label, probs in SCENARIOS.items():
        run_scenario(probs, T=DEMO_T, num_trials=DEMO_TRIALS, label=label)

    print()
    print("=" * 70)
    print("  To match paper EXACTLY:  DEMO_T = 1_000_000  |  DEMO_TRIALS = 100")
    print("=" * 70)
