# Deep Reinforcement Learning Assignment — Group 7

This project reproduces experiments comparing classical and variance-aware multi-armed bandit algorithms on Bernoulli reward distributions. It follows the evaluation protocol described in Elise Wolf's 2025 paper, *A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms* ([arXiv:2510.27001](https://arxiv.org/abs/2510.27001)).

## What the simulation does

The runner evaluates ten algorithms in each of three two-arm scenarios:

| Scenario | Arm 1 reward probability | Arm 2 reward probability | Description |
|---|---:|---:|---|
| A | 0.800 | 0.900 | Baseline |
| B | 0.895 | 0.900 | Low-variance micro-gap |
| C | 0.890 | 0.895 | High-variance micro-gap |

Algorithms compared:

- Explore-Then-Commit (ETC), with exploration lengths of 100 and 10,000
- Epsilon-Greedy, with epsilon 0.05 and 0.5
- UCB1
- UCB-Tuned
- UCB-V
- EUCBV
- PAC-UCB
- UCB-Improved

The experiment uses a shared reward table for all algorithms within a trial, pulls each arm once for initialization, uses deterministic trial seeds, and averages results over both arm orderings. These choices help make comparisons consistent and reproducible.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- pandas
- Matplotlib

Install the dependencies:

```bash
python -m pip install numpy scipy pandas matplotlib
```

## Run

From the project directory:

```bash
python Group_7_DRL_ASSIGNMENT.py
```

By default, the script runs 10 trials at a horizon of 100,000 steps. To configure the experiment, pass:

```bash
python Group_7_DRL_ASSIGNMENT.py --T 100000 --trials 10 --workers 4
```

Arguments:

| Argument | Default | Description |
|---|---:|---|
| `--T` | `100000` | Number of time steps per trial |
| `--trials` | `10` | Number of independent trials |
| `--workers` | System default | Number of multiprocessing workers |

For a longer run matching the paper's stated horizon and number of trials:

```bash
python Group_7_DRL_ASSIGNMENT.py --T 1000000 --trials 100
```

This exact-scale run is computationally intensive: it evaluates every algorithm for all three scenarios, trials, and arm orderings.

## Results

The script writes two files to the current working directory:

- `results_table.csv` — one row per scenario and algorithm. Includes average and standard deviation of regret, suboptimal-pull ratio, reward-variance chi-squared-test p-value, paper reference regret when available, horizon, and trial count.
- `regret_curves.png` — a three-panel, log-log plot of average cumulative regret across the scenarios.

Regret is computed as the cumulative expected reward gap between the optimal arm and the selected arms. Because regret scales with the horizon, compare algorithm rankings rather than raw regret values when using smaller horizons than the paper.

## Project files

- `Group_7_DRL_ASSIGNMENT.py` — simulation, algorithm implementations, and result generation.
- `Group_7_DRL_ASSIGNMENT.pptx` — assignment presentation.
- `results_table.csv` — included example results.
- `regret_curves.png` — included example regret plot.

## Reproducibility notes

The script defaults to a 100,000-step horizon and 10 trials, while the paper-scale configuration is 1,000,000 steps and 100 trials. Trial seeds are deterministic, and arm-ordering averages reduce tie-breaking bias. Output files are overwritten when the script is run again, so save or rename existing results before starting a run whose outputs you want to preserve.
