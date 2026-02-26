
from scripts.td3.plot.plot import plot

run_groups = {
    "Default": [
        ("data/td3/HockeyOne-v0/final/generalization/default/default_1_pr_0_no_strong-opp__42__1772011206", "No PER"),
        ("data/td3/HockeyOne-v0/final/generalization/default/default_1_pr_1_pr-intr-factor_0_no_strong-opp__42__1772011202", "PER (TD-error)"),
        ("data/td3/HockeyOne-v0/final/generalization/default/default_1_pr_1_pr-intr-factor_1_no_strong-opp__42__1772011206", "PER (Intrinsic Rewards)"), 
    ], 
    "Pink Noise": [
        ("data/td3/HockeyOne-v0/final/generalization/pn/pn_0x2_pr_0_no_strong-opp__42__1772011188", "No PER"),
        ("data/td3/HockeyOne-v0/final/generalization/pn/pn_0x2_pr_1_pr-intr-factor_0_no_strong-opp__42__1772011188", "PER (TD-error)"),
        ("data/td3/HockeyOne-v0/final/generalization/pn/pn_0x2_pr_1_pr-intr-factor_1_no_strong-opp__42__1772011185", "PER (Intrinsic Rewards)")
    ], 
    "Intrinsic Rewards": [
        ("data/td3/HockeyOne-v0/final/generalization/rnd/rnd_0x5-1_pr_0_no_strong-opp__42__1772011158", "No PER"),
        ("data/td3/HockeyOne-v0/final/generalization/rnd/rnd_0x5-1_pr_1_pr-intr-factor_0_no_strong-opp__42__1772011159", "PER (TD-error)"),
        ("data/td3/HockeyOne-v0/final/generalization/rnd/rnd_0x5-1_pr_1_pr-intr-factor_1_no_strong-opp__42__1772011159", "PER (Intrinsic Rewards)")
    ], 
}

metrics = [
    {"key": "eval/strong/win_rate", "metric_label": "Win Rate (Strong Opponent)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    {"key": "eval/weak/win_rate", "metric_label": "Win Rate (Weak Opponent)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    {"key": "eval/crossq0/win_rate", "metric_label": "Win Rate (CrossQ)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    {"key": "eval/sac1m/win_rate", "metric_label": "Win Rate (SAC1m)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    # {"key": "charts/mean_episode_length", "metric_label": "Mean Episode Length", "smoothing_windows": [200, 100]},
]
outpath = "plots/td3/generalization.pdf"

plot(run_groups, metrics, outpath, max_steps=400_000)