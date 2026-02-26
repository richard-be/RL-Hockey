
from scripts.td3.plot.plot import plot

run_groups = {
    "Default": [
        ("data/td3/HockeyOne-v0/final/no_sp/default/default_1_pr_0__42__1771959788", "No PER"),
        ("data/td3/HockeyOne-v0/final/no_sp/default/default_1_pr_1_pr-intr-factor_0__42__1771959788", "PER (TD-error)"),
        ("data/td3/HockeyOne-v0/final/no_sp/default/default_1_pr_1_pr-intr-factor_1__42__1771959788", "PER (Intrinsic Rewards)"), 
    ], 
    "Pink Noise": [
        ("data/td3/HockeyOne-v0/final/no_sp/pn/pn_0x2_pr_0__42__1771959767", "No PER"),
        ("data/td3/HockeyOne-v0/final/no_sp/pn/pn_0x2_pr_1_pr-intr-factor_0__42__1771959767", "PER (TD-error)"),
        ("data/td3/HockeyOne-v0/final/no_sp/pn/pn_0x2_pr_1_pr-intr-factor_1__42__1771959767", "PER (Intrinsic Rewards)")
    ], 
    "Intrinsic Rewards": [
        ("data/td3/HockeyOne-v0/final/no_sp/rnd/rnd_0x5-1_pr_0__42__1771959759", "No PER"), 
        ("data/td3/HockeyOne-v0/final/no_sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_0__42__1771959759", "PER (TD-error)"),
        ("data/td3/HockeyOne-v0/final/no_sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_1__42__1771959759", "PER (Intrinsic Rewards)"), 
    ],
}

metrics = [
    {"key": "eval/strong/win_rate", "metric_label": "Win Rate (Strong Opponent)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    {"key": "charts/mean_episode_length", "metric_label": "Mean Episode Length", "smoothing_windows": [200, 100]},
]
outpath = "plots/td3/no_self_play.pdf"

plot(run_groups, metrics, outpath, max_steps=400_000, cell_width=2, cell_height=1.7)