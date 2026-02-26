from scripts.td3.plot.plot import plot

run_groups = {
    "Default": [
        ("data/td3/HockeyOne-v0/final/sp/default/default_1_pr_0_sp_1__42__1772009398", "No PER"), 
        ("data/td3/HockeyOne-v0/final/sp/default/default_1_pr_1_pr-intr-factor_0_sp_1__42__1772009398", "PER (TD-error)"), 
        ("data/td3/HockeyOne-v0/final/sp/default/default_1_pr_1_pr-intr-factor_1_sp_1__42__1772009398", "PER (Intrinsic Rewards)"), 
    ], 
    "Pink Noise": [
        ("data/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_0_sp_1__42__1772009398", "No PER"), 
        ("data/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_1_pr-intr-factor_0_sp_1__42__1772009398", "PER (TD-error)"), 
        ("data/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_1_pr-intr-factor_1_sp_1__42__1772009398", "PER (Intrinsic Rewards)"), 
    ], 
    "Intrinsic Rewards": [
        ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_0_sp_1__42__1772009409", "No PER"), 
        ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_0_sp_1__42__1772009409", "PER (TD-error)"), 
        ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_1_sp_1__42__1772009409", "PER (Intrinsic Rewards)"), 
        # ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771317357", "PER (Intrinsic Rewards) (run 1)"),
    ], 
}   

metrics = [
    {"key": "eval/strong/win_rate", "metric_label": "Win Rate (Strong Opponent)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    {"key": "charts/mean_episode_length", "metric_label": "Mean Episode Length", "smoothing_windows": [200, 100], "range": (0, 256)},
    # {"key": "charts/intrinsic_reward", "metric_label": "Mean Intrinsic Reward", "smoothing_windows": [100, 10]},
    # {"key": "charts/extrinsic_reward", "metric_label": "Mean Extrinsic Reward", "smoothing_windows": [10, 3]},
]
outpath = "plots/td3/self_play2.pdf"

plot(run_groups, metrics, outpath, cell_width=2, cell_height=1.7)