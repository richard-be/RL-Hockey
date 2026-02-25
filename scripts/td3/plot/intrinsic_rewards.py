from scipts.td3.plot.plot import plot


run_groups = {
    "Intrinsic Rewards (0.5 / 1)": [
        ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771317357", "0.5 / 1"), 
        ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771854378", "0.5 / 1 (run 2)"),
        ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771859562", "0.5 / 1 (run 3)"),
        ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_0_sp_1__42__1772009409", "0.5 / 1 (run 4)"),
    ], 
    "Intrinsic Rewards (1 / 0)": [
        ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_1-0_sp_1__42__1771317357", "(1 / 0)"), 
        ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_1-0_sp_1__42__1771854378", "(1 / 0) (run 2)"),
    ]
}   

metrics = [
    {"key": "eval/strong/win_rate", "metric_label": "Win Rate (Strong Opponent)", "smoothing_windows": [10, 3], "vertical_line": 0.8}, 
    {"key": "charts/mean_episode_length", "metric_label": "Mean Episode Length", "smoothing_windows": [200, 100]},
]
outpath = "plots/td3/test.svg"

plot(run_groups, metrics, outpath)