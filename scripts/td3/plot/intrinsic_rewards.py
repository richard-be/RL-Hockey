from scripts.td3.plot.plot import plot
import matplotlib.pyplot as plt

run_groups = {
    # "Intrinsic Rewards (0.5 / 1)": [
    #     ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771317357", "0.5 / 1"), 
    #     ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771854378", "0.5 / 1 (run 2)"),
    #     ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771859562", "0.5 / 1 (run 3)"),
    #     ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_0_sp_1__42__1772009409", "0.5 / 1 (run 4)"),
    # ], 
    # "Intrinsic Rewards (1 / 0)": [
    #     ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_1-0_sp_1__42__1771317357", "(1 / 0)"), 
    #     ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_1-0_sp_1__42__1771854378", "(1 / 0) (run 2)"),
    # ]
    "": [
        ("data/td3/HockeyOne-v0/final/no_sp/default/default_1_pr_0__42__1771959788", "Gaussian noise only"),
        ("data/td3/HockeyOne-v0/final/no_sp/rnd/rnd_0x5-1_pr_0__42__1771959759", "Intrinsic + extrinsic rewards"),
        ("data/td3/HockeyOne-v0/final/no_sp/rnd/rnd_1-0_pr_0__42__1772039473", "Intrinsic rewards only")
    ]
}   

metrics = [
    {"key": "eval/strong/win_rate", "metric_label": "Win Rate (Strong Opponent)", "smoothing_windows": [10, 3], "vertical_line": 0.9}, 
    {"key": "eval/strong/draw_rate", "metric_label": "Draw Rate (Strong Opponent)", "smoothing_windows": [10, 3]}, 
    {"key": "charts/mean_episode_length", "metric_label": "Mean Episode Length", "smoothing_windows": [200, 100]},
    {"key": "charts/intrinsic_reward", "metric_label": "Mean Intrinsic Reward", "smoothing_windows": [100, 10], "skip": ["Gaussian noise only"], "log_scale": True},
    # {"key": "charts/extrinsic_reward", "metric_label": "Mean Extrinsic Reward", "smoothing_windows": [10, 3]},
]
outpath = "plots/td3/rnd.pdf"

plot(run_groups, metrics, outpath, max_steps=400_000, cmap=plt.get_cmap("Set2"), cell_width=1.5, cell_height=1.8)