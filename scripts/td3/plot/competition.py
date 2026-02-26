
import os 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.td3.algorithm.evaluation import run_evaluation_multiple_runs
from hockey.hockey_env import Mode
import numpy as np 
import numpy as np
from matplotlib import colormaps
import seaborn as sns
import scripts.preamble

run_groups = {
    "Default": [
        ("td3:models/td3/HockeyOne-v0/final/sp/default/default_1_pr_0_sp_1__42__1772009398/1000000.model", "No PER"), 
        ("td3:models/td3/HockeyOne-v0/final/sp/default/default_1_pr_1_pr-intr-factor_0_sp_1__42__1772009398/1000000.model", "PER\n(TD-error)"), 
        ("td3:models/td3/HockeyOne-v0/final/sp/default/default_1_pr_1_pr-intr-factor_1_sp_1__42__1772009398/1000000.model", "PER\n(Intrinsic)"), 
    ], 
    "Pink Noise": [
        ("td3:models/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_0_sp_1__42__1772009398/1000000.model", "No PER"), 
        ("td3:models/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_1_pr-intr-factor_0_sp_1__42__1772009398/1000000.model", "PER\n(TD-error)"), 
        ("td3:models/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_1_pr-intr-factor_1_sp_1__42__1772009398/1000000.model", "PER\n(Intrinsic)"), 
    ], 
    # "Others": [
    #     ("sac:models/sac/sac_2_1_1000000_1771781495.pkl", "SAC"),
    #     ("crq:models/crossq/model.pkl", "CrossQ"), 
    # ],
    "Intr. Rew.": [
        ("td3:models/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_0_sp_1__42__1772009409/1000000.model", "No PER"), 
        # ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_0_sp_1__42__1772009409", "PER\n(TD-error)"), 
        # ("data/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_1_sp_1__42__1772009409", "PER\n(Intrinsic)"), 
        # ("data/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771317357", "PER\n(Intrinsic) (run 1)"),
    ], 
}   
path = f"plots/td3/cross_model_win_rates.pdf"

tab10 = plt.get_cmap("tab10")
per_label_to_color = {
    "No PER": tab10(0),
    "PER\n(TD-error)": tab10(1),
    "PER\n(Intrinsic)": tab10(2),
}

n_eval_episodes = 1000

player_paths = [model_path for group in run_groups.values() for model_path, _ in group]
eval_run_names = [run_label for group in run_groups.values() for _, run_label in group]
results = run_evaluation_multiple_runs(player_paths, n_episodes=n_eval_episodes, render=False, seed=42, hockey_mode=Mode.NORMAL)

win_rates = []
for player in player_paths: 
    player_results = results[player]
    def win_rate_for_opponent(opponent):
        if opponent not in player_results:
            return np.nan
        else: 
            return results[player][opponent]["win_rate"]
    win_rates.append([win_rate_for_opponent(opponent) for opponent in player_paths])

win_rates = np.array(win_rates)
win_rates *= 100 
win_rates[np.diag_indices_from(win_rates)] = np.nan # hide out diagonal because playing against itself is not meaningful

import json
json.dump({
    "win_rates": win_rates.tolist(),
    "eval_run_names": eval_run_names,
}, open(path[:-3] + "json", "w"))

fig, axs = plt.subplots(1, 1, figsize=(3.5,3.5))

ax = axs

cmap = "RdYlGn" # if metric == "win_rate" else "RdYlGn_r"
cmap = colormaps[cmap].copy() 
cmap.set_bad(color=(0.95, 0.95, 0.95)) # make nan light gray instead of white 

sns.heatmap(win_rates, cbar = False, annot=True, fmt=".0f", cmap=cmap, ax=ax, xticklabels=eval_run_names, yticklabels=eval_run_names, linewidths=0.4, linecolor=(0.2, 0.2, 0.2))


# ax.set_title("Win Rate (\%)")


ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 30))
ax.tick_params(axis='both', which='both', length=0)

for i, name in enumerate(eval_run_names):
    if not name in per_label_to_color: 
        continue
    color = per_label_to_color[name]
    
    ax.get_xticklabels()[i].set_color(color)
    ax.get_yticklabels()[i].set_horizontalalignment("center")
    ax.get_yticklabels()[i].set_fontweight("bold")
    ax.get_yticklabels()[i].set_color(color)
    
group_names = list(run_groups.keys())[:3]
group_centers = [0.5 + i for i in [1, 4, 6]]
group_lengths = [1.3, 1.3, 0.3]

bottom_y = ax.get_ylim()[0] 
for name, center, group_len in zip(group_names, group_centers, group_lengths):
    ax.text(-0.3, center, name, rotation=90, va="center", ha="center", fontweight="normal", clip_on=False)
    ax.plot([-0.6, -0.6], [center-group_len, center+group_len], color="black", lw=1, clip_on=False)

    ax.text(center, bottom_y+0.3 , name, va="center", ha="center", fontweight="normal", clip_on=False)
    ax.plot([center-group_len, center+group_len], [bottom_y+0.6, bottom_y+0.6], color="black", lw=1, clip_on=False)

    # ax.text(center, bottom_y+2.4, name, va="center", ha="center", fontsize=11, fontweight="normal", clip_on=False)
    # ax.plot([center-1.3, center+1.3], [bottom_y+2.2, bottom_y+2.2], color="black", lw=1.5, clip_on=False)

plt.savefig(path, bbox_inches="tight")
plt.savefig(path[:-3] + "png", bbox_inches="tight")
print("Plot saved as", path)
# fig.suptitle(f"Cross-model evaluation, {n_eval_episodes} games".title(), fontsize=16)
