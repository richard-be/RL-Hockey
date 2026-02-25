
from scripts.td3.plot.utils import plot_log, get_data, make_run_names_for_plot, pr_token_to_short_name_dict, token_to_color_dict, token_to_name_dict
import os 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.td3.algorithm.evaluation import run_evaluation_multiple_runs
from hockey.hockey_env import Mode
import numpy as np 
import numpy as np
from matplotlib import colormaps
import seaborn as sns

plot_groups = {
    "rnd=0.5-1": "Intrinsic Rewards", 
    "pn=0.2": "Pink Noise", 
    "default=1": "Default", 
}

os.makedirs("plots/td3/sp", exist_ok=True)

data, runs, log_names, model_paths = get_data("sp")

def get_run_name(run): 
    run_name = ""
    for group_token, gn in plot_groups.items():
        if group_token in run:
            run_name += gn
    for pr_token, short_name in pr_token_to_short_name_dict.items():
        if pr_token in run:
            run_name += " - " + short_name
    return run_name

player_paths = [model_paths[run] for run in runs]
player_paths = player_paths + ["weak_opponent", "strong_opponent"]
eval_run_names = [get_run_name(run) for run in runs] + ["weak opponent", "strong opponent"]

n_eval_episodes = 10

results = run_evaluation_multiple_runs(player_paths, n_episodes=n_eval_episodes, render=False, seed=42, hockey_mode=Mode.NORMAL)
scores = {}

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


fig, axs = plt.subplots(1, 1, figsize=(8, 8))

ax = axs

cmap = "RdYlGn" # if metric == "win_rate" else "RdYlGn_r"
cmap = colormaps[cmap].copy() 
cmap.set_bad(color=(0.95, 0.95, 0.95)) # make nan light gray instead of white 

sns.heatmap(win_rates, cbar = False, annot=True, fmt=".0f", cmap=cmap, ax=ax, xticklabels=eval_run_names, yticklabels=eval_run_names)
ax.set_title("Win Rate (%)")


path = f"plots/td3/sp/cross_model_evaluation.svg"
plt.savefig(path, bbox_inches="tight")
print("Plot saved as", path)

# fig.suptitle(f"Cross-model evaluation, {n_eval_episodes} games".title(), fontsize=16)
