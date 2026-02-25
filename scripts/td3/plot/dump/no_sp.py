from scripts.td3.plot.utils import plot_log, get_data, make_run_names_for_plot, map_run_name_to_token, token_to_color_dict, token_to_name_dict
import os 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



plot_groups = {
    "rnd=0.5-1": "Intrinsic Rewards", 
    "pn=0.2": "Pink Noise", 
    "default=1": "Default", 
}

os.makedirs("plots/td3/no_sp", exist_ok=True)

data, runs, log_names, model_paths = get_data("no_sp")




plot_metrics = [
    ("eval/strong/win_rate", "Win Rate (Strong Opponent)"), 
    ("charts/mean_episode_length", "Mean Episode Length"),
]
# cmap = {run: cmap_tab10(i) for i, run in enumerate(runs)})}
fig, axs = plt.subplots(len(plot_metrics), len(plot_groups), figsize=(len(plot_groups)*5, 5*len(plot_metrics)))
fig.suptitle("Agent performance against different opponents".title(), fontsize=16)


# force scientific notation with exponent 5
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((5, 5))  # force 10^5 scaling

max_steps = 400_000
smoothing_window = 10
less_smooth_window = 3
# first 25k steps are not evaluated because training starts only then 
start_xlim = 25_000 


def plot_metric(metric, y_label, axs):
    all_pr_values = set()
    for i, (plot_group, plot_group_name) in enumerate(plot_groups.items()):
        selected_runs = [r for r in runs if plot_group in r]

        pr_values = {r: map_run_name_to_token(r) for r in selected_runs}
        all_pr_values.update(set(pr_values.values()))

        cmap = {run: token_to_color_dict[pr_values[run]] for run in selected_runs}
        
        ax = axs[i]
        ax.set_title(plot_group_name)
        plot_log(data, selected_runs, metric, ax, cmap, smoothing_window=smoothing_window, less_smooth_window=less_smooth_window, alpha=0.3, max_steps=max_steps)
        ax.set_xlim(start_xlim, max_steps)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Steps")
        if i == 0:
            ax.set_ylabel(y_label)
    return all_pr_values

all_pr_values = set()
for i, (metric, y_label) in enumerate(plot_metrics):
    all_pr_values.update(plot_metric(metric, y_label, axs[i]))

handles = [plt.Line2D([0], [0], color=token_to_color_dict[pr_value], lw=2) for pr_value in all_pr_values]

filename = "plots/td3/no_sp/opponent_metrics.svg"
fig.legend(handles, [token_to_name_dict[v] for v in all_pr_values], loc="lower center", ncol=len(all_pr_values), bbox_to_anchor=(0.5, -0.15))



plt.savefig(filename, bbox_inches="tight")
print("Plot saved as", filename)