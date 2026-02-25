from scripts.td3.plot.utils import plot_log, get_data, make_run_names_for_plot
import os 
import matplotlib.pyplot as plt



def plot_opponent_metrics(data, selected_runs, exp_name, log_names, cmap, outdir, max_steps=400_000):
    opponent_eval = [l for l in log_names if "eval/" in l]
    opponents = set([l.split("/")[1] for l in opponent_eval])
    opponent_metrics = ["win_rate", "draw_rate"]
    # opponent_metrics = ["win_rate", "lose_rate", "draw_rate"]

    fig, axs = plt.subplots(len(opponent_metrics), len(opponents), figsize=(len(opponents)*5, len(opponent_metrics)*5))
    fig.suptitle("Agent performance against different opponents".title(), fontsize=16)

    # n_steps = data[selected_runs[0]]["eval/strong/win_rate"]["step"].max()

    for opponent_idx, opponent in enumerate(sorted(opponents)):
        if len(opponent_metrics) == 1:
            axs[opponent_idx].set_title(opponent.title())
        else:
            axs[0, opponent_idx].set_title(opponent.title())

        metric_axs = [axs[opponent_idx]] if len(opponent_metrics) == 1 else axs[:, opponent_idx]
        for metric, ax in zip(opponent_metrics, metric_axs):
            ax.set_ylabel(metric.replace("_", " ").title())
            plot_log(data, selected_runs, f"eval/{opponent}/{metric}", ax, cmap, smoothing_window=10, less_smooth_window=3, alpha=0.3, max_steps=max_steps)
            # ax.set_xscale("log")
            ax.set_xlim(0, max_steps)
    handles = [plt.Line2D([0], [0], color=cmap[run], lw=2) for run in selected_runs]

    fig.legend(handles, make_run_names_for_plot(selected_runs), loc="lower center", ncol=len(selected_runs))
    # fig.legend(runs, loc="lower center", ncol=len(data), )
    path = f"{outdir}/opponent_metrics_{exp_name}.svg"
    plt.savefig(path, bbox_inches="tight")
    print("Plot saved as", path)

experiments = ["sp"]# ["sp", "no_sp", # "generalization"]
plot_groups = ["default=1", "rnd=0.5-1"] #["rnd=0.5-1", "pn=0.2", "default=1"]

for experiment in experiments:
    os.makedirs("plots/td3/"+experiment, exist_ok=True)

    data, runs, log_names, model_paths = get_data(experiment)
    cmap_tab10 = plt.get_cmap("tab10")

    # cmap = {run: cmap_tab10(i) for i, run in enumerate(runs)})}
    for plot_group in plot_groups:
        selected_runs = [r for r in runs if plot_group in r]
        cmap = {run: cmap_tab10(i) for i, run in enumerate(selected_runs)}
        plot_opponent_metrics(data, selected_runs, plot_group, log_names, cmap, outdir=f"plots/td3/{experiment}")