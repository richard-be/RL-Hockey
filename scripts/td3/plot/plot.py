import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib
import scripts.preamble
# matplotlib.use("pgf")
import matplot2tikz

def _plot_metric(df, ax, label, color, smoothing_windows, smoothing_alphas=[1, 0.3], vertical_line=None, range=None, log_scale=False, **kwargs):
    for i, (smoothing_window, alpha) in enumerate(zip(smoothing_windows, smoothing_alphas)):
        smoothed_value = df["value"].rolling(window=smoothing_window, center=True).mean()
        ax.plot(df["step"], smoothed_value, label=label if i==0 else None, alpha=alpha, color=color)
    
        if range is not None:
            ax.set_ylim(range)
        if log_scale:
            ax.set_yscale("log")
        if vertical_line is not None and i == 0: 
            # first step where smoothed value reaches the vertical line value
            step_reaching_line = df[smoothed_value >= vertical_line]["step"].min() 
            print(f"{label} reached value at step {step_reaching_line}")
            ax.axvline(x=step_reaching_line, ymax=vertical_line, color=color, linestyle="--", linewidth=1)

def plot(run_groups, metrics, outpath, max_steps=None, cmap=None, cell_width=4, cell_height=2):
    n_groups = len(run_groups)
    n_metrics = len(metrics)

    if n_groups == 1:
        fig, axs = plt.subplots(1, n_metrics, figsize=(cell_width*n_metrics, cell_height))
        only_one_group = True
    else: 
        fig, axs = plt.subplots(n_metrics, n_groups, figsize=(n_groups*cell_width, n_metrics*cell_height))
        only_one_group = False
    
    # only_one_group = only_one_group or n_metrics == 1

    cmap = plt.get_cmap("tab10") if cmap is None else cmap

    for group_idx, (group_name, runs) in enumerate(run_groups.items()):
        for run_idx, (run, run_label) in enumerate(runs):
            for metric_idx, metric_dict in enumerate(metrics):
                metric = metric_dict["key"]
                metric_label = metric_dict["metric_label"]

                df = pd.read_csv(f"{run}/{metric.replace('/', '-')}.csv")
                if max_steps is not None:
                    df = df[df["step"] <= max_steps]

                if n_metrics == 1: 
                    ax = axs[group_idx]
                elif only_one_group:
                    ax = axs[metric_idx]
                else:                
                    ax = axs[metric_idx, group_idx]

                ax.grid(True, alpha=0.3)
                if metric_idx == 0:
                    ax.set_title(group_name)
                if metric_idx == n_metrics - 1 or only_one_group:
                    ax.set_xlabel("Step")  
                # only for first column
                if group_idx == 0:
                    ax.set_ylabel(metric_label)

                # disable metrics for certain runs 
                if run_label in metric_dict.get("skip", []):
                    continue
                _plot_metric(df, ax, label=run_label if metric_idx==0 and group_idx==0 else None, color=cmap(run_idx), **metric_dict)
    fig.legend(loc="lower center", ncol=n_metrics if only_one_group else n_groups, framealpha=1)
    plt.savefig(outpath)
    plt.savefig(outpath[:-3] + "png")
    print("Plot saved as", outpath)

    # matplot2tikz.save(outpath[:-4] + ".tikz", figure=fig)