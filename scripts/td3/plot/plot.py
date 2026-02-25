import matplotlib.pyplot as plt
import pandas as pd 

def _plot_metric(df, ax, label, color, smoothing_windows, smoothing_alphas=[1, 0.3], vertical_line=None, **kwargs):
    for i, (smoothing_window, alpha) in enumerate(zip(smoothing_windows, smoothing_alphas)):
        smoothed_value = df["value"].rolling(window=smoothing_window, center=True).mean()
        ax.plot(df["step"], smoothed_value, label=label if i==0 else None, alpha=alpha, color=color)
    
        if vertical_line is not None and i == 0: 
            # first step where smoothed value reaches the vertical line value
            step_reaching_line = df[smoothed_value >= vertical_line]["step"].min() 
            ax.axvline(x=step_reaching_line, ymax=vertical_line, color=color, linestyle="--", linewidth=1)

def plot(run_groups, metrics, outpath, max_steps=None):
    n_groups = len(run_groups)
    n_metrics = len(metrics)

    fig, axs = plt.subplots(n_metrics, n_groups, figsize=(n_groups*5, 5*n_metrics))
    only_one_group = n_groups == 1
    cmap = plt.get_cmap("tab10")

    for group_idx, (group_name, runs) in enumerate(run_groups.items()):
        for run_idx, (run, run_label) in enumerate(runs):
            for metric_idx, metric_dict in enumerate(metrics):
                metric = metric_dict["key"]
                metric_label = metric_dict["metric_label"]

                df = pd.read_csv(f"{run}/{metric.replace('/', '-')}.csv")

                if max_steps is not None:
                    df = df[df["step"] <= max_steps]

                if only_one_group:
                    ax = axs[metric_idx]
                else:                
                    ax = axs[metric_idx, group_idx]
                
                if metric_idx == 0:
                    ax.set_title(group_name)
                if metric_idx == n_metrics - 1:
                    ax.set_xlabel("Steps")  
                # only for first column
                if group_idx == 0:
                    ax.set_ylabel(metric_label)

                _plot_metric(df, ax, label=run_label if metric_idx==0 and group_idx==0 else None, color=cmap(run_idx), **metric_dict)
    fig.legend(loc="lower center", ncol=len(run_groups))
    plt.savefig(outpath)
    print("Plot saved as", outpath)
