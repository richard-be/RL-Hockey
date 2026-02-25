# import os 
# if os.environ.get("preamble_run", None) is not None: 
#     print("Not re-runnning preamble")
# else: 
#     run -i ../preamble.py
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.td3.test_model import find_latest_time

def prettify_run_name(run):
    parts = run.split("_")
    prettified = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            param = parts[i]
            value = parts[i + 1].replace("x", ".")
            if param == "no": 
                prettified.append(f"no {value}")
            else:
                prettified.append(f"{param}={value}")

    return " | ".join(prettified)

def clean_run_name(run):
    base_name =  run.split("__")[0]
    return prettify_run_name(base_name)

def clean_and_unify_run(run, logs): 
    cleaned_run_name = clean_run_name(run)
    
    re_run_idx = 1
    run = cleaned_run_name
    while run in logs:
        run = f"{cleaned_run_name} [{re_run_idx}]"
        re_run_idx += 1
    return run

token_to_name_dict = {
    "pr=0": "No prioritized experieence replay (PER)", 
    "pr-intr-factor=0": "PER with TD-error",
    "pr-intr-factor=1": "PER with intrinsic reward",
}

pr_token_to_short_name_dict = {
    "pr=0": "No PER", 
    "pr-intr-factor=0": "PER/TD-error",
    "pr-intr-factor=1": "PER/intrinsic reward",
}

cmap_tab10 = plt.get_cmap("tab10")


token_to_color_dict = {
    "pr=0": cmap_tab10(0),
    "pr-intr-factor=0": cmap_tab10(1),
    "pr-intr-factor=1": cmap_tab10(2),
}

def map_run_name_to_token(run_name):
    matched_tokens = [token for token in token_to_name_dict.keys() if token in run_name]
    if len(matched_tokens) > 1:
        raise ValueError(f"Multiple tokens found in run name '{run_name}': {matched_tokens}")
    return matched_tokens[0]

def make_run_names_for_plot(run_names): 
    tokens = [n.split(" | ") for n in run_names]
    tokens_flattened = []
    for token_list in tokens:
        tokens_flattened.extend(token_list)
    non_unique_tokens = set([t for t in tokens_flattened if tokens_flattened.count(t) > 1])
    
    new_names = []
    for name in run_names:
        new_name = name
        for token in non_unique_tokens:
            new_name = new_name.replace(token, "")
        for old_token, new_token in token_to_name_dict.items():
            new_name = new_name.replace(old_token, new_token)
        new_names.append(new_name.strip(" | "))
    return new_names

def get_data(exp_name, parent_dir="final"): 
    data_dir = f"./data/td3/HockeyOne-v0/{parent_dir}/{exp_name}"

    data = dict()
    log_names = set()
    model_paths = dict()

    for experiment in os.listdir(data_dir):
        for run in os.listdir(os.path.join(data_dir, experiment)):

            run_name = clean_and_unify_run(run, data)
            data[run_name] = dict()

            model_dir = f"models/td3/HockeyOne-v0/{parent_dir}/{exp_name}/{experiment}/{run}"
            checkpoint = find_latest_time("*.model", model_dir)

            model_paths[run_name] = f"td3:{model_dir}/{checkpoint}.model"
            for file in os.listdir(os.path.join(data_dir, experiment, run)):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(data_dir, experiment, run, file))
                    log_name = file[:-4].replace("-", "/")
                    data[run_name][log_name] = df
                    log_names.add(log_name)
    return data, sorted(data.keys()), log_names, model_paths

# all_runs = sorted(data.keys())

# print("All runs:")
# for run in all_runs: 
#     print(f"  {run}")

# runs = [r for r in all_runs if "pn=0.2" in r]
# runs = [r for r in all_runs if "rnd=0.5-1" in r]
# runs = [r for r in all_runs if not "pn=0.2" in r and not "rnd=" in r and not "sp=0" in r]
# runs = [r for r in all_runs if (r == "pn=0.2 | sp=1" or r == "rnd=0.5-1 | sp=1") or (not "pn=0.2" in r and not "rnd=" in r and not "sp=0" in r)]
# runs = all_runs.copy()

# print("Runs selected:")
# for run in runs: 
#     print(f"  {run}")

def plot_log(data, runs, log_name, ax, cmap, smoothing_window=100, less_smooth_window=10, alpha=0.2, max_steps=None):
    for run in runs:
        if log_name in data[run]:
            df = data[run][log_name]
            if max_steps is not None:
                df = df[df["step"] <= max_steps]

            less_smoothed_value = df["value"].rolling(window=less_smooth_window, center=True).mean()
            smoothed_value = df["value"].rolling(window=smoothing_window, center=True).mean()
            ax.plot(df["step"], less_smoothed_value, label=run, color=cmap[run], alpha=alpha)
            ax.plot(df["step"], smoothed_value, label=run, color=cmap[run])
