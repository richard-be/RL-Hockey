import os
import shutil

ROOT_DIR = "outputs/mbpo/default/gym___Hockey-One-v0"

for date_name in os.listdir(ROOT_DIR):
    date_path = os.path.join(ROOT_DIR, date_name)
    if not os.path.isdir(date_path):
        continue

    for time_name in os.listdir(date_path):
        time_path = os.path.join(date_path, time_name)

        if not os.path.isdir(time_path):
            continue

        results_csv = os.path.join(time_path, "results.csv")

        if os.path.isfile(results_csv):
            if os.path.getsize(results_csv) == 0:
                print(f"Deleting folder: {time_path}")
                shutil.rmtree(time_path)