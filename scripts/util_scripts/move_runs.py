# CHAT GPT GENERATED, ONLY TO MOVE FILES 
import os
import shutil

src_root = "runs"
dst_root = os.path.join("runs", "HockeyOne-v0")

os.makedirs(dst_root, exist_ok=True)

for run_name in os.listdir(src_root):
    run_path = os.path.join(src_root, run_name)

    if not os.path.isdir(run_path):
        continue
    if run_name == "HockeyOne-v0" or not run_name.startswith("HockeyOne-v0"):
        continue

    run_name = run_name.split("__", 1)[1]
    dst_file = os.path.join(dst_root, run_name)
    shutil.move(run_path, dst_file)
    # print("Moving", run_path, "->", dst_file)
