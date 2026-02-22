# CHAT GPT GENERATED, ONLY TO MOVE FILES 
import os
import shutil

src_root = "runs"
dst_root = os.path.join("models", "td3")

os.makedirs(dst_root, exist_ok=True)

for run_name in os.listdir(src_root):
    run_path = os.path.join(src_root, run_name)

    if not os.path.isdir(run_path):
        continue

    # find *.cleanrl_model file inside run directory
    for filename in os.listdir(run_path):
        if filename.endswith(".cleanrl_model"):
            src_file = os.path.join(run_path, filename)

            dst_file = os.path.join(dst_root, f"{run_name}.model")

            print(f"Moving {src_file} -> {dst_file}")
            shutil.move(src_file, dst_file)
