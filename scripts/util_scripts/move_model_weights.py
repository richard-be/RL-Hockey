# CHAT GPT GENERATED, ONLY TO MOVE FILES 
import os
import shutil
src_root = "models/td3/HockeyOne-v0"


for filename in os.listdir(src_root):
    if os.path.isdir(os.path.join(src_root, filename)):
        continue
    if filename.endswith(".model"):
        sub_dir = f"models/td3/HockeyOne-v0/{filename.split('.')[0]}"

        os.makedirs(sub_dir, exist_ok=False)
        src_file = os.path.join(src_root, filename)

        dst_file = os.path.join(sub_dir, "1000000.model")

        print(f"Moving {src_file} -> {dst_file}")
        shutil.move(src_file, dst_file)
