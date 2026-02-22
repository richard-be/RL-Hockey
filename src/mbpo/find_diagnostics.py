import os
import sys

def find_file(root_folder, target_filename):
    results = []
    for root, dirs, files in os.walk(root_folder):
        matched_files = [os.path.join(root, f) for f in files if target_filename in f]
        # if target_filename in files:
        results += matched_files
    return results

if __name__ == "__main__":
    root_folder = "outputs/mbpo/default/gym___Hockey-One-v0/"
    filename = ".mp4"

    result = find_file(root_folder, filename)

    if len(result) > 0:
        print("Files:")
        print('\n'.join(result))
    else:
        print("File not found.")
