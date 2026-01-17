import sys
from pathlib import Path
import os 
import matplotlib.pyplot as plt 

os.environ["preamble_run"] = "True" 

# add project root to sys path to be able to import ./src in notebooks that are in ./experiments
project_root = Path.cwd().parent.parent
sys.path.append(str(project_root))

# change working directory to root so we can open data/file.csv from notebooks in ./experiments 
# and dont have to open ../data/file.csv
os.chdir(project_root)

print("Moved to", project_root)