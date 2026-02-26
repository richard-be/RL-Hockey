import sys
from pathlib import Path
import os 
from tueplots import bundles
from tueplots.constants.color import rgb
import matplotlib.pyplot as plt 

params = bundles.icml2024() 
params.update({"figure.dpi": 350})
plt.rcParams.update(params)