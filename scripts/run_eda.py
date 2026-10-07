"""Regenerate every EDA figure + table from the plot registry (DRY: same
functions the notebook uses)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.load import load_dev_application
from src.visualization.eda_plots import ALL_PLOTS

if __name__ == "__main__":
    dev = load_dev_application()
    print(f"Dev set: {dev.shape}")
    for fn in ALL_PLOTS:
        fig, takeaway = fn(dev, save=True)
        plt.close(fig)
        print(f"  ✓ {fn.__name__}: {takeaway}")
    print("\nAll figures -> reports/figures/, tables -> reports/tables/")
