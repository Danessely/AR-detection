"""
Анализирует значения переменной PWV в .nc-файлах, строит гистограмму, вычисляет перцентили и сохраняет результаты.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


# константы
INPUT_FOLDER = "data/src"
VARIABLE = "PWV"
OUTPUT_PATH = "data/global_scale.json"
PLOT_PATH = "data/value_distribution.png"
BINS = 2048  # точность
PERCENTILES = (1, 99)

hist = np.zeros(BINS, dtype=np.int64)
bin_edges = None

print("🔍 Пофайловая агрегацию...")

nc_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.nc")))

for nc_file in tqdm(nc_files, desc="Файлы"):
    ds = xr.open_dataset(nc_file)
    var = ds[VARIABLE]

    # Покадровая агрегация по времени
    for t in tqdm(range(len(var.timestamp)), leave=False, desc="  Срезы"):
        frame = var.isel(timestamp=t).values
        frame = frame[~np.isnan(frame)]

        if bin_edges is None:
            vmin0, vmax0 = np.percentile(frame, [0.1, 99.9])
            bin_edges = np.linspace(vmin0, vmax0, BINS + 1)

        h, _ = np.histogram(frame, bins=bin_edges)
        hist += h

# Вычисление перцентилей из гистограммы
cdf = np.cumsum(hist)
cdf = cdf / cdf[-1]

vmin = np.interp(PERCENTILES[0] / 100, cdf, bin_edges[1:])
vmax = np.interp(PERCENTILES[1] / 100, cdf, bin_edges[1:])

# Сохраняем результат
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"vmin": float(vmin), "vmax": float(vmax)}, f, indent=2)

print(f"\n✅ vmin = {vmin:.2f}, vmax = {vmax:.2f}")
print(f"💾 Сохранено в {OUTPUT_PATH}")

plt.figure(figsize=(10, 5))
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.plot(centers, hist, label="Value histogram")
plt.axvline(vmin, color="red", linestyle="--", label=f"{PERCENTILES[0]}th: {vmin:.1f}")
plt.axvline(
    vmax, color="green", linestyle="--", label=f"{PERCENTILES[1]}th: {vmax:.1f}"
)
plt.xlabel(f"{VARIABLE} value")
plt.ylabel("Frequency")
plt.title(f"Distribution of {VARIABLE} values")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig(PLOT_PATH)
plt.close()
print(f"🖼️ Сохранено распределение в: {PLOT_PATH}")
