import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import json
from tqdm import tqdm
from datetime import datetime

# === Конфигурация ===
INPUT_FOLDER = "data/src"               # Папка с .nc файлами
OUTPUT_FOLDER = "data/frames"     # Папка для PNG
MANIFEST_PATH = "data/frames_manifest.json"
VARIABLE = "PWV"                               # Имя переменной в NetCDF
STEP_HOURS = 24                                # Отбор через каждые N часов

# === Создание директорий ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Список файлов ===
nc_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.nc")))
manifest_entries = []

for nc_file in nc_files:
    print(f"Обработка: {os.path.basename(nc_file)}")
    ds = xr.open_dataset(nc_file)
    ds = ds.sortby('lon')
    ds['lon'] = (ds['lon'] - 20) % 360
    ds = ds.sortby('lon')
    times = ds['timestamp'].values
    var = ds[VARIABLE]

    for i in tqdm(range(0, len(times), STEP_HOURS // 3)):  # каждый N часов (3h шаг)
        time = times[i]
        dt = pd.to_datetime(str(time))  # xarray timestamp → datetime
        timestamp = dt.strftime("%Y-%m-%d_%H%M")
        fname = f"{VARIABLE}_{timestamp}.png"
        fpath = os.path.join(OUTPUT_FOLDER, fname)

        # Получение среза и отрисовка
        data = var.sel(timestamp=time)
        plt.figure(figsize=(14.4, 7.2), dpi=100)
        plt.imshow(np.flipud(data.T.values), cmap="viridis", origin="lower",
                extent=[data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()])
        plt.axis('off')  # убираем оси
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
        plt.close()


        manifest_entries.append({
            "data": {
                "image": f"data/images/data/frames/{fname}"
            }
        })

# === Сохранение манифеста ===
# manifest_dir = os.path.dirname(MANIFEST_PATH)
# os.makedirs(manifest_dir, exist_ok=True)
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest_entries, f, indent=4, ensure_ascii=False)

print(f"\n✅ Завершено: сохранено {len(manifest_entries)} PNG и манифест.")
