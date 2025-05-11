import xarray as xr
import numpy as np
from PIL import Image
import os
import glob
import json
from tqdm import tqdm
import pandas as pd

# === Конфигурация ===
INPUT_FOLDER = "data/src"               # Папка с .nc файлами
OUTPUT_FOLDER = "data/frames_pil"     # Папка для PNG
MANIFEST_PATH = "data/frames_pil_manifest.json"
VARIABLE = "PWV"                               # Имя переменной в NetCDF
STEP_HOURS = 24                                # Отбор через каждые N часов

# === Создание директорий ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Список файлов ===
nc_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.nc")))
manifest_entries = []

def normalize(data):
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    return ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)

for nc_file in nc_files:
    print(f"Обработка: {os.path.basename(nc_file)}")
    ds = xr.open_dataset(nc_file)
    ds = ds.sortby('lon')
    ds['lon'] = (ds['lon'] - 20) % 360
    ds = ds.sortby('lon')
    times = ds['timestamp'].values
    var = ds[VARIABLE]

    for i in tqdm(range(0, len(times), STEP_HOURS // 3)):
        time = times[i]
        dt = pd.to_datetime(str(time))
        timestamp = dt.strftime("%Y-%m-%d_%H%M")
        fname = f"{VARIABLE}_{timestamp}.png"
        fpath = os.path.join(OUTPUT_FOLDER, fname)

        data = var.sel(timestamp=time)
        array = data.T.values  # север сверху
        image_array = normalize(array)

        img = Image.fromarray(image_array)
        img.save(fpath)

        manifest_entries.append({
            "data": {
                "image": f"data/images/data/frames/{fname}"
            }
        })
    break

# === Сохранение манифеста ===
# manifest_dir = os.path.dirname(MANIFEST_PATH)
# os.makedirs(manifest_dir, exist_ok=True)
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest_entries, f, indent=4, ensure_ascii=False)

print(f"\n✅ Завершено: сохранено {len(manifest_entries)} PNG и манифест.")
