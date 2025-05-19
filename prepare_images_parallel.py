#!/usr/bin/env python3
"""
Генерирует три версии кадров из NetCDF в мультипоточном режиме:
  1) RGB с цветной картой;
  2) Grayscale;
  3) Схему для разметки.
"""

import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# matplotlib до импорта pyplot
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import cm

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from tqdm import tqdm

# Константы
INPUT_DIR = Path("data/src")
COLOR_DIR = Path("data/frames/frames_pil_color")
GRAY_DIR = Path("data/frames/frames_pil_gray")
VIZ_DIR = Path("data/frames/frames_pil_scheme")
MANIFEST_PATH = Path("data/frames/frames_pil_manifest.json")
VARIABLE = "PWV"
STEP_HOURS = 24  # каждые N часов (при шаге 3 ч)
CMAP_NAME = "jet"  # цветовая палитра
VMIN, VMAX = 0, 60  # диапазон визуализации (0, 99-ый перцентиль)
NUM_WORKERS = 4  # количество параллелньых процессов


# Вспомогательные функции
def to_uint8(arr, vmin: float = VMIN, vmax: float = VMAX) -> np.ndarray:
    """Нормирует массив -> 0...255 uint8."""
    scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    return (scaled * 255).astype(np.uint8)


def vis_gray(field: np.ndarray, ts_str: str) -> None:
    """Сохраняет оттенки серого для YOLO."""
    gray_uint8 = to_uint8(field)  # (H, W)
    gray_rgb = np.dstack([gray_uint8] * 3)  # (H, W, 3)
    Image.fromarray(gray_rgb).save(GRAY_DIR / f"{VARIABLE}_{ts_str}.png")


def vis_color(field: np.ndarray, ts_str: str, cmap) -> None:
    """Сохраняет цветную карту."""
    color_arr = cmap(to_uint8(field) / 255.0)[:, :, :3]  # drop alpha
    color_uint8 = (color_arr * 255).astype(np.uint8)
    Image.fromarray(color_uint8).save(COLOR_DIR / f"{VARIABLE}_{ts_str}.png")


def vis_scheme(var, idx: int, ts_str: str) -> None:
    """Сохраняет схему с изолиниями для разметки."""
    slice_ = var.isel(timestamp=idx)
    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=-160))
    slice_.T.plot(
        ax=ax,
        transform=ccrs.PlateCarree(central_longitude=20),
        cmap=CMAP_NAME,
        vmin=VMIN,
        vmax=VMAX,
        add_colorbar=False,
    )
    # Изолинии
    levels = np.arange(VMIN, VMAX, 10)
    contours = plt.contour(
        slice_.lon,
        slice_.lat,
        slice_.T.squeeze(),
        levels=levels,
        colors="#222222",
        linewidths=0.25,
        transform=ccrs.PlateCarree(central_longitude=20),
    )
    plt.clabel(contours, inline=True, fontsize=8, fmt="%d")
    ax.coastlines(linewidth=0.25)
    ax.gridlines(draw_labels=False, linewidth=0.25, linestyle="--", color="black")
    ax.axis("off")
    ax.set_title("")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        VIZ_DIR / f"{VARIABLE}_{ts_str}.png", dpi=100, bbox_inches=None, pad_inches=0
    )
    plt.close(fig)


# Основная обработка одного файла
def process_nc(nc_path: Path, step: int) -> None:
    """Обрабатывает один NetCDF‑файл."""
    print(f"🚀 {os.getpid():>5} → {nc_path.name}")
    ds = xr.open_dataset(nc_path)
    ds = ds.sortby("lon")
    ds["lon"] = (ds["lon"] - 20) % 360
    ds = ds.sortby("lon")
    var = ds[VARIABLE]
    times = ds["timestamp"].values

    cmap = cm.get_cmap(CMAP_NAME)
    for idx in range(0, len(times), step):
        time_val = pd.to_datetime(str(times[idx]))
        ts_str = time_val.strftime("%Y-%m-%d_%H%M")
        base = f"{VARIABLE}_{ts_str}.png"

        # выходные пути
        color_path = COLOR_DIR / base
        gray_path = GRAY_DIR / base
        viz_path = VIZ_DIR / base

        # если все готово, то пропускаем кадр
        if color_path.exists() and gray_path.exists() and viz_path.exists():
            continue

        field = var.isel(timestamp=idx).T.values

        if not gray_path.exists():
            vis_gray(field, ts_str)
        if not color_path.exists():
            vis_color(field, ts_str, cmap)
        if not viz_path.exists():
            vis_scheme(var, idx, ts_str)

    ds.close()  # явное закрытие


def main() -> None:
    # каталоги
    for d in (COLOR_DIR, GRAY_DIR, VIZ_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # входные файлы
    nc_files = sorted(INPUT_DIR.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"В {INPUT_DIR} нет файлов *.nc")

    step = STEP_HOURS // 3  # индексы при шаге 3 ч

    # параллельный запуск
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        list(
            tqdm(
                pool.map(partial(process_nc, step=step), nc_files), total=len(nc_files)
            )
        )

    # формируем манифест
    manifest = [
        {
            "data": {
                "image": f"/data/local-files/?d=data/images/{p.name}",
                "filename": p.name,
            }
        }
        for p in sorted(VIZ_DIR.glob("*.png"))
    ]
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=4)

    print(f"\n✅  Готово: {len(manifest)} кадров (RGB+Gray) и манифест {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
