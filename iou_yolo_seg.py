#!/usr/bin/env python3
"""
iou_yolo_seg.py

Вычисление Intersection over Union (IoU) для модели YOLO.

Пример использования:
python iou_yolo_seg.py \
    --images_dir ./datasets/ds_val/images \
    --labels_dir ./datasets/ds_val/labels \
    --model_path ./runs/segment/0516_gray_11s_720_b16_ep200/weights/best.pt \
    --pred_masks_dir ./test_iou/pred_masks \
    --gt_masks_dir ./test_iou/gt_masks \
    --results_path ./test_iou/iou_results.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def save_mask(mask: np.ndarray, path: str) -> None:
    """Сохраняет бинарную маску в path (белый = 255, черный = 0)."""
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def polygon_to_mask(points, height, width):
    """
    Растеризует полигон (список пар (x, y), координаты пикселей)
    в бинарную маску размера (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.array(points, dtype=np.int32).reshape(-1, 2)
    if contour.shape[0] >= 3:  # нужно минимум три точки для полигона
        cv2.fillPoly(mask, [contour], 1)
    return mask


def gt_mask_from_label(label_path: str, image_shape):
    """
    Преобразует файл разметки YOLO в бинарную маску.

    Ожидаемый формат разметки:
    class_id x1 y1 x2 y2 ... xn yn  (все координаты нормированы в [0,1])
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            items = [float(x) for x in line.strip().split()]
            if len(items) < 7:
                # класс + минимум 3 пары (x,y) == 7 чисел
                continue
            coords = items[1:]
            pts = [
                (int(coords[i] * width), int(coords[i + 1] * height))
                for i in range(0, len(coords), 2)
            ]
            mask |= polygon_to_mask(pts, height, width)
    return mask


def infer_and_save_mask(model: "YOLO", img_path: str, save_path: str):
    """
    Запускает модель на одном изображении, объединяет все маски для одного класса,
    сохраняет результат и возвращает его как массив NumPy (бинарная маска).
    """
    res = model.predict(img_path, stream=False, save=False, imgsz=720, conf=0.25, retina_masks=True)[0]

    if res.masks is None or res.masks.data is None:
        # Нет детекций - пустая маска
        img = cv2.imread(img_path)
        return np.zeros(img.shape[:2], dtype=np.uint8)

    masks = res.masks.data.cpu().numpy()  # (n, H, W)
    combined = np.max(masks, axis=0).astype(np.uint8)
    save_mask(combined, save_path)
    return combined


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Попиксельный IoU для пары бинарных масок."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        # Крайние случаи: обе пустые - 1; только одна пустая - 0.
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def main():
    parser = argparse.ArgumentParser(
        description="Вычисление IoU между сегментацией модели и разметкой"
    )
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--labels_dir", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--pred_masks_dir", required=True, type=str)
    parser.add_argument("--gt_masks_dir", required=True, type=str)
    parser.add_argument("--results_path", required=True, type=str)
    args = parser.parse_args()

    os.makedirs(args.pred_masks_dir, exist_ok=True)
    os.makedirs(args.gt_masks_dir, exist_ok=True)

    model = YOLO(args.model_path)

    image_files = sorted(
        f
        for f in os.listdir(args.images_dir)
        if f.lower().endswith((".png"))
    )
    if not image_files:
        print(f"Нет изображений в {args.images_dir}")
        sys.exit(1)

    scores = {}
    for img_name in tqdm(image_files, desc="Оценка"):
        img_path = os.path.join(args.images_dir, img_name)
        pred_mask_path = os.path.join(
            args.pred_masks_dir, os.path.splitext(img_name)[0] + "_pred.png"
        )
        gt_mask_path = os.path.join(
            args.gt_masks_dir, os.path.splitext(img_name)[0] + "_gt.png"
        )
        # Предсказание
        pred_mask = infer_and_save_mask(model, img_path, pred_mask_path)
        # Разметка
        label_file = os.path.join(
            args.labels_dir, os.path.splitext(img_name)[0] + ".txt"
        )
        if not os.path.exists(label_file):
            print(f"Нет разметки для {img_name} - пропуск")
            continue
        img = cv2.imread(img_path)
        gt_mask = gt_mask_from_label(label_file, img.shape)
        save_mask(gt_mask, gt_mask_path)

        scores[img_name] = iou(pred_mask, gt_mask)

    mean_iou = sum(scores.values()) / len(scores) if scores else 0.0
    scores["mean_iou"] = mean_iou

    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    print(f"Средний IoU по {len(image_files)} изображениям: {mean_iou:.4f}")
    print(f"Результаты по каждому изображению сохранены в {args.results_path}")


if __name__ == "__main__":
    main()
