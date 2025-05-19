import os
import glob
import shutil
import random
from pathlib import Path

# Константы
LABELS_DIR = "data/batch_0516_1720/labels"
SOURCE_DIR = "data/batch_0516_1720/frames/frames_pil_gray"
DEST_IMG_DIR = "datasets/dataset_0516_1720_gray/images"
DEST_LABELS_DIR = "datasets/dataset_0516_1720_gray/labels"
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.3,
}
RANDOM_SEED = 42
BLOCK_MIN = 10
BLOCK_MAX = 15

random.seed(RANDOM_SEED)
Path(DEST_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DEST_LABELS_DIR).mkdir(parents=True, exist_ok=True)

for split in SPLIT_RATIOS.keys():
    img_split_dir = os.path.join(DEST_IMG_DIR, split)
    label_split_dir = os.path.join(DEST_LABELS_DIR, split)
    os.makedirs(img_split_dir, exist_ok=True)
    os.makedirs(label_split_dir, exist_ok=True)

all_images = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.png")))
# Оставляем только те изображения, у которых есть разметка
valid_images = []
for img_path in all_images:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABELS_DIR, base + ".txt")
    if os.path.isfile(label_path):
        valid_images.append(img_path)

# Группировка в блоки
blocks = []
i = 0
while i < len(valid_images):
    block_size = random.randint(BLOCK_MIN, BLOCK_MAX)
    block = valid_images[i : i + block_size]
    if len(block) > 0:
        blocks.append(block)
    i += block_size

# Перемешивание и разделение блоков
random.shuffle(blocks)

n_blocks = len(blocks)
n_train_blocks = int(n_blocks * SPLIT_RATIOS["train"])
n_val_blocks = n_blocks - n_train_blocks

train_blocks = blocks[:n_train_blocks]
val_blocks = blocks[n_train_blocks:]

splits = {
    "train": [img for block in train_blocks for img in block],
    "val": [img for block in val_blocks for img in block],
}

# Копирование файлов
for split, files in splits.items():
    for img_path in files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, base + ".txt")
        dst_img = os.path.join(DEST_IMG_DIR, split, os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        dst_label = os.path.join(DEST_LABELS_DIR, split, base + ".txt")
        shutil.copy2(label_path, dst_label)

print(
    f"Готово. Количество изображений с разметкой: {len(valid_images)} "
    f"(train: {len(splits['train'])}, val: {len(splits['val'])}, "
    f"train блоки: {n_train_blocks}, val блоки: {n_val_blocks})"
)
