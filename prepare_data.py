"""
Unpack the raw FER-2013 pickle files (data/raw/train.pt, test.pt — each a
list of {"img_bytes": <jpeg bytes>, "labels": <str>}) into normalized numpy
arrays we can feed straight into training. FER-2013 is the deliberately noisy,
real-world 35,887-image dataset — no cleaning/curation applied here.
"""
import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "processed"

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_split(name):
    with open(RAW / name, "rb") as f:
        records = pickle.load(f)
    images = np.zeros((len(records), 48, 48), dtype=np.uint8)
    labels = np.zeros(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        img = Image.open(io.BytesIO(r["img_bytes"])).convert("L")
        images[i] = np.array(img)
        labels[i] = CLASS_TO_IDX[r["labels"]]
    return images, labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split("train.pt")
    test_x, test_y = load_split("test.pt")

    np.save(OUT / "train_x.npy", train_x)
    np.save(OUT / "train_y.npy", train_y)
    np.save(OUT / "test_x.npy", test_x)
    np.save(OUT / "test_y.npy", test_y)

    print(f"train: {train_x.shape}, test: {test_x.shape}")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: train={int((train_y==i).sum())} test={int((test_y==i).sum())}")


if __name__ == "__main__":
    main()
