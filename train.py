"""
Train the from-scratch CNN on FER-2013 with class-weighted loss (the
"disgust" class has ~18x fewer samples than "happy"). Held out 10% of the
official train split for validation/early-stopping; the official test split
is reserved untouched for evaluate.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import EmotionCNN, NUM_CLASSES

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class FERDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()  # horizontal flip
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0).float(), int(self.y[i])


def main(epochs=40, batch_size=128, lr=1e-3):
    train_x = np.load(DATA / "train_x.npy")
    train_y = np.load(DATA / "train_y.npy")

    full_ds = FERDataset(train_x, train_y, augment=True)
    n_val = int(0.1 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    # val split shouldn't use train-time augmentation
    val_ds.dataset = FERDataset(train_x, train_y, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    class_counts = np.bincount(train_y, minlength=NUM_CLASSES)
    weights = torch.tensor((class_counts.sum() / (NUM_CLASSES * class_counts)), dtype=torch.float32).to(DEVICE)
    print("class weights:", weights.tolist())

    model = EmotionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        tr_loss /= tr_total
        val_loss /= val_total
        tr_acc = tr_correct / tr_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch+1}/{epochs}  train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ROOT / "best_model.pt")

    elapsed = time.time() - t0
    print(f"training took {elapsed:.1f}s on {DEVICE}, best val_acc={best_val_acc:.4f}")

    (ROOT / "history.json").write_text(json.dumps({**history, "elapsed_sec": elapsed, "device": DEVICE}, indent=2))


if __name__ == "__main__":
    main()
