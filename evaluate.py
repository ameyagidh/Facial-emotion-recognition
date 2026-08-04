"""
Honest reporting on the untouched FER-2013 test split: overall accuracy,
per-class precision/recall/F1, a training curve plot, and a confusion matrix
that shows FER-2013's well-known fear<->sad confusion rather than hiding it.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import EmotionCNN
from train import FERDataset, DEVICE

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main():
    test_x = np.load(DATA / "test_x.npy")
    test_y = np.load(DATA / "test_y.npy")
    ds = FERDataset(test_x, test_y, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    model = EmotionCNN().to(DEVICE)
    model.load_state_dict(torch.load(ROOT / "best_model.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            all_preds.extend(out.argmax(1).cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"test accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    (ROOT / "test_metrics.json").write_text(json.dumps({"test_accuracy": acc, "report": report}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"FER-2013 test confusion matrix (acc={acc:.3f})")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(ROOT / "confusion_matrix.png", dpi=150)
    print(f"wrote {ROOT / 'confusion_matrix.png'}")

    history = json.loads((ROOT / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(ROOT / "training_curves.png", dpi=150)
    print(f"wrote {ROOT / 'training_curves.png'}")


if __name__ == "__main__":
    main()
