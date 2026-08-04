"""
A small CNN trained from scratch on FER-2013 — no pretrained backbone. This
keeps the "genuinely trained, not fine-tuned" story honest for a project
whose whole point is showing real (modest) numbers on a noisy dataset.
"""
import torch
import torch.nn as nn

NUM_CLASSES = 7


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),    # 48 -> 24
            block(32, 64),   # 24 -> 12
            block(64, 128),  # 12 -> 6
            block(128, 256), # 6 -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    m = EmotionCNN()
    out = m(torch.randn(2, 1, 48, 48))
    print(out.shape)
