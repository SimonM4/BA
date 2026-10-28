#!/usr/bin/env python3
"""
char_cnn.py

Das Netz fuer STUFE 3 (Zeichenklassifikation) plus Laden/Anwenden eines
trainierten Modells. Liegt in einer eigenen Datei, damit Training
(train_stage3_char_ocr.py) und Inferenz (analyze_video.py) garantiert
dieselbe Architektur und dieselbe Vorverarbeitung verwenden.

Architektur: kleines, selbst geschriebenes CNN (kein vortrainiertes Backbone):
  4 Conv-Bloecke (Conv-BN-ReLU, 3x davon mit MaxPool) -> Global Average
  Pooling -> Dropout -> Linear. Eingabe: 1 Kanal (Graustufen), 48x48.
  ~0,3 Mio. Parameter -> trainiert auf der CPU in wenigen Minuten und laeuft
  bei der Inferenz praktisch ohne Zeitverlust.

Klasse "_" (falls im Training vorhanden) bedeutet "kein Zeichen" (Rahmen,
EU-Feld, Plakette, Schmutz) -- solche Ausschnitte verwirft die Pipeline.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from plate_ocr_utils import CHAR_SIZE, normalize_char_image

NO_CHAR = "_"


def _block(cin, cout, pool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


class CharCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            *_block(1, 32, True),     # 48 -> 24
            *_block(32, 64, True),    # 24 -> 12
            *_block(64, 128, True),   # 12 -> 6
            *_block(128, 192, False),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(gray_imgs: Sequence[np.ndarray]) -> torch.Tensor:
    """Liste normalisierter uint8-Bilder (CHAR_SIZE x CHAR_SIZE) -> Tensor N x 1 x H x W in [-1, 1]."""
    arr = np.stack(gray_imgs).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(1)


class CharClassifier:
    """Laedt ein mit train_stage3_char_ocr.py trainiertes Modell und
    klassifiziert Zeichen-Ausschnitte."""

    def __init__(self, model_path, device: str = "cpu"):
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        self.classes: List[str] = ckpt["classes"]
        self.img_size = ckpt.get("img_size", CHAR_SIZE)
        self.model = CharCNN(len(self.classes))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict_proba(self, char_imgs: Sequence[np.ndarray], auto_polarity: bool = False) -> np.ndarray:
        """Zeichen-Ausschnitte aus segment_characters() (dunkel auf hell) ->
        N x C Softmax-Wahrscheinlichkeiten (Reihenfolge wie self.classes)."""
        if len(char_imgs) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        x = to_tensor([normalize_char_image(c, self.img_size, auto_polarity) for c in char_imgs]).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    @torch.no_grad()
    def predict(self, char_imgs: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """Zeichen-Ausschnitte (beliebige Groesse, BGR oder Graustufen) ->
        Liste (Zeichen, Wahrscheinlichkeit)."""
        if len(char_imgs) == 0:
            return []
        x = to_tensor([normalize_char_image(c, self.img_size) for c in char_imgs]).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)
        conf, idx = probs.max(dim=1)
        return [(self.classes[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]
