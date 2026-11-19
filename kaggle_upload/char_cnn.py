#!/usr/bin/env python3
"""
char_cnn.py  --  DAS NEURONALE NETZ VON STUFE 3 (Zeichenklassifikation)

===========================================================================
WAS STEHT IN DIESER DATEI?
===========================================================================
  CharCNN         : der Aufbau (die Architektur) des Netzes
  to_tensor       : Bilder -> Zahlenformat, das PyTorch versteht
  CharClassifier  : ein trainiertes Modell laden und Zeichen erkennen

Warum eine eigene Datei? Training (train_stage3_char_ocr.py) und Einsatz
(analyze_video.py, eval_stage3_plates.py) muessen GENAU dasselbe Netz und
GENAU dieselbe Vorverarbeitung benutzen. Steht beides nur an einer Stelle,
kann es nicht versehentlich auseinanderlaufen.

===========================================================================
DIE ARCHITEKTUR (fuer die Methodik der Arbeit)
===========================================================================
Ein kleines, selbst entworfenes Convolutional Neural Network (CNN), ohne
vortrainiertes Backbone, von Null trainiert.

  Eingabe: 1 x 48 x 48   (1 Kanal = Graustufen, 48 x 48 Pixel)
     |
  Block 1: Conv 3x3,  32 Filter -> BatchNorm -> ReLU -> MaxPool   32 x 24 x 24
  Block 2: Conv 3x3,  64 Filter -> BatchNorm -> ReLU -> MaxPool   64 x 12 x 12
  Block 3: Conv 3x3, 128 Filter -> BatchNorm -> ReLU -> MaxPool  128 x  6 x  6
  Block 4: Conv 3x3, 192 Filter -> BatchNorm -> ReLU             192 x  6 x  6
     |
  Global Average Pooling  (Mittelwert jeder der 192 Karten)       192
  Dropout 30 %
  Linear 192 -> 37        (ein Wert pro Klasse: 0-9, A-Z, "_")      37
     |
  Softmax (erst bei der Anwendung) -> Wahrscheinlichkeit pro Klasse

Die Bausteine kurz erklaert:
  Conv 3x3    : ein Filter wird ueber das Bild geschoben und erkennt ein
                lokales Muster (Kante, Ecke, Bogen). Spaetere Bloecke
                kombinieren diese zu groesseren Mustern (z.B. "Bauch des B").
  BatchNorm   : bringt die Zwischenwerte auf eine einheitliche Skala ->
                stabileres, schnelleres Training.
  ReLU        : setzt negative Werte auf 0 (Nichtlinearitaet). Ohne sie
                waere das ganze Netz nur eine einzige lineare Rechnung.
  MaxPool 2x2 : halbiert Breite und Hoehe (nimmt das Maximum aus 2x2).
                Macht das Netz unempfindlicher gegen kleine Verschiebungen
                und spart Rechenzeit.
  Global Average Pooling : statt die 6x6-Karten flach auszurollen, wird pro
                Karte nur der Mittelwert genommen. Das spart sehr viele
                Parameter und senkt die Gefahr von Overfitting.
  Dropout     : schaltet im Training zufaellig 30 % der Werte ab, damit das
                Netz sich nicht auf einzelne Merkmale verlaesst.

Groesse: ca. 0,32 Mio. Parameter (zum Vergleich: YOLO26n ca. 2,4 Mio.).
Rechnung: Conv-Gewichte 288 + 18.432 + 73.728 + 221.184, BatchNorm
832, Linear 7.141 -> zusammen 321.605.
Dadurch trainiert es auf einer normalen CPU in knapp 50 Minuten (15 Epochen
zu je ca. 3 min, siehe logs\\stage3_training.txt)
und kostet im Einsatz praktisch keine Zeit.

Klasse "_" bedeutet "kein Zeichen" (Rahmen, EU-Feld, Plakette, Schmutz).
Solche Ausschnitte verwirft die Pipeline.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from plate_ocr_utils import CHAR_SIZE, normalize_char_image

NO_CHAR = "_"   # Name der Klasse "kein Zeichen"


def _block(cin, cout, pool):
    """Ein Faltungsblock: Conv 3x3 -> BatchNorm -> ReLU (-> MaxPool).

    cin  : Anzahl Eingangskanaele (Karten aus dem vorherigen Block)
    cout : Anzahl Filter in diesem Block = Anzahl Ausgangskarten
    pool : danach halbieren (True) oder nicht (False)

    padding=1 haelt die Bildgroesse beim 3x3-Filter konstant.
    bias=False, weil die direkt folgende BatchNorm ohnehin einen eigenen
    Verschiebungswert lernt; ein Bias waere doppelt."""
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


class CharCNN(nn.Module):
    """Das Netz (siehe Skizze im Kopf der Datei)."""

    def __init__(self, num_classes: int):
        super().__init__()
        # features: "Merkmale erkennen" (4 Faltungsbloecke)
        # Die Kanalzahl steigt (32 -> 192), waehrend die Bildgroesse sinkt
        # (48 -> 6): weniger Ortsinformation, dafuer mehr und abstraktere Merkmale.
        self.features = nn.Sequential(
            *_block(1, 32, True),     # 48 -> 24
            *_block(32, 64, True),    # 24 -> 12
            *_block(64, 128, True),   # 12 -> 6
            *_block(128, 192, False), # bleibt 6 x 6
        )
        # head: "aus den Merkmalen eine Klasse machen"
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # 192 x 6 x 6 -> 192 x 1 x 1
            nn.Flatten(),                     # -> Vektor mit 192 Werten
            nn.Dropout(0.3),
            nn.Linear(192, num_classes),      # -> ein Rohwert ("Logit") pro Klasse
        )

    def forward(self, x):
        """Wird bei model(x) aufgerufen: Bild(er) rein, ein Wert pro Klasse raus.
        Die Softmax fehlt hier absichtlich: CrossEntropyLoss im Training
        rechnet sie selbst (numerisch stabiler)."""
        return self.head(self.features(x))


def to_tensor(gray_imgs: Sequence[np.ndarray]) -> torch.Tensor:
    """Liste von 48x48-Graustufenbildern (Werte 0..255) -> PyTorch-Tensor
    der Form N x 1 x 48 x 48 mit Werten in [-1, 1].

    Warum [-1, 1] statt 0..255? Neuronale Netze lernen deutlich stabiler,
    wenn die Eingaben klein und um 0 herum verteilt sind.
      0   -> -1 (schwarz)      127,5 -> 0      255 -> +1 (weiss)
    unsqueeze(1) fuegt die Kanal-Dimension ein (1 Kanal, weil Graustufen)."""
    arr = np.stack(gray_imgs).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(1)


class CharClassifier:
    """Laedt ein mit train_stage3_char_ocr.py trainiertes Modell und erkennt
    damit Zeichen. Wird von der Pipeline benutzt, nicht beim Training."""

    def __init__(self, model_path, device: str = "cpu"):
        # Die gespeicherte Datei enthaelt die Gewichte UND die Klassenliste.
        # Die Klassenliste ist noetig, um Ausgabe-Nummer 10 wieder in "A"
        # zu uebersetzen.
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        self.classes: List[str] = ckpt["classes"]
        self.img_size = ckpt.get("img_size", CHAR_SIZE)
        self.model = CharCNN(len(self.classes))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(device)   # Auswertungsmodus (Dropout aus)
        self.device = device

    @torch.no_grad()
    def predict_proba(self, char_imgs: Sequence[np.ndarray], auto_polarity: bool = False) -> np.ndarray:
        """Zeichen-Ausschnitte aus segment_characters() (dunkel auf hell) ->
        Matrix N x 37 mit Wahrscheinlichkeiten (Reihenfolge wie self.classes).

        Die Pipeline braucht ALLE Wahrscheinlichkeiten, nicht nur die beste
        Klasse: die syntaxgestuetzte Dekodierung kann so z.B. "0" statt "O"
        waehlen, wenn an dieser Stelle laut Kennzeichenformat eine Ziffer
        stehen muss.

        auto_polarity=False: Bei der Anwendung ist schon bekannt, dass das
        Zeichen dunkel auf hell ist (siehe normalize_char_image)."""
        if len(char_imgs) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        x = to_tensor([normalize_char_image(c, self.img_size, auto_polarity) for c in char_imgs]).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    @torch.no_grad()
    def predict(self, char_imgs: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """Einfachere Variante: pro Ausschnitt nur das wahrscheinlichste
        Zeichen und seine Wahrscheinlichkeit, z.B. [("L", 0.98), ("I", 0.71)]."""
        if len(char_imgs) == 0:
            return []
        x = to_tensor([normalize_char_image(c, self.img_size) for c in char_imgs]).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)
        conf, idx = probs.max(dim=1)
        return [(self.classes[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]
