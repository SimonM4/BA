#!/usr/bin/env python3
"""
train_stage3_char_ocr.py  --  TRAINING STUFE 3: einzelne Zeichen lesen

===========================================================================
WOZU GIBT ES STUFE 3?
===========================================================================
  Stufe 1  ganzes Drohnenbild  ->  Fahrzeug-Boxen
  Stufe 2  Bild/Fahrzeug       ->  Kennzeichen-Box
  Stufe 3  Kennzeichen         ->  Text, z.B. "LIP-YE 351"   (dieses Skript)

Stufe 3 besteht in der Pipeline aus drei Schritten:
  a) Segmentierung: das Kennzeichen wird mit klassischer Bildverarbeitung in
     einzelne Zeichen zerlegt (plate_ocr_utils.segment_characters, kein
     neuronales Netz).
  b) Klassifikation: ein kleines, SELBST GEBAUTES und VON NULL trainiertes
     CNN (char_cnn.py) sagt fuer jeden Zeichen-Ausschnitt, welches Zeichen
     es ist: 0-9, A-Z oder "_" (= kein Zeichen).
  c) Dekodierung: die Zeichen werden gegen die erlaubten deutschen
     Kennzeichenformate geprueft (plate_ocr_utils.decode_german_plate).

DIESES Skript trainiert nur Schritt b), das CNN. Das ist KEINE
Objekterkennung wie in Stufe 1/2, sondern KLASSIFIKATION: Eingabe ist ein
kleines Bild mit genau einem Zeichen, Ausgabe ist eine von 37 Klassen.

Im Unterschied zu Stufe 1 und 2 gibt es hier KEIN vortrainiertes Modell:
alle Gewichte werden aus zufaelligen Startwerten gelernt.

===========================================================================
DATEN
===========================================================================
  archive (4).zip           "CNN letter Dataset/<klasse>/...": echte graue
                            Zeichen-Ausschnitte, ca. 35.500 Bilder, OHNE "O"
  DeutschEinzelneZahlen.zip deutsche FE-Schrift (die Schrift auf deutschen
                            Kennzeichen), weiss auf schwarz, ca. 3.000 Bilder
  --negatives-from          optional: "kein Zeichen"-Beispiele (Klasse "_"),
                            automatisch aus dem Kennzeichen-Datensatz von
                            Stufe 2 ausgeschnitten (EU-Feld, Rahmen,
                            Hintergrund)

Die Klasse eines Bildes ergibt sich aus dem Ordnernamen: ".../A/bild.png"
ist ein "A". Die Klassenliste ist die Vereinigung aller gefundenen Ordner.

===========================================================================
DER KERN DER METHODE: das Domain-Gap-Problem und wie es geloest wird
===========================================================================
Problem: Die Trainingsbilder sind saubere, scharfe, grosse Einzelzeichen.
Ein Zeichen auf einem Kennzeichen aus der Drohne ist dagegen klein (oft nur
10-20 px hoch), unscharf, verrauscht, leicht schraeg und von der
Segmentierung ungenau ausgeschnitten. Ein Netz, das nur saubere Bilder
kennt, versagt dort. Messung mit der ersten Version (runs_stage3_versuch1):
99,8 % auf den Validierungsbildern, aber 0 von 13 echten Kennzeichen
richtig (CER 74 %).

Loesung (in dieser Datei):
  1. Einheitliche Vorverarbeitung fuer Training UND Einsatz
     (normalize_char_image): Graustufen, dunkles Zeichen auf hellem Grund,
     Kontrast gestreckt, 48x48 px. Damit sehen beide Quellen gleich aus und
     das Netz sieht im Einsatz dieselbe Art Bild wie im Training.
  2. "Drohnen-Augmentierung" (Funktion augment): Jedes Trainingsbild wird
     zufaellig so verschlechtert, wie es aus der Luft aussehen koennte
     (duennere/dickere Striche, Verschiebung, Schraegstellung, geringe
     Aufloesung, Unschaerfe, Rauschen, JPEG-Artefakte).
  3. Ueberabtastung: die kleine Quelle DeutschEinzelneZahlen (die richtige
     FE-Schrift!) wird 4x wiederholt, damit sie neben den 35.500
     anderen Bildern nicht untergeht.
  4. Klassengewichte gegen ungleich haeufige Klassen (z.B. "O" kommt nur
     ca. 80-mal vor).
Ergebnis mit dieser Version (runs_stage3): 5 von 13 echten Kennzeichen
exakt richtig, CER 19 %.

BEZUG ZUR FORSCHUNGSFRAGE: Die Augmentierung simuliert genau die Groessen
"Aufloesung" (Zeichenhoehe 10-28 px), "Bildqualitaet" (Unschaerfe, Rauschen,
JPEG) und in Grenzen "Blickwinkel" (Scherung, Drehung). Mit der Funktion
augment() kann man diese Faktoren auch einzeln und gezielt einstellen, um
zu messen, ab wann die Erkennung zusammenbricht.

===========================================================================
AUSGABE (in --out-dir, Standard C:\\BA\\runs_stage3)
===========================================================================
  char_ocr_model.pt            bestes Modell (Gewichte + Klassenliste)
  char_ocr_model.classes.json  Klassenliste als Text
  results.csv                  Loss und Genauigkeit pro Epoche
  training_curves.png          Diagramm dazu
  confusion_matrix.png         welche Zeichen werden womit verwechselt?
  summary.json                 Kennzahlen pro Klasse und pro Quelle

Aufruf (aus C:\\BA\\Code), siehe auch Start\\02_stage3_trainieren.bat:
  python train_stage3_char_ocr.py --data "C:\\BA\\Daten\\archive (4).zip" "C:\\BA\\Daten\\DeutschEinzelneZahlen.zip" --negatives-from C:\\BA\\Daten\\kennzeichen_detection_dataset
"""

import argparse
import csv
import io
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2                 # OpenCV: Bilder laden, skalieren, verzerren, weichzeichnen
import numpy as np         # Rechnen mit Bildern als Zahlen-Arrays
import torch               # PyTorch: neuronales Netz und Training
import torch.nn as nn

from char_cnn import NO_CHAR, CharCNN, to_tensor
from plate_ocr_utils import CHAR_ASPECT, CHAR_SIZE, normalize_char_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Fester Startwert fuer alle Zufallsgeneratoren. Damit liefert ein zweiter
# Lauf mit denselben Daten (nahezu) dieselben Ergebnisse: wichtig fuer die
# Reproduzierbarkeit in der Arbeit.
SEED = 42

# Alle Rohbilder werden beim Laden auf 80 x 60 px (Hoehe x Breite) gebracht.
# Warum nicht gleich 48 x 48? Die Augmentierung arbeitet besser auf etwas
# groesseren Bildern mit dem echten Seitenverhaeltnis eines Zeichens
# (Breite/Hoehe = 0,75). Erst NACH der Augmentierung skaliert
# normalize_char_image auf die Netzgroesse 48 x 48.
RAW_H, RAW_W = 80, 60


# ===========================================================================
# TEIL 1: Daten laden
# ===========================================================================

def _decode(buf: bytes):
    """Bytes einer Bilddatei -> Graustufenbild (numpy-Array).
    So kann man Bilder direkt aus dem Zip lesen, ohne sie zu entpacken."""
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _is_class_name(name: str) -> bool:
    """Ist ein Ordnername eine gueltige Klasse? Nur ein einzelner Buchstabe
    oder eine Ziffer zaehlt ("A", "7"). So werden andere Ordner im Zip
    (z.B. "CNN letter Dataset") automatisch ignoriert."""
    return len(name) == 1 and name.isalnum()


def load_source(path: Path):
    """Laedt alle Zeichenbilder einer Quelle (Zip-Datei ODER Ordner).

    Erwartete Struktur: .../<klasse>/<bild>, z.B. .../A/img001.png
    Rueckgabe: Liste von (Bild 80x60 Graustufen, Klasse als Text).

    Alle Bilder werden einmal komplett in den Arbeitsspeicher geladen.
    Bei knapp 40.000 Bildern a 80x60 Byte sind das nur ca. 200 MB. Das
    Training wird dadurch viel schneller, weil die Festplatte waehrend des
    Trainings nicht mehr gelesen werden muss."""
    items = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                parts = name.replace("\\", "/").split("/")
                if len(parts) < 2 or Path(name).suffix.lower() not in IMG_EXTS:
                    continue
                cls = parts[-2].upper()     # Elternordner = Klasse
                if not _is_class_name(cls):
                    continue
                img = _decode(zf.read(name))
                if img is not None:
                    # INTER_AREA ist das empfohlene Verfahren zum Verkleinern
                    # (mittelt Pixel, statt einzelne zu ueberspringen).
                    items.append((cv2.resize(img, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA), cls))
    else:
        for p in path.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and _is_class_name(p.parent.name.upper()):
                img = _decode(p.read_bytes())
                if img is not None:
                    items.append((cv2.resize(img, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA), p.parent.name.upper()))
    return items


def load_negatives(det_root: Path, n: int, rng: random.Random):
    """Erzeugt n Beispiele fuer die Klasse "_" (kein Zeichen).

    WARUM? Die Segmentierung liefert nicht nur echte Zeichen, sondern auch
    Stoerobjekte: das blaue EU-Feld, den Rahmen, Plaketten, Schmutz. Ohne
    eine Klasse "kein Zeichen" muesste das Netz jedes davon zwingend einem
    Buchstaben zuordnen. Mit "_" kann es sagen "das ist gar kein Zeichen",
    und die Pipeline verwirft den Ausschnitt.

    WIE? Aus den Kennzeichen-Boxen des Stufe-2-Datensatzes (YOLO-Labels)
    werden pro Kennzeichen 5 Ausschnitte in Zeichengroesse geschnitten, die
    absichtlich KEIN vollstaendiges Zeichen enthalten:
      - am linken Rand (EU-Feld)       - am rechten Rand
      - am oberen Rahmen               - am unteren Rahmen
      - irgendwo im Bildhintergrund"""
    img_dir = det_root / "images" / "train"     # nur train -> keine Vermischung mit Test
    lbl_dir = det_root / "labels" / "train"
    files = sorted(img_dir.glob("*.jpg"))
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        lf = lbl_dir / (f.stem + ".txt")
        if not lf.exists():
            continue
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        for line in lf.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            # YOLO-Box (relativ, Mittelpunkt) -> Pixel (Ecke oben links + Groesse)
            cx, cy, bw, bh = (float(t) for t in v[1:5])
            x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            pw, ph = int(bw * W), int(bh * H)
            if ph < 12 or pw < 30:
                continue  # zu kleine Kennzeichen liefern unbrauchbare Ausschnitte
            # Ausschnittgroesse wie ein Zeichen: 70 % der Kennzeichenhoehe,
            # Breite ueber das Zeichen-Seitenverhaeltnis 0,75.
            ch = int(0.7 * ph)
            cw = int(ch * CHAR_ASPECT)
            cands = [
                (x0 - cw // 3, y0 + (ph - ch) // 2),                      # EU-Feld / linker Rand
                (x0 + pw - cw + cw // 3, y0 + (ph - ch) // 2),            # rechter Rand
                (x0 + rng.randint(0, max(1, pw - cw)), y0 - ch // 2),      # oberer Rahmen
                (x0 + rng.randint(0, max(1, pw - cw)), y0 + ph - ch // 2),  # unterer Rahmen
                (rng.randint(0, max(1, W - cw)), rng.randint(0, max(1, H - ch))),  # Hintergrund
            ]
            for (x, y) in cands:
                x, y = max(0, x), max(0, y)     # nicht ueber den Bildrand hinaus
                crop = img[y:y + ch, x:x + cw]
                if crop.shape[0] >= 8 and crop.shape[1] >= 6:
                    out.append((cv2.resize(crop, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA), NO_CHAR))
    return out[:n]


# ===========================================================================
# TEIL 2: Drohnen-Augmentierung ("so sieht ein Zeichen aus der Luft aus")
# ===========================================================================
# Diese Funktion ist der wichtigste Baustein gegen den Domain Gap (siehe
# Kopf der Datei). Sie bekommt ein sauberes Zeichenbild und gibt eine
# zufaellig verschlechterte Version zurueck. Jeder Schritt passiert nur mit
# einer gewissen Wahrscheinlichkeit, so dass das Netz alle Mischungen aus
# "leicht" bis "stark verschlechtert" zu sehen bekommt.
#
# Da die Verschlechterung bei jedem Aufruf neu ausgelost wird, sieht das
# Netz jedes Trainingsbild in jeder Epoche in einer anderen Variante.
# ===========================================================================

def augment(g: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = g.shape

    # --- Schritt 0: Strichstaerke veraendern ------------------------------
    # Aus der Luft wirkt Schrift durch Unschaerfe und geringe Aufloesung
    # mal duenner, mal fetter als in den Trainingsbildern.
    #   dilate = helle Bereiche wachsen, erode = dunkle Bereiche wachsen.
    # Bei dunkler Schrift auf hellem Grund macht dilate die Striche also
    # duenner, erode dicker (bei der hell-auf-dunkel-Quelle
    # DeutschEinzelneZahlen umgekehrt). Entscheidend ist: das Netz sieht
    # beide Richtungen.
    #   35 % dilate, 15 % erode, 50 % unveraendert
    r = rng.random()
    if r < 0.35:
        g = cv2.dilate(g, np.ones((rng.choice([2, 3]), rng.choice([2, 3])), np.uint8))
    elif r < 0.5:
        g = cv2.erode(g, np.ones((2, 2), np.uint8))

    # --- Schritt 1: Geometrie (immer) -------------------------------------
    # Simuliert: ungenaue Segmentierung (Zeichen nicht mittig), schraege
    # Perspektive aus der Drohne, schmale FE-Schrift.
    #   sy  : Hoehe skalieren, 85-115 %
    #   sx  : Breite zusaetzlich 70-115 % davon -> auch schmale Zeichen
    #   ang : Drehung -8 bis +8 Grad
    #   sh  : Scherung (Schraegstellung wie Kursivschrift), bis +-0,2
    #   tx/ty: Verschiebung bis 10 % der Breite / 7 % der Hoehe
    sy = rng.uniform(0.85, 1.15)
    sx = sy * rng.uniform(0.7, 1.15)
    ang = np.radians(rng.uniform(-8, 8))
    sh = rng.uniform(-0.2, 0.2)
    # Aus Skalierung, Drehung und Scherung wird EINE 2x3-Matrix M gebaut
    # (affine Transformation). Fuer jeden Pixel gilt:
    #   x_neu = a*x + b*y + Verschiebung_x
    #   y_neu = c*x + d*y + Verschiebung_y
    a, b = sx * np.cos(ang), -sy * np.sin(ang) + sh
    c, d = sx * np.sin(ang), sy * np.cos(ang)
    tx = rng.uniform(-0.1, 0.1) * w
    ty = rng.uniform(-0.07, 0.07) * h
    # Der Ausdruck w/2 - a*w/2 - b*h/2 sorgt dafuer, dass um die BILDMITTE
    # gedreht/skaliert wird (statt um die Ecke oben links). Danach kommt die
    # zufaellige Verschiebung tx/ty dazu.
    M = np.float32([[a, b, w / 2 - a * w / 2 - b * h / 2 + tx],
                    [c, d, h / 2 - c * w / 2 - d * h / 2 + ty]])
    # BORDER_REPLICATE: Pixel, die durch die Verschiebung neu ins Bild
    # kommen, bekommen die Farbe des naechsten Randpixels (statt Schwarz).
    # Sonst entstuenden kuenstliche schwarze Raender, die es in echt nicht gibt.
    g = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- Schritt 2: geringe Aufloesung (70 %) ------------------------------
    # DER zentrale Schritt fuer die Drohnenperspektive: Das Zeichen wird auf
    # 10-28 px Hoehe verkleinert und wieder auf 80 px vergroessert. Dabei
    # gehen Details verloren, genau wie bei einem kleinen Zeichen im
    # Drohnenbild, das spaeter fuer das Netz hochskaliert wird.
    # (Diese Spanne 10-28 px ist ein direkter Stellhebel fuer deine
    # Forschungsfrage nach der Mindestaufloesung.)
    if rng.random() < 0.7:
        th = rng.randint(10, 28)
        g = cv2.resize(g, (max(4, int(th * w / h)), th), interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))

    # --- Schritt 3: Unschaerfe (40 %) -------------------------------------
    # Gauss-Weichzeichner simuliert Bewegungs- und Fokusunschaerfe sowie
    # Luftflimmern. Kernel 3x3 oder 5x5, Staerke sigma 0,5-1,5.
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        g = cv2.GaussianBlur(g, (k, k), rng.uniform(0.5, 1.5))

    # --- Schritt 4: Kontrast, Helligkeit (immer) + Rauschen (50 %) ---------
    # Kontrast: Abstand jedes Pixels von Mittelgrau (128) mit 0,5-1,3
    #           multiplizieren (< 1 = flauer, > 1 = haerter).
    # Helligkeit: -40 bis +40 addieren (Gegenlicht, Schatten, Sonne).
    # Rauschen: zufaellige Abweichung pro Pixel (Kamerasensor, wenig Licht).
    g = g.astype(np.float32)
    g = (g - 128) * rng.uniform(0.5, 1.3) + 128 + rng.uniform(-40, 40)
    if rng.random() < 0.5:
        g += np.random.normal(0, rng.uniform(2, 12), g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)   # zurueck in den gueltigen Bereich 0..255

    # --- Schritt 5: JPEG-Artefakte (30 %) ---------------------------------
    # Videostreams (RTSP) und Aufnahmen sind stark komprimiert. Das Bild
    # wird mit niedriger Qualitaet (25-70) als JPEG kodiert und wieder
    # dekodiert. Dabei entstehen die typischen Block- und Ringartefakte.
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 70)])
        if ok:
            g = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return g


# ===========================================================================
# TEIL 3: Training und Auswertung
# ===========================================================================

def make_batches(n, bs, shuffle, rng):
    """Teilt die Indizes 0..n-1 in Pakete ("Batches") der Groesse bs auf.
    Beim Training wird vorher gemischt (shuffle=True), damit das Netz die
    Bilder nicht immer in derselben Reihenfolge sieht."""
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def evaluate(model, X_raw, y, classes, rng, degrade: bool, bs=512):
    """Misst die Genauigkeit (Anteil richtig erkannter Zeichen).

    degrade=False: auf den sauberen Validierungsbildern
    degrade=True : auf Validierungsbildern MIT Drohnen-Augmentierung.
                   Das ist die aussagekraeftigere Zahl, weil sie naeher an
                   echten Drohnenbildern liegt.

    Gibt (Genauigkeit, Vorhersagen) zurueck."""
    model.eval()            # Auswertungsmodus: Dropout aus, BatchNorm fest
    preds = []
    with torch.no_grad():   # keine Gradienten noetig -> schneller, weniger Speicher
        for b in make_batches(len(X_raw), bs, False, rng):
            imgs = [normalize_char_image(augment(X_raw[i], rng) if degrade else X_raw[i]) for i in b]
            # argmax(1): die Klasse mit dem hoechsten Wert ist die Vorhersage
            preds.append(model(to_tensor(imgs)).argmax(1).numpy())
    preds = np.concatenate(preds) if preds else np.array([], int)
    return float((preds == y).mean()) if len(y) else 0.0, preds


def save_plots(out_dir: Path, history, cm, classes):
    """Speichert zwei Abbildungen fuer die Arbeit:
      training_curves.png  : Loss und Genauigkeit pro Epoche
      confusion_matrix.png : welche wahre Klasse (Zeile) wird als welche
                             Klasse (Spalte) erkannt? Zeilennormiert, d.h. jede
                             Zeile summiert sich zu 1. Helle Felder ausserhalb
                             der Diagonale = typische Verwechslungen (z.B. 8/B)."""
    try:
        import matplotlib
        matplotlib.use("Agg")     # ohne Bildschirmfenster, nur Datei speichern
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stage3] matplotlib nicht installiert -> keine Diagramme.")
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="Train-Loss")
    ax[0].set_xlabel("Epoche"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, [h["val_acc_clean"] for h in history], label="Val (Original)")
    ax[1].plot(ep, [h["val_acc_degraded"] for h in history], label="Val (Drohnen-Augmentierung)")
    ax[1].set_xlabel("Epoche"); ax[1].set_title("Genauigkeit"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=120); plt.close(fig)

    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)   # zeilennormieren
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Vorhersage"); ax.set_ylabel("Wahr")
    ax.set_title("Konfusionsmatrix (Val, Drohnen-Augmentierung, zeilennormiert)")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=120); plt.close(fig)


# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================
def main() -> None:
    # -----------------------------------------------------------------------
    # SCHRITT 1: Einstellungen
    # -----------------------------------------------------------------------
    #   --epochs 15       : 15 Durchlaeufe reichen hier. Im Lauf vom 14.10.
    #                       (logs\stage3_training.txt) stieg val_acc_drohne
    #                       ab Epoche 10 nur noch von 98,8 % auf 99,0 %.
    #   --batch-size 128  : 128 Zeichen pro Gewichts-Update
    #   --lr 2e-3         : maximale Lernrate (Schrittweite beim Lernen)
    #   --val-split 0.15  : 15 % jeder Klasse werden zum Pruefen zurueckgehalten
    #   --oversample-small 4 : kleine Quellen 4x wiederholen (siehe Kopf, Punkt 3)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--data-dirs", nargs="+", required=True,
                    help="Zip-Dateien oder Ordner mit <klasse>/<bild>-Struktur")
    ap.add_argument("--negatives-from", default=None,
                    help="Kennzeichen-Detection-Datensatz (YOLO) fuer 'kein Zeichen'-Beispiele")
    ap.add_argument("--n-negatives", type=int, default=3000)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "runs_stage3"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=0, help="CPU-Threads fuer torch (0 = automatisch)")
    ap.add_argument("--oversample-small", type=int, default=4,
                    help="kleine Quellen (<10.000 Bilder, z.B. DeutschEinzelneZahlen) so oft wiederholen")
    args = ap.parse_args()

    # Alle drei Zufallsgeneratoren (Python, numpy, torch) festlegen ->
    # reproduzierbare Datenaufteilung, Augmentierung und Startgewichte.
    rng = random.Random(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # SCHRITT 2: Daten laden
    # -----------------------------------------------------------------------
    # Parallel zu den Bildern werden zwei Listen gefuehrt:
    #   source_of[i] : aus welcher Quelle stammt Bild i? (fuer die Auswertung
    #                  pro Quelle am Ende)
    #   repeat_of[i] : wie oft soll Bild i im Training wiederholt werden?
    t0 = time.time()
    items, source_of, repeat_of = [], [], []
    for src in args.data:
        p = Path(src)
        its = load_source(p)
        print(f"[stage3] {p.name}: {len(its)} Bilder, Klassen: {''.join(sorted(set(c for _, c in its)))}")
        rep = args.oversample_small if len(its) < 10000 else 1
        items += its
        source_of += [p.name] * len(its)
        repeat_of += [rep] * len(its)
        if rep > 1:
            print(f"[stage3]   -> {rep}x ueberabgetastet (kleine Quelle, aber FE-Schrift wie auf echten Kennzeichen)")
    if args.negatives_from:
        neg = load_negatives(Path(args.negatives_from), args.n_negatives, rng)
        print(f"[stage3] 'kein Zeichen'-Beispiele aus {args.negatives_from}: {len(neg)}")
        items += neg
        source_of += ["negatives"] * len(neg)
        repeat_of += [1] * len(neg)
    print(f"[stage3] Laden: {time.time() - t0:.1f}s")

    # Klassen alphabetisch sortieren und durchnummerieren:
    # "0"->0, "1"->1, ..., "A"->10, ..., "_"->36. Das Netz arbeitet mit
    # diesen Nummern, nicht mit den Buchstaben.
    classes = sorted(set(c for _, c in items))
    cidx = {c: i for i, c in enumerate(classes)}
    X = [im for im, _ in items]
    y = np.array([cidx[c] for _, c in items])
    counts = Counter(y.tolist())
    print(f"[stage3] {len(X)} Bilder, {len(classes)} Klassen: {' '.join(classes)}")

    # -----------------------------------------------------------------------
    # SCHRITT 3: Aufteilen in Training und Validierung (stratifiziert)
    # -----------------------------------------------------------------------
    # "Stratifiziert" heisst: von JEDER Klasse kommen 15 % in die Validierung.
    # Bei einer rein zufaelligen Aufteilung koennte eine seltene Klasse wie
    # "O" (ca. 80 Bilder) zufaellig fast gar nicht in der Validierung landen.
    order = list(range(len(X)))
    rng.shuffle(order)
    val_idx, train_idx, seen = [], [], Counter()
    for i in order:
        if seen[y[i]] < int(round(counts[y[i]] * args.val_split)):
            val_idx.append(i); seen[y[i]] += 1
        else:
            train_idx.append(i)
    # Ueberabtastung erst JETZT und NUR im Trainingsteil. Wuerde man vorher
    # wiederholen, laegen Kopien desselben Bildes in Training UND Validierung,
    # und die Validierung waere geschoent ("Data Leakage").
    train_idx = [i for i in train_idx for _ in range(repeat_of[i])]
    Xtr, ytr = [X[i] for i in train_idx], y[train_idx]
    Xva, yva = [X[i] for i in val_idx], y[val_idx]
    print(f"[stage3] Train {len(Xtr)} / Val {len(Xva)}")

    # -----------------------------------------------------------------------
    # SCHRITT 4: Fehlerfunktion, Modell, Optimierer
    # -----------------------------------------------------------------------
    # Klassengewichte: Seltene Klassen zaehlen beim Fehler staerker.
    #   Gewicht = Wurzel(mittlere Haeufigkeit / Haeufigkeit der Klasse)
    # Die Wurzel daempft den Effekt: eine 100x seltenere Klasse zaehlt 10x
    # so stark, nicht 100x (das wuerde das Training instabil machen).
    # Danach so skaliert, dass das mittlere Gewicht 1 ist.
    freq = np.array([max(1, Counter(ytr.tolist())[k]) for k in range(len(classes))], np.float32)
    w = (freq.mean() / freq) ** 0.5
    # CrossEntropyLoss: Standard-Fehlerfunktion fuer Klassifikation. Sie ist
    # klein, wenn das Netz der richtigen Klasse eine hohe Wahrscheinlichkeit
    # gibt. label_smoothing=0.05: das Ziel ist "95 % richtige Klasse" statt
    # "100 %". Das verhindert, dass das Netz uebertrieben selbstsicher wird,
    # was bei unsicheren Drohnenbildern zu falschen, aber sicheren Antworten
    # fuehren wuerde.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean()), label_smoothing=0.05)

    # Das Netz selbst (Aufbau: siehe char_cnn.py), mit zufaelligen Startgewichten.
    model = CharCNN(len(classes))
    print(f"[stage3] Modell: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter, "
          f"Torch-Threads: {torch.get_num_threads()}")
    # AdamW: verbreiteter Optimierer (passt die Schrittweite pro Gewicht an).
    # weight_decay=1e-4 zieht Gewichte leicht Richtung 0 -> weniger Overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # OneCycle-Lernratenplan: In den ersten 15 % der Schritte steigt die
    # Lernrate langsam bis 2e-3 an ("Aufwaermen"), danach sinkt sie bis zum
    # Ende auf fast 0. Grosse Schritte am Anfang = schnell lernen, kleine
    # Schritte am Ende = fein justieren.
    steps = args.epochs * ((len(Xtr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    # -----------------------------------------------------------------------
    # SCHRITT 5: Trainingsschleife
    # -----------------------------------------------------------------------
    best_acc, history = -1.0, []
    model_path = out_dir / "char_ocr_model.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()           # Trainingsmodus: Dropout an, BatchNorm lernt mit
        tot_loss, n_seen = 0.0, 0
        for b in make_batches(len(Xtr), args.batch_size, True, rng):
            # 85 % der Bilder werden augmentiert, 15 % bleiben sauber. So
            # verlernt das Netz die sauberen Zeichen nicht.
            imgs = [normalize_char_image(augment(Xtr[i], rng) if rng.random() < 0.85 else Xtr[i]) for i in b]
            xb, yb = to_tensor(imgs), torch.from_numpy(ytr[b])
            # Die vier Schritte eines Lernschritts:
            optimizer.zero_grad()            # 1. alte Gradienten loeschen
            loss = criterion(model(xb), yb)  # 2. Vorhersage + Fehler berechnen
            loss.backward()                  # 3. Gradienten berechnen (Backpropagation):
                                             #    in welche Richtung muss jedes Gewicht?
            optimizer.step()                 # 4. Gewichte einen Schritt anpassen
            sched.step()                     # Lernrate fuer den naechsten Schritt
            tot_loss += loss.item() * len(b); n_seen += len(b)

        # Nach jeder Epoche zweimal messen: sauber und "wie aus der Drohne".
        # Fuer die "Drohnen"-Messung wird immer derselbe Startwert (123)
        # benutzt, damit alle Epochen moeglichst gleich verschlechterte
        # Bilder sehen und vergleichbar sind.
        # (Kleine Einschraenkung: das Rauschen in augment() kommt aus
        # np.random und nicht aus diesem rng; es ist daher nicht in jeder
        # Epoche exakt gleich. Fuer den Vergleich ist das vernachlaessigbar.)
        acc_clean, _ = evaluate(model, Xva, yva, classes, rng, degrade=False)
        acc_deg, _ = evaluate(model, Xva, yva, classes, random.Random(123), degrade=True)
        h = dict(epoch=epoch, train_loss=tot_loss / n_seen, val_acc_clean=acc_clean,
                 val_acc_degraded=acc_deg, seconds=time.time() - t0)
        history.append(h)
        print(f"[stage3] Epoche {epoch:2d}/{args.epochs} loss={h['train_loss']:.4f} "
              f"val_acc={acc_clean:.4f} val_acc_drohne={acc_deg:.4f} ({h['seconds']:.0f}s)", flush=True)

        # Bestes Modell speichern. Massstab ist die Genauigkeit auf den
        # VERSCHLECHTERTEN Bildern, weil es auf den Einsatz aus der Luft ankommt.
        # Gespeichert werden Gewichte UND Klassenliste, damit die Pipeline
        # weiss, welche Ausgabenummer welcher Buchstabe ist.
        if acc_deg > best_acc:
            best_acc = acc_deg
            torch.save({"model_state": model.state_dict(), "classes": classes, "img_size": CHAR_SIZE,
                        "epoch": epoch, "val_acc_clean": acc_clean, "val_acc_degraded": acc_deg}, model_path)
        # Verlauf nach JEDER Epoche neu schreiben -> bei einem Abbruch ist
        # der Stand bis dahin trotzdem vorhanden.
        with open(out_dir / "results.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            wr.writeheader(); wr.writerows(history)

    # -----------------------------------------------------------------------
    # SCHRITT 6: Abschlussauswertung mit dem besten Modell
    # -----------------------------------------------------------------------
    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    acc_deg, preds = evaluate(model, Xva, yva, classes, random.Random(123), degrade=True)
    # Konfusionsmatrix: cm[wahr, vorhergesagt] zaehlt, wie oft Klasse "wahr"
    # als Klasse "vorhergesagt" erkannt wurde. Die Diagonale = richtig.
    cm = np.zeros((len(classes), len(classes)), np.int64)
    for t, p in zip(yva, preds):
        cm[t, p] += 1
    # Genauigkeit pro Klasse (welche Zeichen sind schwierig?) und pro Quelle
    # (funktioniert die FE-Schrift so gut wie die anderen Bilder?).
    per_class = {c: float(cm[i, i] / max(1, cm[i].sum())) for i, c in enumerate(classes)}
    src_va = [source_of[i] for i in val_idx]
    per_source = {s: float(np.mean([p == t for p, t, ss in zip(preds, yva, src_va) if ss == s]))
                  for s in sorted(set(src_va))}
    summary = {"best_epoch": ckpt["epoch"], "val_acc_clean": ckpt["val_acc_clean"],
               "val_acc_degraded": ckpt["val_acc_degraded"], "per_class_acc_degraded": per_class,
               "per_source_acc_degraded": per_source, "classes": classes,
               "n_train": len(Xtr), "n_val": len(Xva)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "char_ocr_model.classes.json").write_text(json.dumps(classes), encoding="utf-8")
    save_plots(out_dir, history, cm, classes)
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:6]
    print(f"\n[stage3] Fertig. Bestes Modell: Epoche {ckpt['epoch']}, val_acc={ckpt['val_acc_clean']:.4f}, "
          f"val_acc_drohne={ckpt['val_acc_degraded']:.4f}")
    print(f"[stage3] Schwaechste Klassen: {', '.join(f'{c}={a:.2f}' for c, a in worst)}")
    print(f"[stage3] Gespeichert unter: {model_path}")
    # HINWEIS FUER DIE ARBEIT: Die Validierungsbilder wurden auch zur Auswahl
    # der besten Epoche benutzt, und es sind Einzelzeichen, keine echten
    # Kennzeichen. Die belastbare Messung ist deshalb eval_stage3_plates.py
    # auf echten Kennzeichen-Ausschnitten (Start\04_stage3_auf_kennzeichen_testen.bat).


if __name__ == "__main__":
    main()
