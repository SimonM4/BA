#!/usr/bin/env python3
"""
train_stage3_char_ocr.py

STUFE 3 der Pipeline: Die einzelnen Zeichen auf dem (durch Stufe 2
gefundenen und ausgeschnittenen) Kennzeichen lesen (OCR).

Anders als Stufe 1 und 2 ist das HIER KEINE Objekterkennung (keine
Bounding-Boxes), sondern eine KLASSIFIKATION: Wir gehen davon aus, dass die
einzelnen Zeichen auf dem Kennzeichen bereits einzeln ausgeschnitten wurden
(z.B. durch Konturerkennung/Segmentierung auf dem Kennzeichen-Crop aus
Stufe 2) und wollen fuer jeden Zeichen-Ausschnitt nur noch sagen: "welcher
von 0-9/A-Z ist das". Deshalb ein eigenes, viel einfacheres CNN statt YOLO.

Datenquellen: zwei Ordner mit JEWEILS einer Unterordner-pro-Klasse-Struktur
(z.B. ".../0/bild1.jpg", ".../A/bild2.jpg", ...):
  - archive (4).zip entpackt  (enthaelt intern "CNN letter Dataset/<klasse>/...")
  - DeutschEinzelneZahlen.zip entpackt (Klassen-Ordner direkt auf oberster Ebene)

Das Script findet den "Klassen-Ordner" in beiden Quellen automatisch (siehe
find_class_root) und fuehrt beide zu EINEM gemeinsamen Klassifikator
zusammen -- die Klassenliste ist die UNION der in beiden Quellen gefundenen
Ordnernamen (kein hartcodiertes Alphabet), damit z.B. der Unterschied "mit
oder ohne 'O'" zwischen den beiden Quellen (siehe Chat-Verlauf) nicht zu
einem Fehler fuehrt, sondern das Modell einfach beide Zeichensaetze lernt.

Architektur: bewusst ein kleines, selbst geschriebenes CNN statt eines
grossen vortrainierten Backbones (z.B. ResNet) -- Zeichenerkennung auf
kleinen, einfachen Graustufen-aehnlichen Ausschnitten ist eine vergleichsweise
einfache Aufgabe, ein ueberdimensioniertes Modell wuerde nur Trainingszeit
kosten, ohne einen echten Vorteil zu bringen, und ist in der Arbeit auch
leichter zu begruenden/erklaeren als "wir nutzen ImageNet-Gewichte fuer
Ziffern".

Voraussetzung: `pip install torch torchvision`.

Aufruf:
  python train_stage3_char_ocr.py --data-dirs "C:\\...\\DatenSatz_Zeichen1" "C:\\...\\DeutschEinzelneZahlen"
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


def find_class_root(root: Path) -> Path:
    """Sucht innerhalb von `root` (bis zu 2 Ebenen tief) den Ordner, dessen
    direkte Unterordner wie Klassen-Label aussehen (Namen mit hoechstens
    2 Zeichen, z.B. "0".."9", "A".."Z"). Deckt sowohl Quellen ab, bei denen
    die Klassenordner direkt auf oberster Ebene liegen, als auch solche mit
    einer zusaetzlichen Zwischenebene (z.B. "CNN letter Dataset/<klasse>/")."""
    def looks_like_classes(d: Path) -> bool:
        subdirs = [c for c in d.iterdir() if c.is_dir()]
        if not subdirs:
            return False
        short_named = [c for c in subdirs if len(c.name) <= 2]
        return len(short_named) >= max(1, len(subdirs) - 1)

    if looks_like_classes(root):
        return root
    for child in root.iterdir():
        if child.is_dir() and looks_like_classes(child):
            return child
    raise FileNotFoundError(
        f"Konnte in {root} keinen Ordner mit Klassen-Unterordnern (0-9/A-Z) finden."
    )


def collect_samples(data_dirs, seed=SEED):
    """Sammelt (Bildpfad, Klassenname) aus allen Quellordnern und baut die
    gemeinsame, alphabetisch sortierte Klassenliste (Union aller Quellen)."""
    samples = []
    class_names = set()

    for data_dir in data_dirs:
        class_root = find_class_root(Path(data_dir))
        print(f"[stage3] Quelle {data_dir} -> Klassenordner: {class_root}")
        for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
            cname = class_dir.name
            class_names.add(cname)
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    samples.append((img_path, cname))

    class_list = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    random.Random(seed).shuffle(samples)
    print(f"[stage3] {len(samples)} Bilder insgesamt, {len(class_list)} Klassen: {class_list}")
    return samples, class_to_idx


class CharDataset(Dataset):
    """Einfaches Dataset: laedt ein Zeichen-Bild und gibt (Tensor, Klassen-Index) zurueck."""

    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cname = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, self.class_to_idx[cname]


class SmallCharCNN(nn.Module):
    """Kleines CNN fuer Zeichenklassifikation: 3 Conv-Bloecke (Conv+BN+ReLU+Pool)
    gefolgt von zwei Fully-Connected-Schichten. Erwartet quadratische
    Eingabebilder (Default 64x64)."""

    def __init__(self, num_classes: int, img_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        reduced = img_size // 8  # 3x MaxPool(2) halbiert die Groesse dreimal
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced * reduced, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dirs", nargs="+", required=True, help="Ein oder mehrere Ordner mit Klassen-Unterordnern (0-9/A-Z)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs_stage3" / "char_ocr_model.pt"), help="Wohin das trainierte Modell gespeichert wird")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None, help="z.B. 'cuda', 'cpu'. Default: automatisch (GPU falls vorhanden)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage3] Verwende Geraet: {device}")

    samples, class_to_idx = collect_samples(args.data_dirs)

    n_val = int(len(samples) * args.val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Leichte Augmentierung fuers Training (die Quelldaten sind teils schon
    # augmentiert, deshalb hier bewusst zurueckhaltend); fuer die Validierung
    # nur Resize+Normalisierung, damit die Metrik nicht durch Zufall schwankt.
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    train_ds = CharDataset(train_samples, class_to_idx, train_transform)
    val_ds = CharDataset(val_samples, class_to_idx, val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SmallCharCNN(num_classes=len(class_to_idx), img_size=args.img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0
        print(
            f"[stage3] Epoche {epoch:3d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "img_size": args.img_size,
                },
                out_path,
            )
            # class_to_idx zusaetzlich als reines JSON speichern -- so kannst
            # du beim spaeteren Einlesen (Inferenz) die Zuordnung
            # Index -> Zeichen nachschlagen, ohne den ganzen Checkpoint laden
            # zu muessen.
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            with open(out_path.with_suffix(".classes.json"), "w", encoding="utf-8") as f:
                json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"\n[stage3] Fertig. Bestes Modell (val_acc={best_val_acc:.4f}) gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()
