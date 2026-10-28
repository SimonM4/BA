#!/usr/bin/env python3
"""
train_stage3_char_ocr.py

STUFE 3 der Pipeline: die einzelnen Zeichen eines (durch Stufe 2 gefundenen
und durch plate_ocr_utils.segment_characters() zerlegten) Kennzeichens lesen.

Das ist KEINE Objekterkennung, sondern eine KLASSIFIKATION: Fuer jeden
Zeichen-Ausschnitt sagt ein kleines, selbst geschriebenes CNN (char_cnn.py),
welches Zeichen (0-9, A-Z) es ist -- oder "_" = kein Zeichen.

Datenquellen (Ordner ODER direkt die Zip-Dateien, Struktur je
".../<klasse>/<bild>"):
  - archive (4).zip            ("CNN letter Dataset/<klasse>/...", echte
                                 graue Zeichen-Ausschnitte, ~35.500 Bilder, ohne "O")
  - DeutschEinzelneZahlen.zip   (FE-Schrift, weiss auf schwarz, ~3.000 Bilder)
Die Klassenliste ist die Vereinigung aller gefundenen Ordnernamen.

Optional (--negatives-from): "kein Zeichen"-Beispiele (Klasse "_") werden
automatisch aus dem Kennzeichen-Detection-Datensatz geschnitten (EU-Feld,
Rahmen, Plaketten-Bereich, Hintergrund). Damit lernt das Netz, Stoerobjekte,
die die Segmentierung faelschlich als Zeichen liefert, zu verwerfen.

Unterschiede zur ersten Version (wichtig fuer die Arbeit):
  * Einheitliche Vorverarbeitung fuer Training UND Inferenz
    (plate_ocr_utils.normalize_char_image: Graustufen, dunkles Zeichen auf
    hellem Grund, Kontrast gestreckt, 48x48) -> gleicht die zwei sehr
    unterschiedlichen Quellen an.
  * Starke "Drohnen-Augmentierung": zufaellige Verschiebung/Skalierung/Scherung
    (ungenaue Segmentierung, Perspektive), Herunter- und wieder Hochskalieren
    auf 10-28 px Zeichenhoehe (geringe Aufloesung aus der Luft), Unschaerfe,
    Rauschen, Kontrast. Ohne diese Augmentierung erkennt das Modell zwar die
    Validierungsbilder fast perfekt, aber echte Kennzeichen-Ausschnitte schlecht.
  * Klassengewichte gegen Ungleichgewicht (z.B. "O" nur ~80 Bilder).
  * Bilder werden einmal in den Speicher geladen -> schnelles Training auf
    der CPU (kein DataLoader-Multiprocessing noetig, Windows-sicher).
  * Ausgabe: Modell, Klassenliste, Trainingsverlauf (CSV + PNG) und
    Konfusionsmatrix im Ausgabeordner.

Aufruf (aus C:\\BA\\Code):
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

import cv2
import numpy as np
import torch
import torch.nn as nn

from char_cnn import NO_CHAR, CharCNN, to_tensor
from plate_ocr_utils import CHAR_ASPECT, CHAR_SIZE, normalize_char_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
RAW_H, RAW_W = 80, 60  # Rohbilder werden in dieser Groesse (Seitenverhaeltnis 0,75) gespeichert


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

def _decode(buf: bytes):
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _is_class_name(name: str) -> bool:
    return len(name) == 1 and name.isalnum()


def load_source(path: Path):
    """Liefert Liste (graues Bild RAW_HxRAW_W, Klasse) aus Ordner oder Zip.
    Klasse = Name des direkten Elternordners, sofern ein einzelnes Zeichen."""
    items = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                parts = name.replace("\\", "/").split("/")
                if len(parts) < 2 or Path(name).suffix.lower() not in IMG_EXTS:
                    continue
                cls = parts[-2].upper()
                if not _is_class_name(cls):
                    continue
                img = _decode(zf.read(name))
                if img is not None:
                    items.append((cv2.resize(img, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA), cls))
    else:
        for p in path.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and _is_class_name(p.parent.name.upper()):
                img = _decode(p.read_bytes())
                if img is not None:
                    items.append((cv2.resize(img, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA), p.parent.name.upper()))
    return items


def load_negatives(det_root: Path, n: int, rng: random.Random):
    """Schneidet 'kein Zeichen'-Beispiele aus dem Kennzeichen-Detection-
    Datensatz (YOLO-Format): Ausschnitte am linken Kennzeichenrand (EU-Feld),
    an Ober-/Unterkante (Rahmen), in der Kennzeichenmitte mit falschem
    Seitenverhaeltnis und aus dem Hintergrund."""
    img_dir = det_root / "images" / "train"
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
            cx, cy, bw, bh = (float(t) for t in v[1:5])
            x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            pw, ph = int(bw * W), int(bh * H)
            if ph < 12 or pw < 30:
                continue
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
                x, y = max(0, x), max(0, y)
                crop = img[y:y + ch, x:x + cw]
                if crop.shape[0] >= 8 and crop.shape[1] >= 6:
                    out.append((cv2.resize(crop, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA), NO_CHAR))
    return out[:n]


# ---------------------------------------------------------------------------
# Augmentierung ("so sieht ein Zeichen aus der Drohne aus")
# ---------------------------------------------------------------------------

def augment(g: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = g.shape
    # 0) Strichstaerke variieren: Kennzeichenschrift ist aus der Luft oft
    #    duenner/verwaschener als die Trainingszeichen (Graustufen-Erosion =
    #    dunkle Striche dicker, Dilatation = duenner)
    r = rng.random()
    if r < 0.35:
        g = cv2.dilate(g, np.ones((rng.choice([2, 3]), rng.choice([2, 3])), np.uint8))
    elif r < 0.5:
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    # 1) geometrisch: Verschiebung, Skalierung (Breite und Hoehe getrennt,
    #    simuliert schmale FE-Schrift), Scherung, leichte Drehung
    sy = rng.uniform(0.85, 1.15)
    sx = sy * rng.uniform(0.7, 1.15)
    ang = np.radians(rng.uniform(-8, 8))
    sh = rng.uniform(-0.2, 0.2)
    a, b = sx * np.cos(ang), -sy * np.sin(ang) + sh
    c, d = sx * np.sin(ang), sy * np.cos(ang)
    tx = rng.uniform(-0.1, 0.1) * w
    ty = rng.uniform(-0.07, 0.07) * h
    M = np.float32([[a, b, w / 2 - a * w / 2 - b * h / 2 + tx],
                    [c, d, h / 2 - c * w / 2 - d * h / 2 + ty]])
    g = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 2) geringe Aufloesung simulieren
    if rng.random() < 0.7:
        th = rng.randint(10, 28)
        g = cv2.resize(g, (max(4, int(th * w / h)), th), interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    # 3) Unschaerfe (Bewegung/Fokus)
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        g = cv2.GaussianBlur(g, (k, k), rng.uniform(0.5, 1.5))
    # 4) Kontrast/Helligkeit + Rauschen
    g = g.astype(np.float32)
    g = (g - 128) * rng.uniform(0.5, 1.3) + 128 + rng.uniform(-40, 40)
    if rng.random() < 0.5:
        g += np.random.normal(0, rng.uniform(2, 12), g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    # 5) JPEG-Artefakte
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 70)])
        if ok:
            g = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return g


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_batches(n, bs, shuffle, rng):
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def evaluate(model, X_raw, y, classes, rng, degrade: bool, bs=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for b in make_batches(len(X_raw), bs, False, rng):
            imgs = [normalize_char_image(augment(X_raw[i], rng) if degrade else X_raw[i]) for i in b]
            preds.append(model(to_tensor(imgs)).argmax(1).numpy())
    preds = np.concatenate(preds) if preds else np.array([], int)
    return float((preds == y).mean()) if len(y) else 0.0, preds


def save_plots(out_dir: Path, history, cm, classes):
    try:
        import matplotlib
        matplotlib.use("Agg")
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

    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Vorhersage"); ax.set_ylabel("Wahr")
    ax.set_title("Konfusionsmatrix (Val, Drohnen-Augmentierung, zeilennormiert)")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=120); plt.close(fig)


def main() -> None:
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

    rng = random.Random(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Daten ---
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

    classes = sorted(set(c for _, c in items))
    cidx = {c: i for i, c in enumerate(classes)}
    X = [im for im, _ in items]
    y = np.array([cidx[c] for _, c in items])
    counts = Counter(y.tolist())
    print(f"[stage3] {len(X)} Bilder, {len(classes)} Klassen: {' '.join(classes)}")

    # stratifizierter Split
    order = list(range(len(X)))
    rng.shuffle(order)
    val_idx, train_idx, seen = [], [], Counter()
    for i in order:
        if seen[y[i]] < int(round(counts[y[i]] * args.val_split)):
            val_idx.append(i); seen[y[i]] += 1
        else:
            train_idx.append(i)
    # Ueberabtastung NUR im Trainingsteil (sonst landen Duplikate auch in Val)
    train_idx = [i for i in train_idx for _ in range(repeat_of[i])]
    Xtr, ytr = [X[i] for i in train_idx], y[train_idx]
    Xva, yva = [X[i] for i in val_idx], y[val_idx]
    print(f"[stage3] Train {len(Xtr)} / Val {len(Xva)}")

    # Klassengewichte (inverse Wurzel der Haeufigkeit)
    freq = np.array([max(1, Counter(ytr.tolist())[k]) for k in range(len(classes))], np.float32)
    w = (freq.mean() / freq) ** 0.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean()), label_smoothing=0.05)

    model = CharCNN(len(classes))
    print(f"[stage3] Modell: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter, "
          f"Torch-Threads: {torch.get_num_threads()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(Xtr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    best_acc, history = -1.0, []
    model_path = out_dir / "char_ocr_model.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, n_seen = 0.0, 0
        for b in make_batches(len(Xtr), args.batch_size, True, rng):
            imgs = [normalize_char_image(augment(Xtr[i], rng) if rng.random() < 0.85 else Xtr[i]) for i in b]
            xb, yb = to_tensor(imgs), torch.from_numpy(ytr[b])
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            sched.step()
            tot_loss += loss.item() * len(b); n_seen += len(b)
        acc_clean, _ = evaluate(model, Xva, yva, classes, rng, degrade=False)
        acc_deg, _ = evaluate(model, Xva, yva, classes, random.Random(123), degrade=True)
        h = dict(epoch=epoch, train_loss=tot_loss / n_seen, val_acc_clean=acc_clean,
                 val_acc_degraded=acc_deg, seconds=time.time() - t0)
        history.append(h)
        print(f"[stage3] Epoche {epoch:2d}/{args.epochs} loss={h['train_loss']:.4f} "
              f"val_acc={acc_clean:.4f} val_acc_drohne={acc_deg:.4f} ({h['seconds']:.0f}s)", flush=True)
        if acc_deg > best_acc:
            best_acc = acc_deg
            torch.save({"model_state": model.state_dict(), "classes": classes, "img_size": CHAR_SIZE,
                        "epoch": epoch, "val_acc_clean": acc_clean, "val_acc_degraded": acc_deg}, model_path)
        with open(out_dir / "results.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            wr.writeheader(); wr.writerows(history)

    # Abschlussauswertung mit dem besten Modell
    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    acc_deg, preds = evaluate(model, Xva, yva, classes, random.Random(123), degrade=True)
    cm = np.zeros((len(classes), len(classes)), np.int64)
    for t, p in zip(yva, preds):
        cm[t, p] += 1
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


if __name__ == "__main__":
    main()
