#!/usr/bin/env python3
"""
eval_stage3_plates.py

Bewertet STUFE 3 (Segmentierung + Zeichen-CNN + Dekodierung) auf ECHTEN
Kennzeichen-Ausschnitten statt auf einzelnen Zeichenbildern. Die
Kennzeichen-Boxen kommen aus den YOLO-Labels (Ground Truth), damit nur
Stufe 3 bewertet wird und Fehler von Stufe 2 nicht hineinspielen.

Der Klartext der Kennzeichen steht in einer CSV (Datei;Index;Text). Die
mitgelieferte Datei Daten/kennzeichen_text_gt.csv enthaelt eine kleine,
von Hand abgelesene Stichprobe aus dem Testsplit -- fuer belastbare Zahlen in
der Arbeit bitte pruefen und erweitern.

Kennzahlen:
  - Exakt richtig gelesene Kennzeichen (ohne Bindestrich/Leerzeichen)
  - CER (Character Error Rate) = Levenshtein-Distanz / Anzahl Zeichen

Aufruf (aus C:\\BA\\Code):
  python eval_stage3_plates.py
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from char_cnn import CharClassifier
from plate_ocr_utils import decode_german_plate, gap_positions, segment_characters

ROOT = Path(__file__).resolve().parent.parent


def levenshtein(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[:], i
        for j, cb in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (ca != cb))
    return d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "Daten" / "kennzeichen_detection_dataset"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--gt", default=str(ROOT / "Daten" / "kennzeichen_text_gt.csv"))
    ap.add_argument("--ocr-model", default=str(ROOT / "runs_stage3" / "char_ocr_model.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs_stage3" / "eval_kennzeichen"))
    ap.add_argument("--seg-candidates", type=int, default=1, help="wie in analyze_video.py (1 = nur Heuristik)")
    args = ap.parse_args()

    ds = Path(args.dataset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clf = CharClassifier(args.ocr_model)

    rows, vis_rows = [], []
    n_exact = n_edits = n_chars = 0
    with open(args.gt, encoding="utf-8-sig") as f:
        gt = [r for r in csv.DictReader(f, delimiter=";")]
    for r in gt:
        img_path = ds / "images" / args.split / r["datei"]
        lbl_path = ds / "labels" / args.split / (Path(r["datei"]).stem + ".txt")
        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        lines = lbl_path.read_text().splitlines()
        H, W = img.shape[:2]
        _, cx, cy, bw, bh = (float(v) for v in lines[int(r["index"])].split()[:5])
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        crop = img[max(0, y1):y2, max(0, x1):x2]

        cache = {}

        def scorer(cs, bx):
            res = decode_german_plate(clf.predict_proba(cs), clf.classes, gap_positions(bx))
            cache[id(bx)] = res
            return float(np.log(max(res.confidence, 1e-6))) + (0.3 if res.plausible else 0.0) - 0.15 * max(0, 5 - len(cs))

        chars, boxes, gray = segment_characters(crop, scorer if args.seg_candidates > 1 else None,
                                                max_candidates=max(1, args.seg_candidates))
        rd = cache.get(id(boxes)) or decode_german_plate(clf.predict_proba(chars), clf.classes, gap_positions(boxes))
        pred = rd.text.replace("-", "").replace(" ", "")
        truth = r["text"].replace("-", "").replace(" ", "").upper()
        e = levenshtein(pred, truth)
        n_exact += pred == truth
        n_edits += e
        n_chars += len(truth)
        rows.append({"datei": r["datei"], "wahr": r["text"], "gelesen": rd.text, "roh": rd.raw,
                     "konfidenz": round(rd.confidence, 3), "plausibel": rd.plausible,
                     "zeichen_segmentiert": len(boxes), "fehler": e})
        print(f"{r['datei']:<24} wahr={r['text']:<12} gelesen={rd.text:<12} roh={rd.raw:<10} "
              f"konf={rd.confidence:.2f} fehler={e}")

        v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 and gray.size > 1 else np.zeros((80, 200, 3), np.uint8)
        for b in boxes:
            cv2.rectangle(v, (b.x, b.y), (b.x2, b.y + b.h), (0, 0, 255), 1)
        v = cv2.resize(v, (int(v.shape[1] * 80 / v.shape[0]), 80))[:, :420]
        v = cv2.copyMakeBorder(v, 0, 0, 0, 420 - v.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        txt = np.full((80, 380, 3), 255, np.uint8)
        cv2.putText(txt, f"wahr:    {r['text']}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        color = (0, 140, 0) if e == 0 else (0, 0, 200)
        cv2.putText(txt, f"gelesen: {rd.text} ({rd.confidence:.2f})", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        vis_rows.append(np.hstack([v, txt]))

    with open(out / "ergebnisse.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        wr.writeheader()
        wr.writerows(rows)
    if vis_rows:
        ok, buf = cv2.imencode(".jpg", np.vstack(vis_rows))
        buf.tofile(str(out / "uebersicht.jpg"))
    cer = n_edits / max(1, n_chars)
    print(f"\nExakt richtig: {n_exact}/{len(gt)} ({100 * n_exact / max(1, len(gt)):.0f} %)   CER: {100 * cer:.1f} %")
    print(f"Details: {out}")


if __name__ == "__main__":
    main()
