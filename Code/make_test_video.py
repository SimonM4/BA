#!/usr/bin/env python3
"""
make_test_video.py

Erzeugt ein kuenstliches Testvideo aus Einzelbildern (Standard: Testsplit des
Kennzeichen-Datensatzes), solange noch keine eigenen Drohnenaufnahmen
vorliegen. Jedes Bild wird ~2 s lang mit langsamer Kamerafahrt (Zoom +
Verschiebung) gezeigt, damit Detektion, Tracking und die Abstimmung ueber
mehrere Frames realistisch getestet werden.

ACHTUNG fuer die Arbeit: Die Testbilder sind Bodenaufnahmen, keine
Luftbilder -- das Video testet die Technik der Pipeline, nicht die
Erkennungsleistung aus Drohnenperspektive.

Aufruf:
  python make_test_video.py
  python make_test_video.py --images C:\\BA\\Daten\\kennzeichen_detection_dataset\\images\\test --n 30 --out C:\\BA\\Videos\\testvideo.mp4
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "Daten" / "kennzeichen_detection_dataset" / "images" / "test"))
    ap.add_argument("--out", default=str(ROOT / "Videos" / "testvideo_kennzeichen.mp4"))
    ap.add_argument("--n", type=int, default=30, help="Anzahl Bilder")
    ap.add_argument("--seconds", type=float, default=2.0, help="Sekunden pro Bild")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[:args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter konnte nicht geoeffnet werden (Codec mp4v nicht verfuegbar?)")
    n_frames = int(args.seconds * args.fps)
    listing = []
    for f in files:
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        base = min(W / iw, H / ih)  # Bild fuellt das Video in einer Richtung
        z0, z1 = rng.uniform(1.0, 1.1), rng.uniform(1.15, 1.35)
        dx, dy = rng.uniform(-0.08, 0.08) * W, rng.uniform(-0.06, 0.06) * H
        for k in range(n_frames):
            a = k / max(1, n_frames - 1)
            z = base * (z0 + (z1 - z0) * a)
            tx = W / 2 - z * iw / 2 + dx * a
            ty = H / 2 - z * ih / 2 + dy * a
            M = np.float32([[z, 0, tx], [0, z, ty]])
            frame = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(40, 40, 40))
            vw.write(frame)
        listing.append(f.name)
    vw.release()
    (out.with_suffix(".bilder.txt")).write_text("\n".join(listing), encoding="utf-8")
    print(f"Testvideo: {out}  ({len(listing)} Bilder, {len(listing) * args.seconds:.0f} s, {W}x{H} @ {args.fps} fps)")


if __name__ == "__main__":
    main()
