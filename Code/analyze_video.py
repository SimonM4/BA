#!/usr/bin/env python3
"""
analyze_video.py

Die komplette Pipeline in EINEM Aufruf: Video (Datei, Ordner mit Videos,
RTSP-Stream der Drohne oder Webcam) rein -> erkannte Kennzeichen raus.

  Stufe 1 (optional): Fahrzeuge erkennen        (YOLO, VisDrone-trainiert)
  Stufe 2:            Kennzeichen erkennen       (YOLO, eigener Datensatz)
  Stufe 3:            Kennzeichen lesen          (Segmentierung + eigenes CNN
                                                  + syntaxgestuetzte Dekodierung)
  Tracking:           dasselbe Kennzeichen ueber mehrere Frames verfolgen und
                      die Lesungen per gewichteter Abstimmung zusammenfassen
                      (ein einzelner unscharfer Frame verfaelscht so nicht das
                      Ergebnis).

Ist ein Stufe-1-Modell vorhanden, sucht Stufe 2 nur innerhalb der
Fahrzeug-Ausschnitte (schneller und weniger Fehlalarme, Kennzeichen wird dem
Fahrzeug zugeordnet). Ohne Stufe-1-Modell laeuft Stufe 2 auf dem ganzen Bild.

Ergebnis (Ordner C:\\BA\\Ergebnisse\\<videoname>_<zeitstempel>\\):
  <video>_annotiert.mp4   Video mit eingezeichneten Fahrzeugen/Kennzeichen/Text
  kennzeichen.csv         EINE Zeile pro erkanntem Kennzeichen (Track):
                          Kennzeichen, Konfidenz, Zeitpunkt erste/letzte Sichtung, ...
  detektionen.csv         EINE Zeile pro Kennzeichen pro Frame (Rohdaten)
  crops\\                  bester Kennzeichen-Ausschnitt je Track + Segmentierung
  zusammenfassung.json    verwendete Modelle, Parameter, Laufzeit

Aufruf (aus C:\\BA\\Code):
  python analyze_video.py --source C:\\BA\\Videos\\flug1.mp4
  python analyze_video.py --source C:\\BA\\Videos                    (alle Videos im Ordner)
  python analyze_video.py --source rtsp://<ip>:8554/live --show     (Live-Stream)
  python analyze_video.py --source 0 --show                         (Webcam, zum Testen)

Wichtige Optionen:
  --stride 2          nur jeden 2. Frame analysieren (doppelt so schnell)
  --imgsz 1280        Eingabegroesse fuer die Kennzeichensuche im Gesamtbild
                      (groesser = kleine Kennzeichen besser, aber langsamer)
  --no-vehicles       Stufe 1 auch dann ueberspringen, wenn ein Modell da ist
  --plate-model / --vehicle-model / --ocr-model   Modelle explizit angeben
                      (sonst werden die neuesten trainierten Modelle in C:\\BA
                      automatisch gefunden)
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from plate_ocr_utils import GERMAN_PLATE_RE, PLAUSIBLE_MIN_CONF, PlateReading, decode_german_plate, gap_positions, segment_characters

ROOT = Path(__file__).resolve().parent.parent  # C:\BA
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".ts"}
VEHICLE_NAMES = {"car", "van", "truck", "bus", "motor", "motorcycle", "vehicle"}


# ---------------------------------------------------------------------------
# Hilfen: Modelle finden, Bilder speichern
# ---------------------------------------------------------------------------

def find_yolo_model(wanted: set, search_dirs: List[Path]) -> Optional[Path]:
    """Neuestes best.pt unter search_dirs, dessen Klassennamen eine der
    gesuchten Klassen enthalten (so wird z.B. ein Kennzeichen-Modell nicht
    versehentlich als Fahrzeug-Modell benutzt)."""
    from ultralytics import YOLO
    cands = []
    for d in search_dirs:
        if d.exists():
            cands += [p for p in d.rglob("best.pt") if "Daten" not in p.parts]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        try:
            names = {str(n).lower() for n in YOLO(str(p)).names.values()}
        except Exception:
            continue
        if names & wanted:
            return p
    return None


def imwrite(path: Path, img) -> None:
    """cv2.imwrite, das auch mit Umlauten im Pfad (Windows) funktioniert."""
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


def clip_box(x1, y1, x2, y2, W, H):
    return max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Tracking + Abstimmung
# ---------------------------------------------------------------------------

class Track:
    def __init__(self, tid: int, box, frame_idx: int, t: float):
        self.id = tid
        self.box = box
        self.last_frame = frame_idx
        self.first_t = t
        self.last_t = t
        self.votes: Dict[str, float] = defaultdict(float)
        self.n_readings = 0
        self.n_seen = 0
        self.vehicle_cls = ""
        self.best_crop = None
        self.best_crop_score = -1.0
        self.best_seg_vis = None

    def add(self, reading: PlateReading, det_conf: float):
        if not reading.text:
            return
        w = reading.confidence * det_conf * (2.0 if reading.plausible else 0.5)
        self.votes[reading.text] += w
        self.n_readings += 1

    def result(self):
        if not self.votes:
            return "", 0.0, 0.0
        text = max(self.votes, key=self.votes.get)
        total = sum(self.votes.values())
        return text, self.votes[text], self.votes[text] / total


class SimpleTracker:
    """IoU-Tracker: ordnet jede neue Box der Spur mit der groessten
    Ueberlappung zu (bzw. bei schneller Bewegung der naechstgelegenen Spur),
    sonst neue Spur. Bewusst einfach gehalten und ohne Zusatzpakete."""

    def __init__(self, max_age: int = 15, iou_thr: float = 0.2):
        self.tracks: Dict[int, Track] = {}
        self.all_tracks: List[Track] = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_thr = iou_thr

    def update(self, boxes, frame_idx: int, t: float) -> List[Track]:
        active = [tr for tr in self.tracks.values() if frame_idx - tr.last_frame <= self.max_age]
        pairs = []
        for bi, b in enumerate(boxes):
            for tr in active:
                o = iou(b, tr.box)
                if o < self.iou_thr:
                    # Fallback: Mittelpunktabstand relativ zur Boxgroesse
                    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                    tc = ((tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2)
                    diag = max(1.0, np.hypot(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1]))
                    d = np.hypot(bc[0] - tc[0], bc[1] - tc[1]) / diag
                    o = 0.19 - 0.1 * d if d < 1.0 else 0.0
                pairs.append((o, bi, tr.id))
        pairs.sort(reverse=True)
        assigned, used, result = {}, set(), [None] * len(boxes)
        for o, bi, tid in pairs:
            if o <= 0 or bi in assigned or tid in used:
                continue
            assigned[bi] = tid
            used.add(tid)
        for bi, b in enumerate(boxes):
            if bi in assigned:
                tr = self.tracks[assigned[bi]]
            else:
                tr = Track(self.next_id, b, frame_idx, t)
                self.tracks[tr.id] = tr
                self.all_tracks.append(tr)
                self.next_id += 1
            tr.box, tr.last_frame, tr.last_t = b, frame_idx, t
            tr.n_seen += 1
            result[bi] = tr
        return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, args):
        from ultralytics import YOLO
        from char_cnn import CharClassifier

        search = [ROOT / "runs_stage1", ROOT / "runs_stage2", ROOT / "kaggle_output", ROOT / "runs"]
        plate_path = Path(args.plate_model) if args.plate_model else find_yolo_model({"license_plate", "plate", "kennzeichen"}, search)
        if not plate_path or not plate_path.exists():
            sys.exit("FEHLER: kein Kennzeichen-Modell (Stufe 2) gefunden. Mit --plate-model angeben.")
        vehicle_path = None
        if not args.no_vehicles:
            vehicle_path = Path(args.vehicle_model) if args.vehicle_model else find_yolo_model(VEHICLE_NAMES, search)
        ocr_path = Path(args.ocr_model) if args.ocr_model else ROOT / "runs_stage3" / "char_ocr_model.pt"
        if not ocr_path.exists():
            sys.exit(f"FEHLER: kein Zeichen-Modell (Stufe 3) unter {ocr_path}. Erst train_stage3_char_ocr.py ausfuehren.")

        self.plate_model = YOLO(str(plate_path))
        self.vehicle_model = YOLO(str(vehicle_path)) if vehicle_path else None
        self.ocr = CharClassifier(ocr_path)
        self.vehicle_cls_ids = []
        if self.vehicle_model:
            self.vehicle_cls_ids = [i for i, n in self.vehicle_model.names.items() if str(n).lower() in VEHICLE_NAMES]
        self.paths = {"stufe1_fahrzeuge": str(vehicle_path) if vehicle_path else None,
                      "stufe2_kennzeichen": str(plate_path), "stufe3_zeichen": str(ocr_path)}
        self.args = args
        print("[pipeline] Stufe 1 (Fahrzeuge):    ", vehicle_path or "--- (nicht vorhanden/deaktiviert, Kennzeichensuche im Gesamtbild)")
        print("[pipeline] Stufe 2 (Kennzeichen):  ", plate_path)
        print("[pipeline] Stufe 3 (Zeichen-CNN):  ", ocr_path, f"({len(self.ocr.classes)} Klassen)")

    # --- Stufe 1 + 2 ---
    def detect_plates(self, frame):
        """-> Liste (plate_box(x1,y1,x2,y2), det_conf, vehicle_box|None, vehicle_cls)."""
        a = self.args
        H, W = frame.shape[:2]
        out = []
        if self.vehicle_model is not None:
            vr = self.vehicle_model.predict(frame, imgsz=a.imgsz, conf=a.vehicle_conf, classes=self.vehicle_cls_ids,
                                            verbose=False)[0]
            vboxes = vr.boxes.xyxy.cpu().numpy().astype(int).tolist()
            vcls = [self.vehicle_model.names[int(c)] for c in vr.boxes.cls.cpu().numpy()]
            crops, offs = [], []
            for (x1, y1, x2, y2) in vboxes:
                mx, my = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
                cx1, cy1, cx2, cy2 = clip_box(x1 - mx, y1 - my, x2 + mx, y2 + my, W, H)
                crops.append(frame[cy1:cy2, cx1:cx2])
                offs.append((cx1, cy1))
            if crops:
                prs = self.plate_model.predict(crops, imgsz=a.crop_imgsz, conf=a.plate_conf, verbose=False)
                for vb, vc, (ox, oy), pr in zip(vboxes, vcls, offs, prs):
                    if len(pr.boxes) == 0:
                        out.append((None, 0.0, vb, vc))
                        continue
                    k = int(pr.boxes.conf.argmax())  # pro Fahrzeug das sicherste Kennzeichen
                    x1, y1, x2, y2 = pr.boxes.xyxy[k].cpu().numpy().tolist()
                    out.append(((int(x1 + ox), int(y1 + oy), int(x2 + ox), int(y2 + oy)),
                                float(pr.boxes.conf[k]), vb, vc))
        else:
            pr = self.plate_model.predict(frame, imgsz=a.imgsz, conf=a.plate_conf, verbose=False)[0]
            for b, c in zip(pr.boxes.xyxy.cpu().numpy(), pr.boxes.conf.cpu().numpy()):
                out.append((tuple(int(v) for v in b), float(c), None, ""))
        return out

    # --- Stufe 3 ---
    def read_plate(self, frame, box):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
        x1, y1, x2, y2 = clip_box(x1 - mx, y1 - my, x2 + mx, y2 + my, W, H)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < self.args.min_plate_px or crop.shape[1] < 2 * self.args.min_plate_px:
            return PlateReading("", 0.0, False, ""), crop, [], None
        cache = {}

        def scorer(chars, boxes):
            # erkennungsgestuetzte Segmentierung: das CNN bewertet den Kandidaten
            r = decode_german_plate(self.ocr.predict_proba(chars), self.ocr.classes, gap_positions(boxes))
            cache[id(boxes)] = r
            return float(np.log(max(r.confidence, 1e-6))) + (0.3 if r.plausible else 0.0) - 0.15 * max(0, 5 - len(chars))

        use_scorer = self.args.seg_candidates > 1
        chars, cboxes, gray = segment_characters(crop, scorer if use_scorer else None,
                                                 max_candidates=max(1, self.args.seg_candidates))
        reading = cache.get(id(cboxes))
        if reading is None:
            reading = decode_german_plate(self.ocr.predict_proba(chars), self.ocr.classes, gap_positions(cboxes))
        return reading, crop, cboxes, gray


def draw_label(img, text, x, y, color):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    y = max(tht + 4, y)
    cv2.rectangle(img, (x, y - tht - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (0, 0, 0), th, cv2.LINE_AA)


def seg_visualization(gray, cboxes, reading: PlateReading):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in cboxes:
        cv2.rectangle(vis, (b.x, b.y), (b.x2, b.y + b.h), (0, 0, 255), 1)
    pad = np.full((28, vis.shape[1], 3), 255, np.uint8)
    cv2.putText(pad, f"{reading.text}  raw={reading.raw}", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([vis, pad])


def open_source(src: str):
    if src.isdigit():
        return cv2.VideoCapture(int(src)), True
    is_stream = src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))
    return cv2.VideoCapture(src), is_stream


def process(source: str, pipe: Pipeline, args) -> Path:
    cap, is_stream = open_source(source)
    if not cap.isOpened():
        print(f"[pipeline] FEHLER: Quelle laesst sich nicht oeffnen: {source}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
    name = "stream" if is_stream else Path(source).stem
    out_dir = Path(args.out_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    print(f"\n[pipeline] Quelle: {source}  ({n_total or '?'} Frames, {fps:.1f} fps)\n[pipeline] Ausgabe: {out_dir}")

    writer = None
    tracker = SimpleTracker(max_age=args.track_age)
    vtracker = SimpleTracker(max_age=args.track_age)
    det_rows = []
    frame_idx, processed, t_start = -1, 0, time.time()
    t_stream0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        if frame_idx % args.stride:
            continue
        t = (time.time() - t_stream0) if is_stream else frame_idx / fps
        processed += 1

        dets = pipe.detect_plates(frame)
        vis = frame.copy()

        # Fahrzeuge (falls Stufe 1 aktiv) zeichnen
        vboxes = [d[2] for d in dets if d[2] is not None]
        vtracks = vtracker.update(vboxes, frame_idx, t) if vboxes else []
        vt_iter = iter(vtracks)
        vtrack_of = []
        for d in dets:
            vtrack_of.append(next(vt_iter) if d[2] is not None else None)
            if d[2] is not None:
                x1, y1, x2, y2 = d[2]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 2)

        plates = [(d, vt) for d, vt in zip(dets, vtrack_of) if d[0] is not None]
        ptracks = tracker.update([d[0] for d, _ in plates], frame_idx, t)
        for (d, vt), tr in zip(plates, ptracks):
            box, conf, vbox, vcls = d
            reading, crop, cboxes, gray = pipe.read_plate(frame, box)
            tr.add(reading, conf)
            tr.vehicle_cls = vcls or tr.vehicle_cls
            score = conf * crop.shape[0] * (1.5 if reading.plausible else 1.0)
            if crop.size and score > tr.best_crop_score:
                tr.best_crop_score = score
                tr.best_crop = crop.copy()
                if gray is not None:
                    tr.best_seg_vis = seg_visualization(gray, cboxes, reading)
            stable, _, agree = tr.result()
            det_rows.append({
                "frame": frame_idx, "zeit_s": round(t, 3), "track_id": tr.id,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "det_konfidenz": round(conf, 3),
                "fahrzeug_klasse": vcls, "fahrzeug_track": vt.id if vt else "",
                "text_frame": reading.text, "ocr_konfidenz": round(reading.confidence, 3),
                "plausibel": reading.plausible, "roh": reading.raw, "text_stabil": stable,
            })
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"#{tr.id} {stable}" if stable else f"#{tr.id} ?"
            draw_label(vis, label, x1, y1 - 2, (0, 220, 0))

        # Kopfzeile
        info = f"Frame {frame_idx}  t={t:6.1f}s  Kennzeichen-Tracks: {len(tracker.all_tracks)}"
        cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None and not args.no_video:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / f"{name}_annotiert.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.stride, (w, h))
        if writer is not None:
            writer.write(vis)
        if args.show:
            disp = vis if vis.shape[1] <= 1600 else cv2.resize(vis, None, fx=1600 / vis.shape[1], fy=1600 / vis.shape[1])
            cv2.imshow("BA Kennzeichenerkennung (q = beenden)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if processed % 25 == 0:
            el = time.time() - t_start
            prog = f"{frame_idx}/{n_total}" if n_total else str(frame_idx)
            print(f"[pipeline] Frame {prog}  {processed / el:.1f} Frames/s  Tracks: {len(tracker.all_tracks)}", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    runtime = time.time() - t_start

    # --- Ergebnisse schreiben ---
    with open(out_dir / "detektionen.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["frame", "zeit_s", "track_id", "x1", "y1", "x2", "y2", "det_konfidenz", "fahrzeug_klasse",
                  "fahrzeug_track", "text_frame", "ocr_konfidenz", "plausibel", "roh", "text_stabil"]
        wr = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        wr.writeheader()
        wr.writerows(det_rows)

    summary_rows = []
    for tr in tracker.all_tracks:
        if tr.n_seen < args.min_track_frames:
            continue
        text, _, agree = tr.result()
        crop_file = ""
        if tr.best_crop is not None and tr.best_crop.size:
            crop_file = f"crops/track_{tr.id:04d}.jpg"
            imwrite(out_dir / crop_file, tr.best_crop)
            if tr.best_seg_vis is not None:
                imwrite(out_dir / f"crops/track_{tr.id:04d}_segmentierung.jpg", tr.best_seg_vis)
        mean_conf = np.mean([r["ocr_konfidenz"] for r in det_rows if r["track_id"] == tr.id and r["text_frame"] == text]) if text else 0.0
        summary_rows.append({
            "track_id": tr.id, "kennzeichen": text or "(nicht lesbar)",
            "ocr_konfidenz": round(float(mean_conf), 3), "uebereinstimmung": round(agree, 3),
            "lesungen": tr.n_readings, "frames": tr.n_seen,
            "erste_sichtung_s": round(tr.first_t, 2), "letzte_sichtung_s": round(tr.last_t, 2),
            "fahrzeug_klasse": tr.vehicle_cls, "plausibel": bool(text) and bool(GERMAN_PLATE_RE.match(text)) and mean_conf >= PLAUSIBLE_MIN_CONF,
            "crop": crop_file,
        })
    with open(out_dir / "kennzeichen.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["track_id", "kennzeichen", "ocr_konfidenz", "uebereinstimmung", "lesungen", "frames",
                  "erste_sichtung_s", "letzte_sichtung_s", "fahrzeug_klasse", "plausibel", "crop"]
        wr = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        wr.writeheader()
        wr.writerows(summary_rows)

    summary = {
        "quelle": source, "ausgabe": str(out_dir), "modelle": pipe.paths,
        "parameter": {k: v for k, v in vars(args).items() if k != "source"},
        "frames_gesamt": frame_idx + 1, "frames_analysiert": processed, "laufzeit_s": round(runtime, 1),
        "frames_pro_s": round(processed / max(runtime, 1e-6), 2),
        "kennzeichen": [r["kennzeichen"] for r in summary_rows],
    }
    (out_dir / "zusammenfassung.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[pipeline] Fertig: {processed} Frames in {runtime:.0f}s ({summary['frames_pro_s']} Frames/s)")
    print(f"[pipeline] {len(summary_rows)} Kennzeichen-Tracks:")
    for r in summary_rows:
        print(f"   #{r['track_id']:<4} {r['kennzeichen']:<14} Konf. {r['ocr_konfidenz']:.2f}  "
              f"Uebereinst. {r['uebereinstimmung']:.2f}  t={r['erste_sichtung_s']}-{r['letzte_sichtung_s']}s")
    print(f"[pipeline] Ergebnisse: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Videodatei, Ordner mit Videos, RTSP-URL oder Kameranummer")
    ap.add_argument("--out-dir", default=str(ROOT / "Ergebnisse"))
    ap.add_argument("--plate-model", default=None)
    ap.add_argument("--vehicle-model", default=None)
    ap.add_argument("--ocr-model", default=None)
    ap.add_argument("--no-vehicles", action="store_true", help="Stufe 1 nicht verwenden")
    ap.add_argument("--imgsz", type=int, default=1280, help="Eingabegroesse Gesamtbild (Stufe 1 bzw. Stufe 2 ohne Stufe 1)")
    ap.add_argument("--crop-imgsz", type=int, default=640, help="Eingabegroesse Stufe 2 auf Fahrzeug-Ausschnitten")
    ap.add_argument("--plate-conf", type=float, default=0.35)
    ap.add_argument("--vehicle-conf", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=1, help="nur jeden n-ten Frame analysieren")
    ap.add_argument("--max-frames", type=int, default=0, help="nach so vielen Frames abbrechen (0 = alle)")
    ap.add_argument("--min-plate-px", type=int, default=10, help="kleinere Kennzeichen (Hoehe in Pixel) nicht lesen")
    ap.add_argument("--seg-candidates", type=int, default=1,
                    help="so viele Segmentierungs-Kandidaten vom CNN bewerten lassen (1 = nur Geometrie-Heuristik)")
    ap.add_argument("--track-age", type=int, default=15, help="Frames, die eine Spur ohne Detektion ueberlebt")
    ap.add_argument("--min-track-frames", type=int, default=2, help="Spuren mit weniger Frames nicht berichten")
    ap.add_argument("--no-video", action="store_true", help="kein annotiertes Video schreiben")
    ap.add_argument("--show", action="store_true", help="Live-Fenster anzeigen")
    ap.add_argument("--open-result", action="store_true", help="Ergebnisordner am Ende im Explorer oeffnen (Windows)")
    args = ap.parse_args()

    pipe = Pipeline(args)
    src = args.source
    p = Path(src)
    if p.is_dir():
        vids = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        if not vids:
            sys.exit(f"Keine Videos in {p} gefunden.")
        outs = [process(str(v), pipe, args) for v in vids]
    else:
        outs = [process(src, pipe, args)]
    outs = [o for o in outs if o]
    if args.open_result and outs and hasattr(os, "startfile"):
        os.startfile(str(outs[-1] if len(outs) == 1 else Path(args.out_dir)))


if __name__ == "__main__":
    main()
