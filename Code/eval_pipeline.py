#!/usr/bin/env python3
"""
eval_pipeline.py  --  WIE GUT IST DIE PIPELINE?  (alle Tests in einem Lauf)

===========================================================================
WAS WIRD GETESTET?
===========================================================================
  A  stufe1      Stufe 1 (Fahrzeuge) auf dem offiziellen VisDrone-Testsplit
                 "test-dev" (1.610 Drohnenbilder, im Training nie gesehen).
  B  stufe2      Stufe 2 (Kennzeichen) auf dem Testsplit des
                 Kennzeichen-Datensatzes (132 Bilder, nie im Training).
                 Altes CPU-Testmodell (6 Epochen) und neues Kaggle-Modell
                 im direkten Vergleich.
  C  simulation  Simulierte Drohnenbedingungen auf denselben 132 Testbildern
                 (Kern der Forschungsfrage):
                   - Aufloesung : Kennzeichenhoehe in Pixeln wird schrittweise
                                  verkleinert (Bild herunterskaliert)
                   - Blickwinkel: Kennzeichen von schraeg oben gesehen
                                  (vertikale Stauchung um cos(Winkel))
                   - Unschaerfe : Gauss-Weichzeichner
                   - Kompression: JPEG-Qualitaet
                 Gemessen wird pro Bedingung:
                   Stufe 2   : Wird das Kennzeichen gefunden? (Recall, IoU >= 0,5)
                   Stufe 3   : Wird es richtig gelesen, wenn die Box stimmt?
                               (OCR mit Ground-Truth-Box, 13 Bilder mit Klartext)
                   Ende-zu-Ende: Stufe 2 + 3 zusammen (13 Bilder mit Klartext)
  D  mrtmd       ECHTE Kennzeichen aus erhoehter Verkehrskamera in 6
                 Aufloesungen (MRTMD-Datensatz, Bugeja et al.), nur Stufe 3.
                 Achtung: auslaendische Kennzeichen (andere Schrift, anderes
                 Format) -> untere Schranke, zeigt aber den Aufloesungseffekt
                 an echten, nicht simulierten Bildern.
  E  speed       Rechenzeit pro Stufe auf diesem Rechner (CPU).

===========================================================================
WICHTIGE ANNAHMEN (fuer die Methodik der Arbeit)
===========================================================================
- Aufloesung: Das ganze Testbild wird so verkleinert, dass das Kennzeichen
  die Zielhoehe in Pixeln hat. Die Testbilder zeigen meist ein Auto formatfuellend,
  entsprechen also dem Fahrzeug-Ausschnitt, den Stufe 1 in der Pipeline an
  Stufe 2 weitergibt. Ein verkleinertes Testbild entspricht damit dem
  Fahrzeug-Ausschnitt aus einem Drohnenbild, in dem das Kennzeichen genau
  diese Pixelhoehe hat. Wie in der Pipeline wird der Ausschnitt fuer Stufe 2
  auf 640 px hochskaliert und Kennzeichen unter 10 px Hoehe werden nicht
  gelesen (--min-plate-px 10).
- Blickwinkel: affine Naeherung (Stauchung um cos(Winkel), keine
  Perspektivverzerrung). Bei grossem Abstand im Verhaeltnis zur
  Kennzeichengroesse ist das eine uebliche Naeherung.
- Es wird pro Bild nur EIN Kennzeichen bewertet (das mit Klartext-Ground-Truth
  bzw. sonst das groesste), damit jede Bedingung sauber einer Pixelhoehe
  zugeordnet werden kann.
- Kleine Stichprobe: Nur 13 Testbilder haben Klartext-Ground-Truth. Die
  OCR-Zahlen sind deshalb grob (ein Bild = 7,7 Prozentpunkte). Die
  Detektionszahlen (Stufe 2) beruhen auf allen Testbildern.

===========================================================================
AUSGABE
===========================================================================
C:\\BA\\Tests\\pipeline_test_<datum_zeit>\\
  bericht.md                    alle Kennzahlen als Tabellen
  ergebnisse.json               dieselben Zahlen maschinenlesbar
  simulation_details.csv        jede einzelne Messung der Simulation
  mrtmd_details.csv             jede einzelne Lesung der MRTMD-Kennzeichen
  *.png                         Diagramme
  stufe1_visdrone_test/ ...     Ultralytics-Auswertung (PR-Kurve, Konfusionsmatrix)

Aufruf (aus C:\\BA\\Code), siehe Start\\05_pipeline_testen.bat:
  python eval_pipeline.py                      (alle Tests, ca. 15-30 min auf CPU)
  python eval_pipeline.py --tests stufe2 simulation
  python eval_pipeline.py --quick              (Kurztest mit wenigen Bildern)
"""

import argparse
import csv
import json
import random
import shutil
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # C:\BA
sys.path.insert(0, str(Path(__file__).resolve().parent))

from char_cnn import CharClassifier                                    # noqa: E402
from eval_stage3_plates import levenshtein                             # noqa: E402
from plate_ocr_utils import (PlateReading, decode_german_plate,        # noqa: E402
                             gap_positions, segment_characters)

# ---------------------------------------------------------------------------
# Modelle und Daten (Standardpfade)
# ---------------------------------------------------------------------------
STAGE1_MODEL = ROOT / "runs_stage1" / "vehicle_detection" / "weights" / "best.pt"
STAGE2_MODEL = ROOT / "runs_stage2" / "plate_detection" / "weights" / "best.pt"
STAGE2_OLD = ROOT / "runs_stage2" / "plate_detection_first_test" / "weights" / "best.pt"
STAGE3_MODEL = ROOT / "runs_stage3" / "char_ocr_model.pt"
PLATE_DS = ROOT / "Daten" / "kennzeichen_detection_dataset"
GT_CSV = ROOT / "Daten" / "kennzeichen_text_gt.csv"
MRTMD_ZIP = ROOT / "Daten" / "MRTMD-main.zip"

# Pipeline-Einstellungen (wie in analyze_video.py)
PLATE_CONF = 0.35
CROP_IMGSZ = 640
MIN_PLATE_PX = 10
IOU_MATCH = 0.5

# Stufen der Simulation
RES_LEVELS = [None, 28, 24, 20, 17, 14, 12, 10, 8, 6]   # Kennzeichenhoehe in px (None = Original)
ANGLES = [0, 20, 30, 40, 50, 60, 70]                    # Grad von der Horizontalen
ANGLE_BASES = [None, 20]                                # bei Original bzw. 20 px Ausgangshoehe
BLURS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]                  # Gauss-Sigma in Pixeln
JPEGS = [95, 75, 50, 30, 15]                            # JPEG-Qualitaet
QUALITY_BASES = [None, 20]                              # Unschaerfe/JPEG bei Original bzw. 20 px

MRTMD_RES = ["2160p", "1440p", "1080p", "720p", "480p", "360p"]


def log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def norm_plate(s: str) -> str:
    """Vergleichsform: ohne Bindestrich/Leerzeichen, Grossbuchstaben."""
    return (s or "").replace("-", "").replace(" ", "").replace("_", "").upper()


def norm_lenient(s: str) -> str:
    """Wie norm_plate, zusaetzlich O=0 und I=1 (bei kleinen Zeichen kaum unterscheidbar)."""
    return norm_plate(s).replace("O", "0").replace("I", "1")


# ===========================================================================
# Hilfsfunktionen: Boxen, Bildveraenderungen, Lesen
# ===========================================================================

def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def scale_xy(img, sx: float, sy: float):
    """Bild um sx (Breite) und sy (Hoehe) verkleinern. INTER_AREA mittelt die
    Pixel wie ein Kamerasensor mit groesseren Pixeln."""
    H, W = img.shape[:2]
    return cv2.resize(img, (max(8, int(round(W * sx))), max(8, int(round(H * sy)))), interpolation=cv2.INTER_AREA)


def scale_boxes(boxes, sx, sy):
    return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (x1, y1, x2, y2) in boxes]


def jpeg(img, q: int):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def read_plate(clf, img, box):
    """Genau wie Pipeline.read_plate in analyze_video.py (mit --seg-candidates 1):
    Rand zugeben, Ausschnitt nehmen, unter MIN_PLATE_PX nicht lesen,
    segmentieren, CNN, deutsche Dekodierung."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(W, x2 + mx), min(H, y2 + my)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < MIN_PLATE_PX or crop.shape[1] < 2 * MIN_PLATE_PX:
        return PlateReading("", 0.0, False, "")
    chars, cboxes, _ = segment_characters(crop, None, max_candidates=1)
    return decode_german_plate(clf.predict_proba(chars), clf.classes, gap_positions(cboxes))


def load_test_split(quick: bool):
    """Testbilder + Kennzeichen-Boxen (Pixel) + ggf. Klartext laden."""
    gt_text = {}
    if GT_CSV.exists():
        with open(GT_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                gt_text[r["datei"]] = (int(r["index"]), r["text"])
    items = []
    for img_path in sorted((PLATE_DS / "images" / "test").glob("*.jpg")):
        lbl = PLATE_DS / "labels" / "test" / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) >= 5:
                cx, cy, bw, bh = (float(t) for t in v[1:5])
                boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        if not boxes:
            continue
        if img_path.name in gt_text:
            ref, text = gt_text[img_path.name]
        else:
            ref = int(np.argmax([b[3] - b[1] for b in boxes]))   # groesstes Kennzeichen
            text = None
        items.append({"name": img_path.name, "img": img, "boxes": boxes, "ref": ref, "text": text})
    if quick:
        with_text = [it for it in items if it["text"]]
        items = with_text[:4] + [it for it in items if not it["text"]][:8]
    return items


# ===========================================================================
# TEST A: Stufe 1 auf VisDrone test-dev
# ===========================================================================

def test_stage1(out_dir: Path, quick: bool) -> dict:
    from ultralytics import YOLO
    import prepare_visdrone as pv

    ds = ROOT / "Daten" / "visdrone_testdev_yolo"
    if not (ds / "data.yaml").exists():
        log("Bereite VisDrone test-dev im YOLO-Format vor (einmalig) ...")
        work = ROOT / "Daten" / "_visdrone_work_test"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        pv.reset_output(ds)
        src = pv.resolve_extracted_dir(ROOT / "Daten", "VisDrone2019-DET-test-dev.zip", "VisDrone2019-DET-test-dev", work)
        stats = {"train": 0, "val": 0, "test": 0}
        pv.process_split(src, "test", ds, stats)
        shutil.rmtree(work, ignore_errors=True)
        lines = [f"path: {ds.as_posix()}", "train: images/test", "val: images/test", "test: images/test",
                 f"nc: {pv.NUM_CLASSES}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(pv.CLASS_NAMES)]
        (ds / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"  {stats['test']} Testbilder vorbereitet.")

    model = YOLO(str(STAGE1_MODEL))
    m = model.val(data=str(ds / "data.yaml"), split="test", imgsz=640, batch=16, device="cpu",
                  project=str(out_dir), name="stufe1_visdrone_test", plots=True, verbose=False)
    res = {"bilder": int(m.seen) if hasattr(m, "seen") else None,
           "precision": float(m.box.mp), "recall": float(m.box.mr),
           "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "klassen": {}}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        res["klassen"][m.names[int(c)]] = {"precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50_95": float(ap)}
    return res


# ===========================================================================
# TEST B: Stufe 2 auf dem Kennzeichen-Testsplit (alt vs. neu)
# ===========================================================================

def test_stage2(out_dir: Path) -> dict:
    from ultralytics import YOLO
    res = {}
    for label, path in (("neu_kaggle_94_epochen", STAGE2_MODEL), ("alt_cpu_6_epochen", STAGE2_OLD)):
        if not path.exists():
            continue
        m = YOLO(str(path)).val(data=str(PLATE_DS / "data.yaml"), split="test", imgsz=640, batch=16,
                                device="cpu", project=str(out_dir), name=f"stufe2_test_{label}",
                                plots=True, verbose=False)
        res[label] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                      "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
        log(f"  Stufe 2 {label}: P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    return res


# ===========================================================================
# TEST C: Simulation der Drohnenbedingungen
# ===========================================================================

def build_conditions(item):
    """Erzeugt alle Bedingungen fuer ein Testbild.
    Rueckgabe: Liste (gruppe, stufe, basis, bild, boxen)."""
    img, boxes = item["img"], item["boxes"]
    h0 = boxes[item["ref"]][3] - boxes[item["ref"]][1]
    conds = []

    def at_height(target):
        if target is None:
            return img, boxes
        s = target / h0
        if s >= 0.999:
            return None, None          # groesser als im Original -> nicht simulierbar
        return scale_xy(img, s, s), scale_boxes(boxes, s, s)

    # 1) Aufloesung
    for t in RES_LEVELS:
        im, bx = at_height(t)
        if im is not None:
            conds.append(("aufloesung", "original" if t is None else t, "-", im, bx))
    # 2) Blickwinkel
    for base in ANGLE_BASES:
        im0, bx0 = at_height(base)
        if im0 is None:
            continue
        for a in ANGLES:
            c = float(np.cos(np.radians(a)))
            conds.append(("blickwinkel", a, "original" if base is None else base,
                          scale_xy(im0, 1.0, c), scale_boxes(bx0, 1.0, c)))
    # 3) Unschaerfe und 4) JPEG
    for base in QUALITY_BASES:
        im0, bx0 = at_height(base)
        if im0 is None:
            continue
        for s in BLURS:
            conds.append(("unschaerfe", s, "original" if base is None else base,
                          cv2.GaussianBlur(im0, (0, 0), s) if s > 0 else im0, bx0))
        for q in JPEGS:
            conds.append(("jpeg", q, "original" if base is None else base, jpeg(im0, q), bx0))
    return conds


def test_simulation(out_dir: Path, items, plate_model, clf) -> dict:
    rows = []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        for group, level, base, im, bx in build_conditions(item):
            ref_box = bx[item["ref"]]
            # --- Stufe 2 ---
            pr = plate_model.predict(im, imgsz=CROP_IMGSZ, conf=PLATE_CONF, verbose=False)[0]
            dets = pr.boxes.xyxy.cpu().numpy().tolist()
            confs = pr.boxes.conf.cpu().numpy().tolist()
            ious = [iou(d, ref_box) for d in dets]
            k = int(np.argmax(ious)) if ious else -1
            found = k >= 0 and ious[k] >= IOU_MATCH
            fp = sum(1 for d in dets if max(iou(d, g) for g in bx) < IOU_MATCH)
            row = {"bild": item["name"], "gruppe": group, "stufe": level, "basis": base,
                   "kz_hoehe_px": round(ref_box[3] - ref_box[1], 1),
                   "gefunden": int(found), "iou": round(ious[k], 3) if k >= 0 else 0.0,
                   "det_konf": round(confs[k], 3) if k >= 0 else 0.0, "fehlalarme": fp}
            # --- Stufe 3 (nur Bilder mit Klartext) ---
            if item["text"]:
                truth = norm_plate(item["text"])
                rd_gt = read_plate(clf, im, ref_box)                       # OCR mit perfekter Box
                rd_e2e = read_plate(clf, im, dets[k]) if found else PlateReading("", 0.0, False, "")
                for tag, rd in (("ocr", rd_gt), ("e2e", rd_e2e)):
                    pred = norm_plate(rd.text)
                    row[f"{tag}_gelesen"] = rd.text
                    row[f"{tag}_exakt"] = int(pred == truth)
                    row[f"{tag}_fehler"] = levenshtein(pred, truth)
                row["wahr"] = item["text"]
                row["zeichen"] = len(truth)
            rows.append(row)
        if n % 10 == 0 or n == len(items):
            log(f"  Simulation: {n}/{len(items)} Bilder ({time.time() - t0:.0f} s)")

    # Detail-CSV
    order = ["bild", "gruppe", "stufe", "basis", "kz_hoehe_px", "gefunden", "iou", "det_konf", "fehlalarme",
             "wahr", "zeichen", "ocr_gelesen", "ocr_exakt", "ocr_fehler", "e2e_gelesen", "e2e_exakt", "e2e_fehler"]
    keys = [k for k in order if any(k in r for r in rows)]
    with open(out_dir / "simulation_details.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        wr.writeheader()
        wr.writerows(rows)

    return summarize_simulation(rows)


FIXED_MIN_PX = 20   # feste Bildauswahl: nur Kennzeichen, die im Original mind. so hoch sind


def summarize_simulation(rows) -> dict:
    """Fasst die Einzelmessungen pro (gruppe, basis, stufe) zusammen.

    Zusaetzlich Gruppe "aufloesung_fest": dieselbe Aufloesungsreihe, aber nur
    mit den Bildern, deren Kennzeichen im Original >= FIXED_MIN_PX hoch ist.
    Dadurch stehen in JEDER Stufe dieselben Bilder -> die Stufen sind direkt
    vergleichbar (in der normalen Reihe fallen bei grossen Zielhoehen die
    Bilder mit kleineren Kennzeichen heraus)."""
    num = ("gefunden", "fehlalarme", "ocr_exakt", "ocr_fehler", "e2e_exakt", "e2e_fehler", "zeichen")
    rows = [dict(r) for r in rows]
    for r in rows:                           # Werte aus der CSV sind Text -> Zahlen
        for k in num:
            if k in r and r[k] not in ("", None):
                r[k] = int(float(r[k]))
            elif k in r:
                del r[k]
        r["kz_hoehe_px"] = float(r["kz_hoehe_px"])
    orig_h = {r["bild"]: r["kz_hoehe_px"] for r in rows if r["gruppe"] == "aufloesung" and str(r["stufe"]) == "original"}
    fixed = []
    for r in rows:
        if r["gruppe"] == "aufloesung" and orig_h.get(r["bild"], 0) >= FIXED_MIN_PX:
            st_ = str(r["stufe"])
            if st_ == "original" or float(st_) <= FIXED_MIN_PX:
                fixed.append(dict(r, gruppe="aufloesung_fest"))

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows + fixed:
        key = (r["gruppe"], str(r["basis"]), str(r["stufe"]))
        a = agg[key]
        a["gefunden"].append(r["gefunden"])
        a["fehlalarme"].append(r["fehlalarme"])
        a["kz_hoehe_px"].append(r["kz_hoehe_px"])
        if "ocr_exakt" in r:
            for tag in ("ocr", "e2e"):
                a[f"{tag}_exakt"].append(r[f"{tag}_exakt"])
                a[f"{tag}_fehler"].append(r[f"{tag}_fehler"])
            a["zeichen"].append(r["zeichen"])
    summary = []
    for (g, b, s), a in agg.items():
        e = {"gruppe": g, "basis": b, "stufe": s, "n_detektion": len(a["gefunden"]),
             "kz_hoehe_median_px": float(np.median(a["kz_hoehe_px"])),
             "recall_stufe2": float(np.mean(a["gefunden"])),
             "fehlalarme_pro_bild": float(np.mean(a["fehlalarme"]))}
        if a["zeichen"]:
            nz = sum(a["zeichen"])
            e.update({"n_ocr": len(a["zeichen"]),
                      "ocr_exakt": float(np.mean(a["ocr_exakt"])), "ocr_cer": sum(a["ocr_fehler"]) / nz,
                      "e2e_exakt": float(np.mean(a["e2e_exakt"])), "e2e_cer": sum(a["e2e_fehler"]) / nz})
        summary.append(e)
    return {"zeilen": summary}


# ===========================================================================
# TEST D: MRTMD - echte Kennzeichen in mehreren Aufloesungen
# ===========================================================================

def test_mrtmd(out_dir: Path, clf, n_max: int) -> dict:
    if not MRTMD_ZIP.exists():
        log("  MRTMD-main.zip nicht gefunden -> uebersprungen")
        return {}
    z = zipfile.ZipFile(MRTMD_ZIP)
    names = z.namelist()
    per_res = {}
    for res in MRTMD_RES:
        g = json.loads(z.read(f"MRTMD-main/NPR/{res}/number_plate_coco_{res}.json"))
        files = {Path(x).name: x for x in names if f"/NPR/{res}/images/" in x and x.lower().endswith(".jpg")}
        imgs = {im["id"]: im for im in g["images"]}
        d = {}
        for a in g["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None or im["file_name"] not in files:
                continue
            key = str(a["image_id"]).replace(f"_{res}", "")      # gleiche ID ueber alle Aufloesungen
            d[key] = (files[im["file_name"]], a.get("value", ""), a["bbox"])
        per_res[res] = d
    common = sorted(set.intersection(*[set(d) for d in per_res.values()]))
    random.Random(42).shuffle(common)
    common = common[:n_max]
    log(f"  MRTMD: {len(common)} Kennzeichen, je in {len(MRTMD_RES)} Aufloesungen")

    rows = []
    for res in MRTMD_RES:
        for key in common:
            path, value, bbox = per_res[res][key]
            img = cv2.imdecode(np.frombuffer(z.read(path), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h = img.shape[0]
            if h < MIN_PLATE_PX or img.shape[1] < 2 * MIN_PLATE_PX:
                rd = PlateReading("", 0.0, False, "")
            else:
                chars, cboxes, _ = segment_characters(img, None, max_candidates=1)
                rd = decode_german_plate(clf.predict_proba(chars), clf.classes, gap_positions(cboxes))
            truth = norm_lenient(value)
            raw = norm_lenient(rd.raw)
            rows.append({"aufloesung": res, "id": key, "wahr": value, "roh_gelesen": rd.raw,
                         "kz_hoehe_px": h, "kz_breite_px": img.shape[1],
                         "bewertet": int(mrtmd_usable(value, img.shape[1], h)),
                         "exakt": int(raw == truth), "fehler": levenshtein(raw, truth), "zeichen": len(truth)})
    with open(out_dir / "mrtmd_details.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        wr.writeheader()
        wr.writerows(rows)
    return summarize_mrtmd(rows)


def mrtmd_usable(value: str, w: int, h: int) -> bool:
    """Welche MRTMD-Kennzeichen werden bewertet?
    - "NRNP" ist im Datensatz die Markierung fuer ein NICHT lesbares
      Kennzeichen (kein Klartext vorhanden) -> nicht bewertbar.
    - Zweizeilige/quadratische Kennzeichen (Breite/Hoehe < 2,5, z.B. Motorrad)
      kann die Segmentierung (eine Textzeile) grundsaetzlich nicht -> getrennt
      ausgewiesen, nicht mitbewertet. Deutsche Kennzeichen im Fokus der
      Arbeit sind einzeilig."""
    return value.strip().upper() != "NRNP" and w / max(1, h) >= 2.5


def summarize_mrtmd(rows) -> dict:
    rows = [dict(r) for r in rows]
    for r in rows:
        for k in ("kz_hoehe_px", "kz_breite_px", "exakt", "fehler", "zeichen"):
            r[k] = int(float(r[k]))
        r["bewertet"] = int(mrtmd_usable(r["wahr"], r["kz_breite_px"], r["kz_hoehe_px"]))
    n_all = len(rows)
    n_nrnp = sum(1 for r in rows if r["wahr"].strip().upper() == "NRNP")
    ok = [r for r in rows if r["bewertet"]]
    summary = []
    for res in MRTMD_RES:
        rr = [r for r in ok if r["aufloesung"] == res]
        if not rr:
            continue
        nz = sum(r["zeichen"] for r in rr)
        summary.append({"aufloesung": res, "n": len(rr),
                        "kz_hoehe_median_px": float(np.median([r["kz_hoehe_px"] for r in rr])),
                        "anteil_unter_10px": float(np.mean([r["kz_hoehe_px"] < MIN_PLATE_PX for r in rr])),
                        "exakt": float(np.mean([r["exakt"] for r in rr])),
                        "max_1_fehler": float(np.mean([r["fehler"] <= 1 for r in rr])),
                        "cer": sum(r["fehler"] for r in rr) / max(1, nz)})
    bins = [(0, 10), (10, 14), (14, 18), (18, 22), (22, 30), (30, 999)]
    by_h = []
    for lo, hi in bins:
        rr = [r for r in ok if lo <= r["kz_hoehe_px"] < hi]
        if rr:
            nz = sum(r["zeichen"] for r in rr)
            by_h.append({"hoehe_px": f"{lo}-{hi - 1 if hi < 999 else ''}", "n": len(rr),
                         "exakt": float(np.mean([r["exakt"] for r in rr])),
                         "max_1_fehler": float(np.mean([r["fehler"] <= 1 for r in rr])),
                         "cer": sum(r["fehler"] for r in rr) / max(1, nz)})
    return {"pro_aufloesung": summary, "pro_hoehe": by_h,
            "lesungen_gesamt": n_all, "davon_nrnp": n_nrnp,
            "davon_zweizeilig": n_all - n_nrnp - len(ok), "bewertet": len(ok)}


# ===========================================================================
# TEST E: Rechenzeit
# ===========================================================================

def test_speed(items, plate_model, vehicle_model, clf) -> dict:
    def timeit(fn, n=10):
        fn()                                   # Aufwaermen
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return 1000 * (time.perf_counter() - t) / n

    img = items[0]["img"]
    frame4k = cv2.resize(img, (3840, 2160))
    res = {
        "stufe2_fahrzeugausschnitt_640_ms": timeit(lambda: plate_model.predict(img, imgsz=640, conf=PLATE_CONF, verbose=False)),
        "stufe2_4k_bild_1280_ms": timeit(lambda: plate_model.predict(frame4k, imgsz=1280, conf=PLATE_CONF, verbose=False), 5),
        "stufe3_ein_kennzeichen_ms": timeit(lambda: read_plate(clf, img, items[0]["boxes"][items[0]["ref"]]), 20),
    }
    if vehicle_model is not None:
        res["stufe1_4k_bild_1280_ms"] = timeit(lambda: vehicle_model.predict(frame4k, imgsz=1280, verbose=False), 5)
    return res


# ===========================================================================
# Bericht + Diagramme
# ===========================================================================

def pct(x):
    return "-" if x is None else f"{100 * x:.0f} %"


def write_report(out_dir: Path, R: dict) -> None:
    L = [f"# Pipeline-Test vom {R['datum']}", "",
         f"Modelle: Stufe 1 `{R['modelle']['stufe1']}`, Stufe 2 `{R['modelle']['stufe2']}`, Stufe 3 `{R['modelle']['stufe3']}`", ""]
    if R.get("stufe1"):
        s = R["stufe1"]
        L += ["## A Stufe 1 – VisDrone test-dev (Drohnenbilder)", "",
              "| | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|",
              f"| alle Klassen | {s['precision']:.3f} | {s['recall']:.3f} | {s['mAP50']:.3f} | {s['mAP50_95']:.3f} |"]
        for k, v in s["klassen"].items():
            L.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |")
        L.append("")
    if R.get("stufe2"):
        L += ["## B Stufe 2 – Kennzeichen-Testsplit (132 Bodenaufnahmen)", "",
              "| Modell | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
        for k, v in R["stufe2"].items():
            L.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |")
        L.append("")
    if R.get("simulation"):
        rows = R["simulation"]["zeilen"]
        titles = {"aufloesung_fest": f"Aufloesung, feste Bildauswahl (nur Kennzeichen >= {FIXED_MIN_PX} px im Original, in jeder Stufe dieselben Bilder)",
                  "aufloesung": "Aufloesung, alle Bilder (pro Stufe nur Bilder, deren Kennzeichen groesser als die Zielhoehe ist)", "blickwinkel": "Blickwinkel (Grad ueber der Horizontalen)",
                  "unschaerfe": "Unschaerfe (Gauss-Sigma in px)", "jpeg": "JPEG-Qualitaet"}
        L += ["## C Simulation der Drohnenbedingungen", "",
              "Recall Stufe 2: Kennzeichen gefunden (IoU >= 0,5). OCR: Stufe 3 mit perfekter Box. "
              "Ende-zu-Ende: Stufe 2 + 3. Exakt = ganzes Kennzeichen richtig, CER = Zeichenfehlerrate.", ""]
        for g, title in titles.items():
            for b in sorted({r["basis"] for r in rows if r["gruppe"] == g}, key=lambda v: (v != "original", v)):
                sel = [r for r in rows if r["gruppe"] == g and r["basis"] == b]
                if not sel:
                    continue

                def sk(r):
                    return -1e9 if r["stufe"] == "original" else (-float(r["stufe"]) if g in ("aufloesung", "aufloesung_fest", "jpeg") else float(r["stufe"]))
                sel.sort(key=sk)
                head = title + ("" if b == "-" else f", Ausgangshoehe {b}" + (" px" if b != "original" else ""))
                L += [f"### {head}", "",
                      "| Stufe | KZ-Hoehe (Median px) | n | Recall Stufe 2 | Fehlalarme/Bild | n OCR | OCR exakt | OCR CER | E2E exakt | E2E CER |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
                for r in sel:
                    L.append(f"| {r['stufe']} | {r['kz_hoehe_median_px']:.0f} | {r['n_detektion']} | {pct(r['recall_stufe2'])} | "
                             f"{r['fehlalarme_pro_bild']:.2f} | {r.get('n_ocr', 0)} | {pct(r.get('ocr_exakt'))} | {pct(r.get('ocr_cer'))} | "
                             f"{pct(r.get('e2e_exakt'))} | {pct(r.get('e2e_cer'))} |")
                L.append("")
    if R.get("mrtmd"):
        m = R["mrtmd"]
        L += ["## D MRTMD – echte Kennzeichen (erhoehte Verkehrskamera), nur Stufe 3", "",
              "Auslaendische Kennzeichen: Rohlesung ohne deutsche Formatpruefung, O=0 und I=1 gleichgesetzt. "
              f"Von {m.get('lesungen_gesamt', '?')} Lesungen ausgeschlossen: {m.get('davon_nrnp', '?')} als unlesbar markierte (NRNP) "
              f"und {m.get('davon_zweizeilig', '?')} zweizeilige/quadratische Kennzeichen; bewertet: {m.get('bewertet', '?')}.", "",
              "| Aufloesung | n | KZ-Hoehe (Median px) | Anteil < 10 px | exakt | max. 1 Fehler | CER |", "|---|---|---|---|---|---|---|"]
        for r in m["pro_aufloesung"]:
            L.append(f"| {r['aufloesung']} | {r['n']} | {r['kz_hoehe_median_px']:.0f} | {pct(r['anteil_unter_10px'])} | {pct(r['exakt'])} | {pct(r.get('max_1_fehler'))} | {pct(r['cer'])} |")
        L += ["", "| KZ-Hoehe px | n | exakt | max. 1 Fehler | CER |", "|---|---|---|---|---|"]
        for r in m["pro_hoehe"]:
            L.append(f"| {r['hoehe_px']} | {r['n']} | {pct(r['exakt'])} | {pct(r.get('max_1_fehler'))} | {pct(r['cer'])} |")
        L.append("")
    if R.get("speed"):
        L += ["## E Rechenzeit (CPU dieses Rechners)", "", "| Schritt | ms |", "|---|---|"]
        for k, v in R["speed"].items():
            L.append(f"| {k} | {v:.0f} |")
        L.append("")
    if R.get("fehler"):
        L += ["## Fehler waehrend des Tests", ""] + [f"- {e}" for e in R["fehler"]]
    (out_dir / "bericht.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_plots(out_dir: Path, R: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sim = (R.get("simulation") or {}).get("zeilen", [])
    res = [r for r in sim if r["gruppe"] == "aufloesung_fest"] or [r for r in sim if r["gruppe"] == "aufloesung"]
    if res:
        # x = tatsaechliche (mediane) Kennzeichenhoehe, damit auch "original" einen Platz hat
        res.sort(key=lambda r: r["kz_hoehe_median_px"])
        x = [r["kz_hoehe_median_px"] for r in res]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, [100 * r["recall_stufe2"] for r in res], "o-", label="Stufe 2: Kennzeichen gefunden")
        if any("e2e_cer" in r for r in res):
            ax.plot(x, [100 * (1 - r["e2e_cer"]) if "e2e_cer" in r else np.nan for r in res], "s-",
                    label="Ende-zu-Ende: Zeichen richtig (1 - CER)")
            ax.plot(x, [100 * r.get("e2e_exakt", np.nan) for r in res], "^--", label="Ende-zu-Ende: ganzes Kennzeichen exakt")
        ax.axvline(MIN_PLATE_PX, color="grey", ls=":", lw=1)
        ax.set_xlabel("Kennzeichenhoehe im Bild [px]"); ax.set_ylabel("Anteil [%]"); ax.set_ylim(-3, 103)
        ax.set_title(f"Einfluss der Aufloesung (simuliert, Kennzeichen >= {FIXED_MIN_PX} px im Original)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "aufloesung.png", dpi=150); plt.close(fig)
    for g, xl in (("blickwinkel", "Blickwinkel ueber der Horizontalen [Grad]"), ("unschaerfe", "Gauss-Sigma [px]"), ("jpeg", "JPEG-Qualitaet")):
        sel = [r for r in sim if r["gruppe"] == g]
        if not sel:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for b in sorted({r["basis"] for r in sel}):
            rr = sorted([r for r in sel if r["basis"] == b], key=lambda r: float(r["stufe"]))
            x = [float(r["stufe"]) for r in rr]
            lab = "Original" if b == "original" else f"{b} px"
            ax.plot(x, [100 * r["recall_stufe2"] for r in rr], "o-", label=f"Stufe 2 gefunden ({lab})")
            if any("e2e_exakt" in r for r in rr):
                ax.plot(x, [100 * r.get("e2e_exakt", np.nan) for r in rr], "^--", label=f"Ende-zu-Ende exakt ({lab})")
        if g == "jpeg":
            ax.invert_xaxis()
        ax.set_xlabel(xl); ax.set_ylabel("Anteil [%]"); ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / f"{g}.png", dpi=150); plt.close(fig)
    m = (R.get("mrtmd") or {}).get("pro_aufloesung", [])
    if m:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = [r["kz_hoehe_median_px"] for r in m]
        ax.plot(x, [100 * r["cer"] for r in m], "o-", label="CER")
        ax.plot(x, [100 * r["exakt"] for r in m], "s-", label="exakt gelesen")
        for r in m:
            ax.annotate(r["aufloesung"], (r["kz_hoehe_median_px"], 100 * r["cer"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Kennzeichenhoehe (Median) [px]"); ax.set_ylabel("[%]"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("MRTMD: echte Kennzeichen, Stufe 3")
        fig.tight_layout(); fig.savefig(out_dir / "mrtmd.png", dpi=150); plt.close(fig)


# ===========================================================================
# Hauptprogramm
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=["stufe1", "stufe2", "simulation", "mrtmd", "speed"],
                    choices=["stufe1", "stufe2", "simulation", "mrtmd", "speed"])
    ap.add_argument("--mrtmd-n", type=int, default=300, help="Anzahl MRTMD-Kennzeichen (je Aufloesung)")
    ap.add_argument("--quick", action="store_true", help="Kurztest mit wenigen Bildern")
    ap.add_argument("--out-dir", default=str(ROOT / "Tests"))
    ap.add_argument("--neu-auswerten", default=None, metavar="TESTORDNER",
                    help="keine neue Messung: Tabellen/Diagramme eines vorhandenen Testordners aus dessen CSV-Dateien neu berechnen")
    args = ap.parse_args()

    if args.neu_auswerten:
        d = Path(args.neu_auswerten)
        R = json.loads((d / "ergebnisse.json").read_text(encoding="utf-8"))
        if (d / "simulation_details.csv").exists():
            with open(d / "simulation_details.csv", encoding="utf-8-sig") as f:
                R["simulation"] = summarize_simulation(list(csv.DictReader(f, delimiter=";")))
        if (d / "mrtmd_details.csv").exists():
            with open(d / "mrtmd_details.csv", encoding="utf-8-sig") as f:
                R["mrtmd"] = summarize_mrtmd(list(csv.DictReader(f, delimiter=";")))
        (d / "ergebnisse.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(d, R)
        write_plots(d, R)
        log(f"Neu ausgewertet: {d / 'bericht.md'}")
        return

    from ultralytics import YOLO

    out_dir = Path(args.out_dir) / f"pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}{'_quick' if args.quick else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    R = {"datum": datetime.now().strftime("%d.%m.%Y %H:%M"), "fehler": [],
         "modelle": {"stufe1": str(STAGE1_MODEL), "stufe2": str(STAGE2_MODEL), "stufe3": str(STAGE3_MODEL)},
         "einstellungen": {"plate_conf": PLATE_CONF, "crop_imgsz": CROP_IMGSZ, "min_plate_px": MIN_PLATE_PX, "iou": IOU_MATCH}}
    log(f"Ausgabe: {out_dir}")

    def run(name, fn):
        log(f"=== Test {name} ===")
        t = time.time()
        try:
            R[name] = fn()
            log(f"=== Test {name} fertig ({time.time() - t:.0f} s) ===")
        except Exception as e:
            R["fehler"].append(f"{name}: {e!r}")
            traceback.print_exc()
        (out_dir / "ergebnisse.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(out_dir, R)

    clf = CharClassifier(STAGE3_MODEL)
    plate_model = YOLO(str(STAGE2_MODEL))
    vehicle_model = YOLO(str(STAGE1_MODEL)) if STAGE1_MODEL.exists() else None
    items = load_test_split(args.quick) if ({"simulation", "speed"} & set(args.tests)) else []
    if items:
        log(f"Testbilder: {len(items)}, davon mit Klartext: {sum(1 for i in items if i['text'])}")

    if "stufe2" in args.tests:
        run("stufe2", lambda: test_stage2(out_dir))
    if "simulation" in args.tests:
        run("simulation", lambda: test_simulation(out_dir, items, plate_model, clf))
    if "mrtmd" in args.tests:
        run("mrtmd", lambda: test_mrtmd(out_dir, clf, 30 if args.quick else args.mrtmd_n))
    if "speed" in args.tests:
        run("speed", lambda: test_speed(items, plate_model, vehicle_model, clf))
    if "stufe1" in args.tests and vehicle_model is not None and not args.quick:
        run("stufe1", lambda: test_stage1(out_dir, args.quick))

    write_plots(out_dir, R)
    log(f"FERTIG. Bericht: {out_dir / 'bericht.md'}")


if __name__ == "__main__":
    main()
