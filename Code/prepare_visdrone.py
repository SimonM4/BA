#!/usr/bin/env python3
"""
prepare_visdrone.py

Datenvorbereitung fuer STUFE 1 (Fahrzeug-/Objekterkennung aus der Luft).

Wandelt die offiziellen VisDrone2019-DET-Zips (train/val/test-dev) in ein
YOLO-Trainingsformat um. VisDrone liefert die Annotationen NICHT im YOLO-
Format, sondern als eigene CSV-Zeilen pro Bild:

    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Wichtige Besonderheit (steht so nicht offensichtlich in der Doku, aber ist
Teil des offiziellen VisDrone-Formats): das Feld "score" ist in den
Ground-Truth-Dateien KEIN Konfidenzwert, sondern ein Flag: score=0 bedeutet
"ignorierte Region" (z.B. dichte Menschenmengen, die nicht einzeln annotiert
wurden) und wird beim Training uebersprungen. Diese Zeilen behandeln wir also
NICHT als eigene Klasse, sondern verwerfen sie.

Die Klassen-IDs in der Datei sind 1-indiziert (1=pedestrian ... 10=motor,
zusaetzlich 11="others"). Fuer YOLO brauchen wir 0-indizierte IDs, also
ziehen wir 1 ab. Diese Konvertierungslogik folgt exakt dem offiziellen,
seit Jahren produktiv genutzten "visdrone2yolo"-Ansatz aus dem
Ultralytics-Repo (siehe ultralytics/cfg/datasets/VisDrone.yaml), damit wir
hier keine eigene, ungetestete Variante erfinden.

Quellen (muessen als Zips in --data-dir liegen):
  - VisDrone2019-DET-train.zip
  - VisDrone2019-DET-val.zip
  - VisDrone2019-DET-test-dev.zip

Bewusst NICHT verwendet:
  - VisDrone2019-DET-test-challenge.zip -> keine Ground-Truth-Annotationen
    verfuegbar, damit weder Training noch Evaluation moeglich.

Output (unter --out-dir, Default "visdrone_yolo"):
  visdrone_yolo/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
    data.yaml   (10 Klassen: pedestrian, people, bicycle, car, van, truck,
                 tricycle, awning-tricycle, bus, motor)

Aufruf:
  python prepare_visdrone.py --data-dir "C:\\Users\\smehl\\OneDrive\\Desktop\\BA\\Daten"
"""

import argparse
import shutil
import time
import zipfile
from pathlib import Path
from PIL import Image

# Offizielle VisDrone-DET-Klassen (0-indiziert, wie von Ultralytics benutzt)
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]
NUM_CLASSES = len(CLASS_NAMES)

# Zip-Dateiname -> (Name des Ordners nach dem Entpacken, Zielsplit bei uns)
SPLITS = {
    "VisDrone2019-DET-train.zip": ("VisDrone2019-DET-train", "train"),
    "VisDrone2019-DET-val.zip": ("VisDrone2019-DET-val", "val"),
    "VisDrone2019-DET-test-dev.zip": ("VisDrone2019-DET-test-dev", "test"),
}


def log(msg: str) -> None:
    print(f"[visdrone] {msg}")


def reset_output(out_dir: Path) -> None:
    """Wie beim Kennzeichen-Merge-Script: mit Retry, falls OneDrive den
    alten Ordner gerade synchronisiert und kurzzeitig sperrt."""
    if out_dir.exists():
        last_err = None
        for attempt in range(6):
            try:
                shutil.rmtree(out_dir)
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                wait = 2 * (attempt + 1)
                log(f"  Ausgabeordner gesperrt (vermutlich OneDrive-Sync), warte {wait}s ...")
                time.sleep(wait)
        if last_err is not None:
            raise RuntimeError(
                f"Konnte {out_dir} nicht loeschen (vermutlich OneDrive-Sync). "
                f"OneDrive kurz pausieren oder --out-dir ausserhalb von OneDrive waehlen."
            ) from last_err
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_box(img_w: float, img_h: float, x: float, y: float, w: float, h: float):
    """VisDrone-Box (links, oben, Breite, Hoehe in Pixeln) -> YOLO
    (x_center, y_center, w, h), alles normiert auf [0, 1]."""
    dw, dh = 1.0 / img_w, 1.0 / img_h
    x_center = (x + w / 2.0) * dw
    y_center = (y + h / 2.0) * dh
    return x_center, y_center, w * dw, h * dh


def convert_annotation_file(ann_path: Path, img_path: Path) -> str:
    """Liest eine VisDrone-Annotationsdatei (eine Zeile pro Objekt) und
    gibt den fertigen YOLO-Label-Text zurueck."""
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    lines = []
    for raw_line in ann_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip().rstrip(",")
        if not raw_line:
            continue
        row = raw_line.split(",")
        if len(row) < 6:
            continue  # unvollstaendige/kaputte Zeile, ueberspringen

        score_flag = row[4]
        if score_flag == "0":
            # "ignorierte Region" laut VisDrone-Spezifikation, nicht fuers
            # Training verwenden (siehe Docstring oben).
            continue

        x, y, w, h = map(float, row[:4])
        if w <= 0 or h <= 0:
            continue

        cls = int(row[5]) - 1  # VisDrone ist 1-indiziert -> YOLO 0-indiziert
        if not (0 <= cls < NUM_CLASSES):
            # Klasse 0 (="ignored regions") oder 11 (="others") vor dem
            # Verschieben, bzw. ungueltige Werte -> bewusst verwerfen statt
            # eine falsche/undefinierte Klassen-ID zu schreiben.
            continue

        xc, yc, bw, bh = convert_box(img_w, img_h, x, y, w, h)
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return ("\n".join(lines) + ("\n" if lines else ""))


def resolve_extracted_dir(data_dir: Path, zip_name: str, extracted_name: str, work_dir: Path) -> Path:
    """Nutzt einen schon entpackten Ordner, falls vorhanden, sonst wird die
    Zip-Datei nach work_dir entpackt."""
    already_extracted = data_dir / extracted_name
    if already_extracted.is_dir():
        return already_extracted

    zip_path = data_dir / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_name} nicht gefunden unter {data_dir}")

    dest = work_dir / extracted_name
    dest.mkdir(parents=True, exist_ok=True)
    log(f"Entpacke {zip_name} (das kann bei ~1,5 GB einen Moment dauern) ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)

    # manche VisDrone-Zips haben eine zusaetzliche Verschachtelung
    # (VisDrone2019-DET-train/VisDrone2019-DET-train/images/...) -- falls
    # images/annotations nicht direkt da liegen, eine Ebene tiefer suchen.
    if not (dest / "images").is_dir():
        candidates = list(dest.rglob("images"))
        if candidates:
            dest = candidates[0].parent
    return dest


def process_split(src_dir: Path, out_split: str, out_dir: Path, stats: dict) -> None:
    img_dir = src_dir / "images"
    ann_dir = src_dir / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Erwarte images/ und annotations/ unter {src_dir}, gefunden: {list(src_dir.iterdir())}")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for img_path in images:
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            log(f"  WARNUNG: keine Annotation fuer {img_path.name}, ueberspringe")
            continue

        label_text = convert_annotation_file(ann_path, img_path)

        out_img = out_dir / "images" / out_split / img_path.name
        out_lbl = out_dir / "labels" / out_split / f"{img_path.stem}.txt"
        shutil.copyfile(img_path, out_img)
        out_lbl.write_text(label_text, encoding="utf-8")
        stats[out_split] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="Ordner mit den VisDrone-Zips (oder bereits entpackten Ordnern)")
    ap.add_argument("--out-dir", default="visdrone_yolo", help="Zielordner fuer den konvertierten Datensatz")
    ap.add_argument("--work-dir", default="_visdrone_work", help="Temporaerer Ordner zum Entpacken")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    reset_output(out_dir)

    stats = {"train": 0, "val": 0, "test": 0}

    for zip_name, (extracted_name, out_split) in SPLITS.items():
        try:
            src_dir = resolve_extracted_dir(data_dir, zip_name, extracted_name, work_dir)
        except FileNotFoundError as e:
            log(f"WARNUNG: {e}, ueberspringe Split '{out_split}'")
            continue
        log(f"Verarbeite {extracted_name} -> Split '{out_split}' ...")
        process_split(src_dir, out_split, out_dir, stats)

    yaml_lines = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {NUM_CLASSES}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {name}")
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    total = sum(stats.values())
    log("Fertig.")
    log(f"  train: {stats['train']}")
    log(f"  val:   {stats['val']}")
    log(f"  test:  {stats['test']}")
    log(f"  gesamt: {total}")
    log(f"  -> {out_dir}/ (mit data.yaml, {NUM_CLASSES} Klassen)")


if __name__ == "__main__":
    main()
