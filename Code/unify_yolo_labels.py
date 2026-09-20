#!/usr/bin/env python3
"""
unify_yolo_labels.py

Vereinheitlicht die Kennzeichen-Detection-Datensaetze (NICHT VisDrone, NICHT
die Zeichen-Klassifikationsdaten) zu EINEM konsistenten YOLO-Datensatz mit
genau einer Klasse "license_plate" (id 0).

Quellen (liegen alle direkt in --data-dir):
  - DatenSatz1/ ODER archive (1).zip -> bereits YOLO (images/labels, train/val)
                                         ["license-plate-dataset"]
  - DatenSatz2/ ODER archive (2).zip -> bereits YOLO (images/labels, train/val)
                                         (Open-Images-Stil, weltweite Kennzeichen,
                                         teils mehrere Kennzeichen pro Bild)
  - archive (3).zip                -> Pascal-VOC-XML (images/, annotations/Cars*.xml),
                                       wird zu YOLO konvertiert
  - german-license-plates.v7i.yolo26.zip
                                    -> Roboflow-Export, bereits YOLO
                                       (train/valid/test), deutsche Kennzeichen

Jede Quelle darf entweder ein bereits entpackter Ordner ODER eine .zip-Datei
sein -- das Script erkennt das automatisch (fuer DatenSatz1/DatenSatz2 werden
beide moeglichen Namen probiert, falls du die Zips noch nicht selbst entpackt
hast -- siehe YOLO_SOURCES unten).

Bewusst NICHT eingeschlossen:
  - VisDrone2019-DET-*.zip      -> andere Aufgabe (Objekterkennung, mehrere
                                    Klassen: car/van/bus/pedestrian/...),
                                    gehoert zur vorgelagerten Fahrzeug-
                                    Detection-Stufe, nicht zur Kennzeichen-Stufe.
  - archive (4).zip, DeutschEinzelneZahlen.zip
                                 -> Zeichen-Klassifikation (OCR-Stufe), kein
                                    Bounding-Box-Format, anderes Problem.

Output (unter --out-dir, Default "kennzeichen_detection_dataset"):
  kennzeichen_detection_dataset/
    images/{train,val,test}/<quelle>_<original-dateiname>.<ext>
    labels/{train,val,test}/<quelle>_<original-dateiname>.txt
    data.yaml

Aufruf:
  python unify_yolo_labels.py --data-dir "C:\\BA\\Daten"
"""

import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

CLASS_NAME = "license_plate"
CLASS_ID = 0
VAL_SPLIT_FOR_UNSPLIT_SOURCES = 0.15  # nur fuer Quellen ohne eigenen Split (z.B. archive(3))
RANDOM_SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- Quellen-Definition: hier anpassen, falls sich Dateinamen aendern ---
# "path" ist jetzt eine LISTE moeglicher Namen: manche Quellen liegen bei dir
# schon entpackt (DatenSatz1/DatenSatz2), manche noch als Original-Zip
# (archive (1).zip / archive (2).zip) -- resolve_source_dir() probiert die
# Kandidaten der Reihe nach und nimmt den ersten, der in --data-dir existiert.
YOLO_SOURCES = [
    {
        "path": ["DatenSatz1", "archive (1).zip"],
        "tag": "datensatz1",
        "split_map": {"train": "train", "val": "val"},
        "known_class_ids": {0},
    },
    {
        "path": ["DatenSatz2", "archive (2).zip"],
        "tag": "datensatz2",
        "split_map": {"train": "train", "val": "val"},
        "known_class_ids": {0},
    },
    {
        "path": ["german-license-plates.v7i.yolo26.zip"],
        "tag": "roboflow",
        "split_map": {"train": "train", "valid": "val", "test": "test"},
        "known_class_ids": {0},
    },
]

VOC_SOURCES = [
    {"path": ["archive (3).zip"], "tag": "kaggle3"},
]


def log(msg: str) -> None:
    print(f"[unify] {msg}")


def reset_output(out_dir: Path) -> None:
    """Loescht einen vorhandenen Ausgabeordner und legt die Split-Unterordner
    neu an. Liegt --out-dir NICHT mehr in einem Cloud-Sync-Ordner (OneDrive/
    Dropbox/...), ist das ein normaler, sofortiger Dateisystem-Vorgang --
    kein Warten/Retry mehr noetig."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _normalize_name(name: str) -> str:
    """Entfernt alles ausser Buchstaben/Zahlen und macht klein. Damit lassen
    sich Kandidaten-Namen auch bei kleinen Abweichungen (Leerzeichen,
    Klammern, Gross-/Kleinschreibung) noch zuordnen -- siehe resolve_source_dir."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_source_dir(data_dir: Path, candidates, work_dir: Path, tag: str) -> Path:
    """Gibt einen entpackten Ordner fuer die Quelle zurueck. `candidates` ist
    eine Liste moeglicher Namen (z.B. sowohl der schon entpackte Ordnername
    als auch der Name der zugehoerigen .zip-Datei) -- der erste Kandidat, der
    unter data_dir tatsaechlich existiert, wird verwendet. So funktioniert das
    Script unabhaengig davon, ob du eine Quelle schon entpackt hast oder nicht.
    Ist der gefundene Kandidat bereits ein Ordner, wird er direkt verwendet;
    ist es eine .zip-Datei, wird sie zuerst nach work_dir/<tag> entpackt.

    WICHTIG (auf Kaggle entdeckt): Kaggle behaelt hochgeladene Zips NICHT als
    Zip -- es entpackt sie beim Dataset-Upload automatisch in einen Ordner mit
    (fast) demselben Namen ohne ".zip"-Endung. Die exakten Kandidaten-Strings
    oben (die fuer die lokale Windows-Umgebung gedacht sind) passen dann nicht
    mehr woertlich. Als Fallback wird deshalb zusaetzlich jeder Ordner unter
    data_dir mit NORMALISIERTEM Namen (klein, ohne Leerzeichen/Klammern/
    Sonderzeichen) gegen die normalisierten Kandidaten (auch ohne .zip)
    verglichen -- das faengt Kaggles automatisches Entpacken robust ab, ohne
    dass man Kaggles genaues Umbenennungsschema kennen muss."""
    if isinstance(candidates, str):
        candidates = [candidates]

    src_path = None
    for rel_path in candidates:
        candidate = data_dir / rel_path
        if candidate.exists():
            src_path = candidate
            break

    if src_path is None and data_dir.is_dir():
        wanted = set()
        for rel_path in candidates:
            stem = rel_path[:-4] if rel_path.lower().endswith(".zip") else rel_path
            wanted.add(_normalize_name(stem))
        for entry in sorted(data_dir.iterdir()):
            if entry.is_dir() and _normalize_name(entry.name) in wanted:
                src_path = entry
                log(f"  (Kandidat per Namensabgleich gefunden: '{entry.name}' fuer {candidates})")
                break

    if src_path is None:
        raise FileNotFoundError(
            f"Keiner dieser Pfade (auch nicht aehnlich benannt) gefunden unter {data_dir}: {candidates}"
        )

    if src_path.is_dir():
        return src_path

    # .zip -> entpacken
    dest = work_dir / tag
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_path) as z:
        z.extractall(dest)
    return dest


# zaehlt hochlaufende, kurze IDs pro Quelle (nicht pro Original-Dateiname!),
# damit die Ausgabepfade unter Windows' MAX_PATH-Grenze (260 Zeichen) bleiben
# -- manche Originaldateinamen (z.B. in DatenSatz1) sind selbst schon > 150
# Zeichen lang.
_name_counters: dict = {}
_mapping_rows: list = []


def write_pair(img_path: Path, label_text: str, out_dir: Path, split: str, source_tag: str) -> None:
    """Kopiert ein Bild + schreibt die zugehoerige YOLO-Label-Datei unter einem
    kurzen, kollisionsfreien Dateinamen (Quelle + laufende Nummer). Der
    urspruengliche Dateiname wird nicht verwendet (kann zu lang fuer Windows-
    Pfade sein), bleibt aber ueber mapping.csv im Ausgabeordner nachvollziehbar."""
    idx = _name_counters.get(source_tag, 0) + 1
    _name_counters[source_tag] = idx
    out_name = f"{source_tag}_{idx:06d}"

    out_img = out_dir / "images" / split / f"{out_name}{img_path.suffix.lower()}"
    out_lbl = out_dir / "labels" / split / f"{out_name}.txt"

    if out_img.exists():
        raise RuntimeError(f"Dateinamen-Kollision trotz laufender Nummer: {out_img}")

    shutil.copyfile(img_path, out_img)
    out_lbl.write_text(label_text, encoding="utf-8")
    _mapping_rows.append((out_name, source_tag, split, img_path.name))


def write_mapping_csv(out_dir: Path) -> None:
    """Schreibt eine mapping.csv: neuer Dateiname -> Quelle/Split/Original-
    Dateiname, damit sich jedes Bild trotz Kurzname zurueckverfolgen laesst
    (z.B. fuer die Methodik-Beschreibung in der Arbeit)."""
    import csv

    with (out_dir / "mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "source_tag", "split", "original_filename"])
        writer.writerows(_mapping_rows)


def remap_yolo_label(label_path: Path, known_class_ids: set) -> str:
    """Liest eine bestehende YOLO-Label-Datei und schreibt alle Zeilen mit der
    einheitlichen CLASS_ID neu. Bricht laut, wenn eine unerwartete Klassen-ID
    auftaucht, statt sie still falsch zuzuordnen."""
    lines_out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        if cls_id not in known_class_ids:
            raise ValueError(
                f"Unerwartete Klassen-ID {cls_id} in {label_path} "
                f"(erwartet: {sorted(known_class_ids)}). Bitte pruefen, "
                f"ob dieser Datensatz wirklich nur Kennzeichen enthaelt."
            )
        rest = parts[1:]
        lines_out.append(" ".join([str(CLASS_ID)] + rest))
    return "\n".join(lines_out) + ("\n" if lines_out else "")


def voc_to_yolo_label(xml_path: Path) -> tuple:
    """Konvertiert eine Pascal-VOC-XML-Annotation in YOLO-Zeilen.
    Gibt (label_text, bild_dateiname) zurueck."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    img_w = float(size.findtext("width"))
    img_h = float(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        box_w = (xmax - xmin) / img_w
        box_h = (ymax - ymin) / img_h

        if box_w <= 0 or box_h <= 0:
            log(f"  WARNUNG: ungueltige Box in {xml_path.name}, ueberspringe Objekt")
            continue

        lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return ("\n".join(lines) + ("\n" if lines else "")), filename


def process_yolo_source(src_dir: Path, split_map: dict, source_tag: str, known_class_ids: set, out_dir: Path, stats: dict) -> None:
    for src_split, out_split in split_map.items():
        # zwei gaengige Layouts abdecken: "images/<split>/" (z.B. DatenSatz1/2)
        # und "<split>/images/" (z.B. Roboflow-Exporte)
        img_dirs = [d for d in src_dir.rglob(f"images/{src_split}") if d.is_dir()]
        if not img_dirs:
            img_dirs = [d for d in src_dir.rglob(f"{src_split}/images") if d.is_dir()]
        if not img_dirs:
            log(f"  kein images-Ordner fuer Split '{src_split}' in {source_tag} gefunden, ueberspringe")
            continue
        img_dir = img_dirs[0]
        if img_dir.parent.name == src_split:
            lbl_dir = img_dir.parent / "labels"
        else:
            lbl_dir = img_dir.parent.parent / "labels" / src_split

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                log(f"  WARNUNG: kein Label fuer {img_path.name} in {source_tag}, ueberspringe")
                continue
            label_text = remap_yolo_label(lbl_path, known_class_ids)
            write_pair(img_path, label_text, out_dir, out_split, source_tag)
            stats[out_split] += 1


def process_voc_source(src_dir: Path, source_tag: str, out_dir: Path, stats: dict) -> None:
    ann_dir = next(src_dir.rglob("annotations"))
    img_dir = ann_dir.parent / "images"

    xml_files = sorted(ann_dir.glob("*.xml"))
    random.Random(RANDOM_SEED).shuffle(xml_files)
    n_val = max(1, int(len(xml_files) * VAL_SPLIT_FOR_UNSPLIT_SOURCES))
    val_set = set(xml_files[:n_val])

    for xml_path in xml_files:
        label_text, img_filename = voc_to_yolo_label(xml_path)
        img_path = img_dir / img_filename
        if not img_path.exists():
            log(f"  WARNUNG: Bild {img_filename} zu {xml_path.name} nicht gefunden, ueberspringe")
            continue
        if not label_text:
            log(f"  WARNUNG: keine gueltigen Boxen in {xml_path.name}, ueberspringe")
            continue
        split = "val" if xml_path in val_set else "train"
        write_pair(img_path, label_text, out_dir, split, source_tag)
        stats[split] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="Ordner, in dem DatenSatz1/DatenSatz2 (oder archive (1)/(2).zip) und die uebrigen Zip-Dateien liegen")
    ap.add_argument("--out-dir", default="kennzeichen_detection_dataset", help="Zielordner fuer den vereinheitlichten Datensatz")
    ap.add_argument("--work-dir", default="_unify_work", help="Temporaerer Ordner zum Entpacken von Zip-Quellen")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    reset_output(out_dir)

    stats = {"train": 0, "val": 0, "test": 0}

    for src in YOLO_SOURCES:
        try:
            src_dir = resolve_source_dir(data_dir, src["path"], work_dir, src["tag"])
        except FileNotFoundError as e:
            log(f"WARNUNG: {e}, ueberspringe")
            continue
        log(f"Verarbeite {src_dir.name} (Tag: {src['tag']}) ...")
        process_yolo_source(src_dir, src["split_map"], src["tag"], src["known_class_ids"], out_dir, stats)

    for src in VOC_SOURCES:
        try:
            src_dir = resolve_source_dir(data_dir, src["path"], work_dir, src["tag"])
        except FileNotFoundError as e:
            log(f"WARNUNG: {e}, ueberspringe")
            continue
        log(f"Verarbeite {src_dir.name} (Tag: {src['tag']}) ...")
        process_voc_source(src_dir, src["tag"], out_dir, stats)

    yaml_text = (
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    write_mapping_csv(out_dir)

    if work_dir.exists():
        shutil.rmtree(work_dir)

    total = sum(stats.values())
    log("Fertig.")
    log(f"  train: {stats['train']}")
    log(f"  val:   {stats['val']}")
    log(f"  test:  {stats['test']}")
    log(f"  gesamt: {total}")
    log(f"  -> {out_dir}/ (mit data.yaml)")


if __name__ == "__main__":
    main()
