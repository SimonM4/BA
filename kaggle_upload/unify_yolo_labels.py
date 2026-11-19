#!/usr/bin/env python3
"""
unify_yolo_labels.py  --  DATENVORBEREITUNG FUER STUFE 2 (Kennzeichen finden)

===========================================================================
WAS MACHT DIESES SKRIPT?
===========================================================================
Fuer Stufe 2 gibt es nicht EINEN Kennzeichen-Datensatz, sondern vier,
und jeder ist anders aufgebaut und anders gelabelt. Dieses Skript fuehrt
sie zu EINEM einheitlichen YOLO-Datensatz zusammen, mit genau einer Klasse:
"license_plate" (Klassen-ID 0).

Warum zusammenfuehren statt nur einen Datensatz nehmen?
  Mehr und vielfaeltigere Bilder (verschiedene Laender, Kameras,
  Lichtverhaeltnisse, Abstaende) -> das Modell lernt "Kennzeichen" allgemeiner
  und klebt weniger an den Eigenheiten eines einzelnen Datensatzes.

Warum nur EINE Klasse?
  Stufe 2 soll nur sagen "hier ist ein Kennzeichen". WAS draufsteht,
  entscheidet erst Stufe 3. Eine einzige Klasse macht die Aufgabe einfacher
  und erlaubt es, Datensaetze aus verschiedenen Laendern zu mischen.

===========================================================================
DIE QUELLEN (alle in --data-dir)
===========================================================================
  Quelle                                Format           Splits     Tag im Dateinamen
  ------------------------------------  ---------------  ---------  -----------------
  DatenSatz1/  oder  archive (1).zip    YOLO             train/val  datensatz1
  DatenSatz2/  oder  archive (2).zip    YOLO (weltweit,  train/val  datensatz2
                                        teils mehrere
                                        Kennzeichen/Bild)
  german-license-plates.v7i.yolo26.zip  YOLO (Roboflow,  train/     roboflow
                                        deutsche KZ)     valid/test
  archive (3).zip                       Pascal-VOC-XML   keine      kaggle3
                                        (muss umgerech-  (wird hier
                                        net werden)      ausgelost)

Bewusst NICHT hier enthalten:
  - VisDrone   -> gehoert zu Stufe 1 (Fahrzeuge, 10 Klassen)
  - archive (4).zip, DeutschEinzelneZahlen.zip -> gehoeren zu Stufe 3
    (einzelne Zeichen, keine Boxen)

===========================================================================
WICHTIG FUER DIE METHODIK
===========================================================================
- Die vorhandenen train/val/test-Aufteilungen der Quellen werden
  UEBERNOMMEN. Nur archive (3) hat keine Aufteilung; dort werden 15 % per
  Zufall (fester Seed 42 -> reproduzierbar) zu "val".
- Einen test-Split gibt es nur aus der Roboflow-Quelle (deutsche
  Kennzeichen). Das ist praktisch, weil der Test damit auf deutsche
  Kennzeichen zielt, er ist aber entsprechend klein.
- Fast alle Bilder sind BODENAUFNAHMEN (Kamera etwa auf Kennzeichenhoehe).
  Das ist der "Domain Gap" zur Drohnenperspektive, den deine
  Forschungsfrage untersucht.
- Jedes Bild bekommt einen kurzen neuen Namen (z.B. datensatz1_000123.jpg).
  mapping.csv haelt fest, aus welcher Quelle und welcher Originaldatei es
  stammt. So bleibt jedes Bild nachvollziehbar.

Ausgabe (in --out-dir, Standard "kennzeichen_detection_dataset"):
  kennzeichen_detection_dataset/
    images/{train,val,test}/<tag>_<nummer>.<endung>
    labels/{train,val,test}/<tag>_<nummer>.txt
    data.yaml      (fuer train_stage2_plate_detection.py --data)
    mapping.csv    (neuer Name -> Quelle, Split, Originalname)

Aufruf:
  python unify_yolo_labels.py --data-dir "C:\\BA\\Daten"
"""

import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET  # zum Lesen der Pascal-VOC-XML-Dateien
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Feste Einstellungen
# ---------------------------------------------------------------------------
CLASS_NAME = "license_plate"
CLASS_ID = 0                          # die einzige Klasse bekommt die ID 0
VAL_SPLIT_FOR_UNSPLIT_SOURCES = 0.15  # Anteil "val" fuer Quellen ohne eigene Aufteilung (archive (3))
RANDOM_SEED = 42                      # fester Zufallswert -> bei jedem Lauf dieselbe Aufteilung

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- Beschreibung der YOLO-Quellen -------------------------------------------
#   path            : moegliche Namen (Ordner ODER Zip). Der erste, der
#                     existiert, wird genommen.
#   tag             : kurzer Name der Quelle, kommt in jeden Dateinamen
#   split_map       : Split-Name in der Quelle -> unser Split-Name
#                     (Roboflow nennt "val" z.B. "valid")
#   known_class_ids : welche Klassen-IDs in den Original-Labels vorkommen
#                     DUERFEN. Taucht eine andere auf, bricht das Skript ab
#                     (siehe remap_yolo_label), statt still falsch zu labeln.
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

# --- Quellen im Pascal-VOC-Format (XML mit Pixel-Ecken) ----------------------
VOC_SOURCES = [
    {"path": ["archive (3).zip"], "tag": "kaggle3"},
]


def log(msg: str) -> None:
    """Konsolenausgabe mit Praefix."""
    print(f"[unify] {msg}")


def reset_output(out_dir: Path) -> None:
    """Loescht einen alten Ausgabeordner und legt die leere Struktur neu an,
    damit keine Reste eines frueheren Laufs im Datensatz landen."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


# ===========================================================================
# TEIL 1: Quellen finden (Ordner oder Zip)
# ===========================================================================

def _normalize_name(name: str) -> str:
    """Macht einen Namen vergleichbar: klein, nur Buchstaben und Ziffern.
    "archive (1)" -> "archive1". Wird fuer den Kaggle-Fallback unten gebraucht."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_source_dir(data_dir: Path, candidates, work_dir: Path, tag: str) -> Path:
    """Liefert einen Ordner, in dem die Dateien einer Quelle liegen.

    Ablauf:
      1. Die Kandidaten-Namen der Reihe nach probieren (z.B. erst der
         entpackte Ordner "DatenSatz1", dann "archive (1).zip").
      2. Nichts gefunden? Dann alle Ordner in data_dir mit "normalisiertem"
         Namen vergleichen. Grund: Kaggle entpackt hochgeladene Zips selbst
         und benennt sie dabei leicht um (z.B. ohne Klammern/Leerzeichen).
      3. Ist das Gefundene ein Ordner -> direkt zurueckgeben.
         Ist es eine Zip-Datei -> nach work_dir/<tag> entpacken.
    """
    if isinstance(candidates, str):
        candidates = [candidates]

    # Schritt 1: exakte Namen
    src_path = None
    for rel_path in candidates:
        candidate = data_dir / rel_path
        if candidate.exists():
            src_path = candidate
            break

    # Schritt 2: Namensabgleich (Kaggle-Fallback)
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

    # Schritt 3: Ordner direkt nutzen oder Zip entpacken
    if src_path.is_dir():
        return src_path

    dest = work_dir / tag
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_path) as z:
        z.extractall(dest)
    return dest


# ===========================================================================
# TEIL 2: Bilder + Labels in den gemeinsamen Datensatz schreiben
# ===========================================================================

# Zaehler pro Quelle fuer die neuen, kurzen Dateinamen.
# WARUM neue Namen? Manche Originalnamen (z.B. in DatenSatz1) sind ueber 150
# Zeichen lang. Zusammen mit dem Ordnerpfad wuerde Windows' Grenze von 260
# Zeichen pro Pfad (MAX_PATH) ueberschritten. Ausserdem koennten zwei Quellen
# zufaellig gleich benannte Dateien haben; der Tag davor verhindert das.
_name_counters: dict = {}
_mapping_rows: list = []   # Zeilen fuer mapping.csv (Nachvollziehbarkeit)


def write_pair(img_path: Path, label_text: str, out_dir: Path, split: str, source_tag: str) -> None:
    """Kopiert EIN Bild in den Zieldatensatz und schreibt sein Label daneben.

    Neuer Name = <tag>_<laufende Nummer>, z.B. roboflow_000042.jpg und
    roboflow_000042.txt. Der Originalname wird in mapping.csv notiert."""
    idx = _name_counters.get(source_tag, 0) + 1
    _name_counters[source_tag] = idx
    out_name = f"{source_tag}_{idx:06d}"

    out_img = out_dir / "images" / split / f"{out_name}{img_path.suffix.lower()}"
    out_lbl = out_dir / "labels" / split / f"{out_name}.txt"

    # Sicherheitsnetz: darf eigentlich nie passieren. Falls doch, lieber
    # laut abbrechen als ein Bild still zu ueberschreiben.
    if out_img.exists():
        raise RuntimeError(f"Dateinamen-Kollision trotz laufender Nummer: {out_img}")

    shutil.copyfile(img_path, out_img)
    out_lbl.write_text(label_text, encoding="utf-8")
    _mapping_rows.append((out_name, source_tag, split, img_path.name))


def write_mapping_csv(out_dir: Path) -> None:
    """Schreibt mapping.csv: neuer Name -> Quelle, Split, Originalname.
    Damit laesst sich fuer jedes Bild nachweisen, woher es stammt, und man
    kann z.B. zaehlen, wie viele Bilder jede Quelle beitraegt (fuer die
    Datensatz-Tabelle in der Arbeit)."""
    import csv

    with (out_dir / "mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "source_tag", "split", "original_filename"])
        writer.writerows(_mapping_rows)


# ===========================================================================
# TEIL 3: Labels umrechnen
# ===========================================================================

def remap_yolo_label(label_path: Path, known_class_ids: set) -> str:
    """Fuer Quellen, die schon im YOLO-Format sind: Klassen-ID jeder Zeile
    auf unsere einheitliche ID 0 setzen, Box-Koordinaten unveraendert lassen.

    Kontrolle: Kommt eine Klassen-ID vor, die laut YOLO_SOURCES nicht
    vorkommen darf, bricht das Skript ab. Sonst koennte z.B. ein Datensatz,
    der zusaetzlich "car" gelabelt hat, unbemerkt Autos als Kennzeichen in
    unseren Datensatz schmuggeln."""
    lines_out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()           # "<klasse> <x> <y> <w> <h>"
        cls_id = int(parts[0])
        if cls_id not in known_class_ids:
            raise ValueError(
                f"Unerwartete Klassen-ID {cls_id} in {label_path} "
                f"(erwartet: {sorted(known_class_ids)}). Bitte pruefen, "
                f"ob dieser Datensatz wirklich nur Kennzeichen enthaelt."
            )
        rest = parts[1:]               # Koordinaten bleiben wie sie sind
        lines_out.append(" ".join([str(CLASS_ID)] + rest))
    return "\n".join(lines_out) + ("\n" if lines_out else "")


def voc_to_yolo_label(xml_path: Path) -> tuple:
    """Rechnet eine Pascal-VOC-XML-Datei (archive (3)) in YOLO-Zeilen um.

    Pascal VOC beschreibt eine Box ueber zwei Ecken in Pixeln:
        xmin, ymin (oben links) und xmax, ymax (unten rechts)
    YOLO braucht Mittelpunkt + Breite/Hoehe, relativ zur Bildgroesse:
        x_mitte = (xmin + xmax) / 2 / Bildbreite   usw.

    Gibt (label_text, bild_dateiname) zurueck. Der Bildname steht in der XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    img_w = float(size.findtext("width"))
    img_h = float(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):     # ein <object> pro Kennzeichen
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        # Ecken auf das Bild begrenzen. Manche Annotationen ragen ein paar
        # Pixel ueber den Rand hinaus; YOLO-Werte muessen aber in 0..1 liegen.
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


# ===========================================================================
# TEIL 4: eine komplette Quelle verarbeiten
# ===========================================================================

def process_yolo_source(src_dir: Path, split_map: dict, source_tag: str, known_class_ids: set, out_dir: Path, stats: dict) -> None:
    """Uebernimmt eine Quelle, die schon im YOLO-Format ist, Split fuer Split."""
    for src_split, out_split in split_map.items():
        # YOLO-Datensaetze gibt es in zwei ueblichen Ordner-Layouts:
        #   A) images/<split>/...  und  labels/<split>/...   (DatenSatz1/2)
        #   B) <split>/images/...  und  <split>/labels/...   (Roboflow)
        # Erst A suchen, dann B.
        img_dirs = [d for d in src_dir.rglob(f"images/{src_split}") if d.is_dir()]
        if not img_dirs:
            img_dirs = [d for d in src_dir.rglob(f"{src_split}/images") if d.is_dir()]
        if not img_dirs:
            log(f"  kein images-Ordner fuer Split '{src_split}' in {source_tag} gefunden, ueberspringe")
            continue
        img_dir = img_dirs[0]
        # Passenden labels-Ordner zum gefundenen Layout bestimmen.
        if img_dir.parent.name == src_split:
            lbl_dir = img_dir.parent / "labels"                    # Layout B
        else:
            lbl_dir = img_dir.parent.parent / "labels" / src_split  # Layout A

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
    """Uebernimmt eine Pascal-VOC-Quelle (archive (3)).

    Diese Quelle hat keine eigene train/val-Aufteilung. Deshalb:
      1. alle XML-Dateien sortieren (feste Ausgangsreihenfolge)
      2. mit festem Seed mischen (zufaellig, aber jedes Mal gleich)
      3. die ersten 15 % -> val, den Rest -> train
    """
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


# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="Ordner, in dem DatenSatz1/DatenSatz2 (oder archive (1)/(2).zip) und die uebrigen Zip-Dateien liegen")
    ap.add_argument("--out-dir", default="kennzeichen_detection_dataset", help="Zielordner fuer den vereinheitlichten Datensatz")
    ap.add_argument("--work-dir", default="_unify_work", help="Temporaerer Ordner zum Entpacken von Zip-Quellen")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)

    # Mit leeren Ordnern starten (reproduzierbar).
    if work_dir.exists():
        shutil.rmtree(work_dir)
    reset_output(out_dir)

    stats = {"train": 0, "val": 0, "test": 0}

    # 1) Alle YOLO-Quellen uebernehmen. Fehlt eine Quelle, wird sie nur
    #    uebersprungen (Warnung), das Skript laeuft weiter.
    for src in YOLO_SOURCES:
        try:
            src_dir = resolve_source_dir(data_dir, src["path"], work_dir, src["tag"])
        except FileNotFoundError as e:
            log(f"WARNUNG: {e}, ueberspringe")
            continue
        log(f"Verarbeite {src_dir.name} (Tag: {src['tag']}) ...")
        process_yolo_source(src_dir, src["split_map"], src["tag"], src["known_class_ids"], out_dir, stats)

    # 2) Alle VOC-Quellen umrechnen und uebernehmen.
    for src in VOC_SOURCES:
        try:
            src_dir = resolve_source_dir(data_dir, src["path"], work_dir, src["tag"])
        except FileNotFoundError as e:
            log(f"WARNUNG: {e}, ueberspringe")
            continue
        log(f"Verarbeite {src_dir.name} (Tag: {src['tag']}) ...")
        process_voc_source(src_dir, src["tag"], out_dir, stats)

    # 3) data.yaml fuer YOLO schreiben (eine Klasse, Pfade relativ zur yaml).
    yaml_text = (
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    write_mapping_csv(out_dir)

    # 4) Temporaere Entpack-Ordner entfernen.
    if work_dir.exists():
        shutil.rmtree(work_dir)

    # 5) Zusammenfassung (Zahlen fuer die Datensatz-Tabelle in der Arbeit).
    total = sum(stats.values())
    log("Fertig.")
    log(f"  train: {stats['train']}")
    log(f"  val:   {stats['val']}")
    log(f"  test:  {stats['test']}")
    log(f"  gesamt: {total}")
    log(f"  -> {out_dir}/ (mit data.yaml)")


if __name__ == "__main__":
    main()
