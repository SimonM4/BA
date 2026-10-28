#!/usr/bin/env python3
"""
prepare_visdrone.py  --  DATENVORBEREITUNG FUER STUFE 1 (Fahrzeuge aus der Luft)

===========================================================================
WAS MACHT DIESES SKRIPT?
===========================================================================
Es nimmt die offiziellen VisDrone2019-DET-Zips und baut daraus einen
Datensatz, den YOLO (Ultralytics) direkt zum Trainieren lesen kann.

Warum ist das ueberhaupt noetig?
  YOLO erwartet ein ganz bestimmtes Format, VisDrone liefert ein anderes:

  VisDrone (eine Zeile pro Objekt, Pixelwerte, Ecke oben links):
      <links>,<oben>,<breite>,<hoehe>,<score>,<klasse>,<truncation>,<occlusion>
      z.B.  684,8,273,116,1,4,0,0

  YOLO (eine Zeile pro Objekt, alles relativ zur Bildgroesse 0..1,
        Box ueber ihren MITTELPUNKT beschrieben):
      <klasse> <x_mitte> <y_mitte> <breite> <hoehe>
      z.B.  3 0.354688 0.034444 0.170625 0.088889

  Warum relativ (0..1) statt Pixel? Weil YOLO die Bilder beim Training
  ohnehin auf eine feste Groesse (z.B. 640 px) skaliert. Relative Werte
  bleiben dabei gueltig, Pixelwerte nicht.

===========================================================================
ZWEI STOLPERFALLEN IM VISDRONE-FORMAT (wichtig fuer die Methodik)
===========================================================================
1) Das Feld "score" ist in den Ground-Truth-Dateien KEIN Konfidenzwert,
   sondern ein Schalter: score = 0 heisst "ignorierte Region" (z.B. dichte
   Menschenmengen, die niemand einzeln annotiert hat). Solche Zeilen werden
   verworfen. Wuerde man sie behalten, lernte das Modell Boxen, die gar
   keinem einzelnen Objekt entsprechen.

2) Die Klassen in VisDrone sind 1-basiert (1 = pedestrian ... 10 = motor,
   11 = others, 0 = ignored region). YOLO zaehlt ab 0. Deshalb wird von
   jeder Klasse 1 abgezogen. "ignored region" (0 -> -1) und "others"
   (11 -> 10) fallen danach aus dem gueltigen Bereich 0..9 heraus und
   werden verworfen.

Die Umrechnung entspricht dem offiziellen "visdrone2yolo"-Vorgehen von
Ultralytics (ultralytics/cfg/datasets/VisDrone.yaml). Das ist zitierbar und
vermeidet eine selbst erfundene, ungepruefte Variante.

===========================================================================
EINGABE / AUSGABE
===========================================================================
Eingabe (in --data-dir, als Zip ODER bereits entpackter Ordner):
  - VisDrone2019-DET-train.zip     -> wird unser Split "train"
  - VisDrone2019-DET-val.zip       -> wird unser Split "val"
  - VisDrone2019-DET-test-dev.zip  -> wird unser Split "test"
  Bewusst NICHT verwendet: VisDrone2019-DET-test-challenge.zip, denn dafuer
  gibt es keine oeffentlichen Labels. Ohne Labels kann man weder trainieren
  noch auswerten.

Ausgabe (in --out-dir, Standard "visdrone_yolo"):
  visdrone_yolo/
    images/{train,val,test}/*.jpg    (Bilder, unveraendert kopiert)
    labels/{train,val,test}/*.txt    (eine YOLO-Label-Datei pro Bild)
    data.yaml                        (sagt YOLO, wo was liegt + Klassennamen)

Aufruf:
  python prepare_visdrone.py --data-dir "C:\\BA\\Daten"
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from PIL import Image  # nur um die Bildgroesse (Breite/Hoehe) zu lesen

# ---------------------------------------------------------------------------
# Feste Einstellungen
# ---------------------------------------------------------------------------

# Die 10 VisDrone-Klassen in YOLO-Reihenfolge (Index 0 = pedestrian usw.).
# Die Reihenfolge MUSS zur Umrechnung "VisDrone-Klasse - 1" passen.
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]
NUM_CLASSES = len(CLASS_NAMES)

# Welche Zip-Datei wird zu welchem unserer Splits?
#   Zip-Dateiname -> (Ordnername nach dem Entpacken, unser Split-Name)
SPLITS = {
    "VisDrone2019-DET-train.zip": ("VisDrone2019-DET-train", "train"),
    "VisDrone2019-DET-val.zip": ("VisDrone2019-DET-val", "val"),
    "VisDrone2019-DET-test-dev.zip": ("VisDrone2019-DET-test-dev", "test"),
}


def log(msg: str) -> None:
    """Einheitliche Konsolenausgabe mit Praefix, damit man im Log sieht,
    von welchem Skript eine Meldung stammt."""
    print(f"[visdrone] {msg}")


def reset_output(out_dir: Path) -> None:
    """Loescht einen alten Ausgabeordner komplett und legt die leere
    Ordnerstruktur neu an.

    WARUM komplett loeschen? Damit nach einem zweiten Lauf keine Dateien vom
    ersten Lauf liegen bleiben. Sonst koennten z.B. alte Labels mit neuen
    Bildern gemischt werden, und der Datensatz waere nicht mehr eindeutig
    reproduzierbar."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_box(img_w: float, img_h: float, x: float, y: float, w: float, h: float):
    """Rechnet EINE Box von VisDrone nach YOLO um.

    VisDrone: (x, y) = Ecke oben links in Pixeln, w/h = Breite/Hoehe in Pixeln
    YOLO:     Mittelpunkt + Breite/Hoehe, alles geteilt durch die Bildgroesse

    Beispiel: Bild 1000 x 500 px, Box links=100, oben=50, Breite=200, Hoehe=100
      x_mitte = (100 + 200/2) / 1000 = 0.2
      y_mitte = ( 50 + 100/2) /  500 = 0.2
      breite  =  200 / 1000          = 0.2
      hoehe   =  100 /  500          = 0.2
    """
    dw, dh = 1.0 / img_w, 1.0 / img_h          # Faktoren "Pixel -> Anteil"
    x_center = (x + w / 2.0) * dw              # linke Kante + halbe Breite = Mitte
    y_center = (y + h / 2.0) * dh              # obere Kante + halbe Hoehe = Mitte
    return x_center, y_center, w * dw, h * dh


def convert_annotation_file(ann_path: Path, img_path: Path) -> str:
    """Liest die VisDrone-Annotationsdatei EINES Bildes und gibt den Inhalt
    der passenden YOLO-Label-Datei als Text zurueck.

    Das Bild selbst wird nur geoeffnet, um seine Breite und Hoehe zu kennen
    (die braucht man fuer die Umrechnung in relative Koordinaten)."""
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    lines = []
    for raw_line in ann_path.read_text(encoding="utf-8").splitlines():
        # Manche Zeilen enden mit einem ueberfluessigen Komma -> entfernen.
        raw_line = raw_line.strip().rstrip(",")
        if not raw_line:
            continue
        row = raw_line.split(",")
        if len(row) < 6:
            continue  # unvollstaendige/kaputte Zeile, ueberspringen

        # Stolperfalle 1 (siehe oben): score = 0 -> ignorierte Region.
        score_flag = row[4]
        if score_flag == "0":
            continue

        x, y, w, h = map(float, row[:4])
        if w <= 0 or h <= 0:
            continue  # Box ohne Flaeche ist ein Annotationsfehler

        # Stolperfalle 2 (siehe oben): 1-basiert -> 0-basiert.
        cls = int(row[5]) - 1
        if not (0 <= cls < NUM_CLASSES):
            # "ignored region" (0) und "others" (11) bzw. ungueltige Werte
            # landen hier ausserhalb von 0..9 und werden bewusst verworfen,
            # statt eine falsche Klassen-ID ins Label zu schreiben.
            continue

        xc, yc, bw, bh = convert_box(img_w, img_h, x, y, w, h)
        # 6 Nachkommastellen reichen: bei 4000 px Bildbreite ist das
        # genauer als 0,01 Pixel.
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # Bild ohne gueltige Objekte -> leere Label-Datei. Das ist fuer YOLO
    # erlaubt und sogar nuetzlich ("Hintergrundbild": hier ist nichts).
    return ("\n".join(lines) + ("\n" if lines else ""))


def resolve_extracted_dir(data_dir: Path, zip_name: str, extracted_name: str, work_dir: Path) -> Path:
    """Findet den Ordner mit images/ und annotations/ fuer einen Split.

    Ablauf:
      1. Gibt es schon einen entpackten Ordner? -> den nehmen.
      2. Sonst die Zip-Datei nach work_dir entpacken.
      3. Liegt images/ nicht direkt darin, im ganzen Unterbaum danach suchen.

    WARUM Schritt 3? Auf Kaggle beobachtet: Kaggle entpackt hochgeladene Zips
    selbst. Weil das VisDrone-Zip intern schon einen gleichnamigen Ordner
    enthaelt, entsteht eine doppelte Verschachtelung:
        VisDrone2019-DET-train/VisDrone2019-DET-train/images/...
    Die Suche in Schritt 3 faengt das ab, egal wie tief verschachtelt."""
    already_extracted = data_dir / extracted_name
    if already_extracted.is_dir():
        dest = already_extracted
    else:
        zip_path = data_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Weder Ordner '{extracted_name}' noch Zip '{zip_name}' gefunden unter {data_dir}"
            )

        dest = work_dir / extracted_name
        dest.mkdir(parents=True, exist_ok=True)
        log(f"Entpacke {zip_name} (das kann bei ~1,5 GB einen Moment dauern) ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)

    # Schritt 3: zusaetzliche Verschachtelungsebenen abfangen.
    if not (dest / "images").is_dir():
        candidates = list(dest.rglob("images"))
        if candidates:
            dest = candidates[0].parent
    return dest


def process_split(src_dir: Path, out_split: str, out_dir: Path, stats: dict) -> None:
    """Wandelt einen kompletten Split (z.B. alle Trainingsbilder) um.

    Fuer jedes Bild:
      - passende Annotationsdatei suchen (gleicher Name, Endung .txt)
      - Annotation nach YOLO umrechnen
      - Bild in den Zielordner kopieren, Label daneben schreiben
    """
    img_dir = src_dir / "images"
    ann_dir = src_dir / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Erwarte images/ und annotations/ unter {src_dir}, gefunden: {list(src_dir.iterdir())}")

    # sorted(): feste Reihenfolge -> der Lauf ist reproduzierbar
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for img_path in images:
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            log(f"  WARNUNG: keine Annotation fuer {img_path.name}, ueberspringe")
            continue

        label_text = convert_annotation_file(ann_path, img_path)

        # YOLO findet das Label zu einem Bild ueber den Pfad:
        #   images/train/abc.jpg  <->  labels/train/abc.txt
        # Deshalb muessen Ordnerstruktur und Dateiname exakt zueinander passen.
        out_img = out_dir / "images" / out_split / img_path.name
        out_lbl = out_dir / "labels" / out_split / f"{img_path.stem}.txt"
        shutil.copyfile(img_path, out_img)
        out_lbl.write_text(label_text, encoding="utf-8")
        stats[out_split] += 1


def main() -> None:
    # --- Kommandozeilen-Parameter -------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="Ordner mit den VisDrone-Zips (oder bereits entpackten Ordnern)")
    ap.add_argument("--out-dir", default="visdrone_yolo", help="Zielordner fuer den konvertierten Datensatz")
    ap.add_argument("--work-dir", default="_visdrone_work", help="Temporaerer Ordner zum Entpacken")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)

    # --- Aufraeumen: immer mit leerem Arbeits- und Zielordner starten -------
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    reset_output(out_dir)

    stats = {"train": 0, "val": 0, "test": 0}  # Zaehler fuer die Abschlussmeldung

    # --- Jeden Split umwandeln ------------------------------------------------
    # Die Aufteilung train/val/test wird von VisDrone UEBERNOMMEN, nicht neu
    # ausgelost. Vorteil: Ergebnisse sind mit anderen Arbeiten auf VisDrone
    # vergleichbar, und es gibt keine Ueberschneidung zwischen den Splits.
    for zip_name, (extracted_name, out_split) in SPLITS.items():
        try:
            src_dir = resolve_extracted_dir(data_dir, zip_name, extracted_name, work_dir)
        except FileNotFoundError as e:
            # Fehlt ein Split (z.B. test-dev nicht heruntergeladen), wird er
            # nur uebersprungen statt das ganze Skript abzubrechen.
            log(f"WARNUNG: {e}, ueberspringe Split '{out_split}'")
            continue
        log(f"Verarbeite {extracted_name} -> Split '{out_split}' ...")
        process_split(src_dir, out_split, out_dir, stats)

    # --- data.yaml schreiben -------------------------------------------------
    # Diese Datei uebergibt man spaeter an train_stage1_vehicle_detection.py
    # (--data). Sie sagt YOLO: wo liegen die Bilder je Split, wie viele
    # Klassen gibt es (nc) und wie heissen sie. Die Pfade sind relativ zum
    # Ordner, in dem die data.yaml liegt.
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

    # Temporaeren Entpack-Ordner wieder entfernen (spart Speicherplatz).
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    # --- Zusammenfassung (Zahlen fuer das Datensatz-Kapitel der Arbeit) ------
    total = sum(stats.values())
    log("Fertig.")
    log(f"  train: {stats['train']}")
    log(f"  val:   {stats['val']}")
    log(f"  test:  {stats['test']}")
    log(f"  gesamt: {total}")
    log(f"  -> {out_dir}/ (mit data.yaml, {NUM_CLASSES} Klassen)")


if __name__ == "__main__":
    main()
