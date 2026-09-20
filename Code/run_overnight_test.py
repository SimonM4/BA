#!/usr/bin/env python3
"""
run_overnight_test.py

EIN einziges Script fuer einen ersten End-to-End-Test von STUFE 2
(Kennzeichenerkennung): bereitet bei Bedarf den Datensatz auf und startet
danach automatisch das Training -- alles in einem einzigen Aufruf, genau wie
fuer einen Ueber-Nacht-Testlauf gewuenscht.

Warum Stufe 2 und nicht Stufe 1 oder 3 als erster Test?
Stufe 2 (Kennzeichen-Detection) ist der methodische Kern deiner Arbeit, und
ihr Datensatz ist kleiner/schneller trainiert als der VisDrone-Datensatz aus
Stufe 1 (VisDrone train-Zip allein ist >1,5 GB, deutlich mehr Bilder). Damit
eignet sich Stufe 2 am besten, um ALS ERSTES zu pruefen, ob die komplette
Pipeline (Datenaufbereitung -> Training -> gespeichertes Modell) technisch
sauber durchlaeuft, bevor du die groessere Stufe-1-Trainingsrunde startest
(z.B. spaeter auf Kaggle mit GPU). Willst du stattdessen zuerst Stufe 1
testen: prepare_visdrone.py und train_stage1_vehicle_detection.py (das jetzt
ebenfalls --time unterstuetzt) von Hand nacheinander aufrufen, oder Bescheid
geben -- dann gibt es dieses Script auch fuer Stufe 1.

Was macht das Script konkret:
  1. Prueft, ob der vereinheitlichte Kennzeichen-Datensatz (aus
     unify_yolo_labels.py) schon existiert. Falls nicht: baut ihn zuerst auf
     (dauert je nach Rechner mehrere Minuten, v.a. wegen des Entpackens der
     grossen Zip-Dateien). Existiert er schon (z.B. von einem frueheren
     Lauf), wird dieser Schritt uebersprungen -- so kannst du dieses Script
     mehrfach starten, ohne jedes Mal neu zu entpacken.
  2. Startet danach das YOLO26-Training aus train_stage2_plate_detection.py
     -- mit --time statt einer festen Epochenzahl: das Training laeuft
     hoechstens so viele STUNDEN (Default: 8) und stoppt danach automatisch
     von selbst, egal wie schnell oder langsam dein Rechner ist. Ultralytics
     speichert dabei nach JEDER Epoche automatisch den aktuellen Stand --
     selbst wenn das Zeitlimit mitten im Training greift, hast du morgens
     ein nutzbares Modell (den Stand der letzten abgeschlossenen Epoche).

WICHTIG, bevor du das ueber Nacht laufen laesst:
  - Windows darf waehrend der Nacht NICHT in den Energiesparmodus/Standby
    gehen, sonst pausiert das Training mit! Windows-Einstellungen -> System
    -> Energie -> "Bildschirm und Standby" -> Standby auf "Nie" (zumindest
    fuer heute Nacht), oder im Terminal als Administrator:
        powercfg /change standby-timeout-ac 0
  - Das Terminal/die IDE, in der du dieses Script startest, muss die ganze
    Nacht offen bleiben (Fenster nicht schliessen).
  - Laptop: am Netzteil lassen.
  - Willst du die komplette Ausgabe zusaetzlich in einer Datei mitschreiben
    (falls du morgens erst spaeter ans Terminal-Fenster kommst), starte statt
    "python run_overnight_test.py" z.B.:
        python run_overnight_test.py > ergebnis_heute_nacht.log 2>&1

Aufruf (aus dem Code-Ordner, z.B. C:\\BA\\Code):
  python run_overnight_test.py
  python run_overnight_test.py --hours 10     (laenger laufen lassen)
  python run_overnight_test.py --hours 0.1    (kurzer Rauch-Test, ~6 Minuten,
                                                um VOR der eigentlichen Nacht
                                                zu pruefen, dass alles startet)

Wo das Ergebnis landet: im Ueberordner von Code/, also z.B.
"C:/BA/runs_stage2/plate_detection_first_test/..." (absoluter Pfad, daher
ohne "runs/detect/"). Den genauen Pfad druckt train_stage2_plate_detection.py
am Ende selbst mit aus ("[stage2] Fertig. Bestes Modell liegt unter: ...").
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Alle Pfade relativ zum Ort DIESES Scripts berechnen statt sie hart zu
# codieren -- so funktioniert das Script unabhaengig davon, wo der BA-Ordner
# liegt (genau solche festen Pfade -- siehe die fruehere Umstellung von
# OneDrive auf C:\BA -- wollen wir kuenftig vermeiden).
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent


def run(cmd: list) -> None:
    """Fuehrt einen Unterprozess aus und gibt seine Ausgabe LIVE weiter (kein
    capture_output), damit du den Trainingsfortschritt direkt im Terminal
    siehst. Bricht mit einer Fehlermeldung ab, falls der Unterprozess
    fehlschlaegt (z.B. Datensatz nicht gefunden, ungueltige Parameter)."""
    print(f"\n[overnight] $ {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(BASE_DIR / "Daten"), help="Ordner mit den Rohdaten-Zips (Default: Daten/ neben Code/)")
    ap.add_argument("--dataset-out", default=str(BASE_DIR / "Daten" / "kennzeichen_detection_dataset"),
                     help="Wohin der vereinheitlichte Datensatz kommt (Default: in Daten/, damit .gitignore automatisch greift)")
    ap.add_argument("--hours", type=float, default=8.0, help="Maximale Trainingsdauer in Stunden (Default: 8)")
    ap.add_argument("--name", default="plate_detection_first_test", help="Name dieses Trainingslaufs (Unterordner in runs_stage2/)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    dataset_out = Path(args.dataset_out)
    data_yaml = dataset_out / "data.yaml"

    # Schritt 1: Datensatz nur aufbauen, wenn er noch nicht existiert.
    if data_yaml.exists():
        print(f"[overnight] Datensatz existiert bereits ({data_yaml}), ueberspringe Aufbereitung.")
    else:
        print("[overnight] Baue vereinheitlichten Kennzeichen-Datensatz auf ...")
        run([
            sys.executable, str(CODE_DIR / "unify_yolo_labels.py"),
            "--data-dir", str(data_dir),
            "--out-dir", str(dataset_out),
        ])

    # Schritt 2: Training, zeitlich begrenzt statt mit fester Epochenzahl --
    # laeuft hoechstens --hours Stunden, egal ob CPU oder GPU, und speichert
    # nach jeder Epoche automatisch den aktuellen Stand.
    print(f"\n[overnight] Starte Training (max. {args.hours} Stunden) ...")
    run([
        sys.executable, str(CODE_DIR / "train_stage2_plate_detection.py"),
        "--data", str(data_yaml),
        "--time", str(args.hours),
        "--name", args.name,
    ])

    # WICHTIG: wo genau Ultralytics speichert, haengt von der installierten
    # Version ab (teils "runs_stage2/<name>/...", teils zusaetzlich unter
    # "runs/detect/" verschachtelt) -- den verlaesslichen Pfad druckt
    # train_stage2_plate_detection.py oben in der Zeile "Fertig. Bestes Modell
    # liegt unter: ..." bereits selbst mit aus (per model.trainer.save_dir,
    # nicht geraten). Hier daher bewusst KEIN zweiter, moeglicherweise
    # falscher Pfad.
    print(
        "\n[overnight] Fertig (oder Zeitlimit erreicht). Den genauen Pfad zu "
        "best.pt/last.pt siehst du oben in der Zeile "
        "'[stage2] Fertig. Bestes Modell liegt unter: ...' des Trainings.\n"
        "\nZum Testen morgen frueh, z.B. in Python (Pfad aus der Zeile oben einsetzen):\n"
        "  from ultralytics import YOLO\n"
        "  model = YOLO(r'<Pfad zu best.pt aus der Zeile oben>')\n"
        "  results = model.predict('ein_testbild.jpg', save=True)  # speichert das Bild mit eingezeichneten Boxen\n"
        "  model.val()  # rechnet mAP & Co. nochmal sauber auf dem Val-Split aus\n"
    )


if __name__ == "__main__":
    main()
