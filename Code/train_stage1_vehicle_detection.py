#!/usr/bin/env python3
"""
train_stage1_vehicle_detection.py

STUFE 1 der Pipeline: Fahrzeuge/Objekte aus der Luft erkennen (VisDrone).

Trainiert ein YOLO26-Modell auf dem mit prepare_visdrone.py erzeugten
Datensatz. Erkennt die 10 VisDrone-Klassen (pedestrian, people, bicycle,
car, van, truck, tricycle, awning-tricycle, bus, motor).

Das trainierte Modell ist Stufe 1 deiner Pipeline: Es findet Fahrzeuge (und
andere Objekte) im vollen Drohnenbild. Die gefundenen Fahrzeug-Boxen
schneidest du danach aus dem Bild aus und gibst sie an Stufe 2
(train_stage2_plate_detection.py) weiter, die darin das Kennzeichen sucht.

Warum ein SEPARATES Modell und nicht ein gemeinsames mit Stufe 2?
VisDrone-Klassen (Fahrzeuge/Personen/etc. im GESAMTEN Luftbild) und
Kennzeichen (kleines Detail auf einem bereits zugeschnittenen Fahrzeugbild)
sind zwei unterschiedliche Skalen/Aufgaben. Ein gemeinsames Modell wuerde
beide Aufgaben vermischen; zwei spezialisierte Modelle sind einfacher zu
trainieren, zu bewerten und in der Arbeit sauber zu beschreiben.

Warum "yolo26n" (nano)?
Die "n"-Variante hat nur ca. 2,4 Mio. Parameter und ist explizit fuer
ressourcenarme Umgebungen ausgelegt (Edge-Deployment). Das passt zu einer
Bachelorarbeit ohne eigene High-End-GPU: sie trainiert deutlich schneller
und passt bequem in die kostenlosen GPU-Kontingente von Kaggle/Colab (siehe
--help unten bzw. die Erklaerung im Chat). Ueber --model kannst du trotzdem
jederzeit auf eine groessere Variante wechseln (yolo26s.pt, yolo26m.pt, ...),
falls du merkst, dass "n" fuer deine Ergebnisse nicht reicht.

Voraussetzung: `pip install ultralytics` (siehe requirements.txt).

Aufruf (Standardwerte reichen meist fuer einen ersten Testlauf):
  python train_stage1_vehicle_detection.py --data "C:\\...\\visdrone_yolo\\data.yaml"

Fuer den finalen Trainingslauf z.B. auf Kaggle mit GPU:
  python train_stage1_vehicle_detection.py --data visdrone_yolo/data.yaml --epochs 100 --device 0
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Pfad zur data.yaml aus prepare_visdrone.py")
    ap.add_argument("--model", default="yolo26n.pt", help="Basismodell (wird bei Bedarf automatisch heruntergeladen)")
    ap.add_argument("--epochs", type=int, default=100, help="Anzahl Trainings-Epochen")
    ap.add_argument("--imgsz", type=int, default=640, help="Trainingsbildgroesse (Pixel, quadratisch)")
    ap.add_argument("--batch", default=16, help="Batch-Groesse. -1 = automatisch ~60%% des GPU-Speichers nutzen")
    ap.add_argument("--device", default=None, help="z.B. '0' fuer erste GPU, 'cpu' fuer CPU. Default: automatisch")
    ap.add_argument("--project", default="runs_stage1", help="Ausgabeordner fuer Trainingsergebnisse/Checkpoints")
    ap.add_argument("--name", default="vehicle_detection", help="Name dieses Trainingslaufs (Unterordner in --project)")
    ap.add_argument("--patience", type=int, default=20, help="Early Stopping: Abbruch nach so vielen Epochen ohne Verbesserung")
    args = ap.parse_args()

    # batch kann eine Ganzzahl ODER ein Bruchteil (z.B. 0.7 = 70% GPU-Speicher)
    # sein -- deshalb hier flexibel parsen statt fest auf int zu tippen.
    try:
        batch = int(args.batch)
    except ValueError:
        batch = float(args.batch)

    # Vortrainiertes YOLO26-Modell laden. Es wurde bereits auf COCO
    # (allgemeine Objekterkennung) trainiert -- wir nutzen das als
    # Ausgangspunkt (Transfer Learning) statt bei Null anzufangen. Das ist
    # bei begrenzten Trainingsdaten/-zeit fast immer die bessere Wahl.
    model = YOLO(args.model)

    # Kernstueck: das eigentliche Training.
    # - mosaic-Augmentierung ist standardmaessig aktiv (mosaic=1.0), d.h. es
    #   werden automatisch 4 Trainingsbilder zu einem zusammengesetzt. Das
    #   hilft dem Modell u.a., kleine/mehrere Objekte pro Bild besser zu
    #   erkennen -- fuer Luftbilder mit oft vielen kleinen Fahrzeugen
    #   genau das, was wir wollen.
    # - patience: bricht das Training vorzeitig ab, wenn sich die
    #   Validierungsmetrik laenger nicht verbessert (spart Zeit/GPU-Stunden).
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )

    # Nach dem Training auf dem eigenen Val-Split evaluieren und die
    # Kennzahlen (mAP etc.) ausgeben -- fuer den Ergebnisteil der Arbeit.
    metrics = model.val()
    print("\n[stage1] Validierungsergebnisse:")
    print(metrics)

    print(
        f"\n[stage1] Fertig. Bestes Modell liegt unter: "
        f"{args.project}/{args.name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
