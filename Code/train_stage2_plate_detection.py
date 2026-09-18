#!/usr/bin/env python3
"""
train_stage2_plate_detection.py

STUFE 2 der Pipeline: Das Kennzeichen auf einem (bereits durch Stufe 1
gefundenen) Fahrzeug lokalisieren.

Trainiert ein YOLO26-Modell auf dem mit unify_yolo_labels.py erzeugten
Datensatz (kennzeichen_detection_dataset/), einer einzigen Klasse
"license_plate".

Technisch ist dieses Script fast identisch zu train_stage1_vehicle_detection.py
-- das ist Absicht: es ist dieselbe Art Aufgabe (Objekterkennung), nur mit
anderen Trainingsdaten und einer Klasse statt zehn. Die beiden Stufen bekommen
bewusst getrennte Scripts (statt eines gemeinsamen mit einem
Modus-Parameter), damit jede Stufe fuer sich klar nachvollziehbar bleibt --
gerade fuer die Methodik-Beschreibung in der Arbeit ist es einfacher, zwei
kurze, eigenstaendige Trainingslaeufe zu beschreiben als ein verzweigtes
Script.

Startet ebenfalls von den COCO-vortrainierten YOLO26-Gewichten (nicht von
den in Stufe 1 trainierten!) -- Stufe 1 und Stufe 2 sehen sehr
unterschiedliche Bildausschnitte (ganzes Luftbild vs. zugeschnittenes
Fahrzeug), ein Verketten der beiden Trainings wuerde hier keinen klaren
Vorteil bringen und die Nachvollziehbarkeit unnoetig verkomplizieren.

Voraussetzung: `pip install ultralytics`.

Aufruf:
  python train_stage2_plate_detection.py --data "C:\\...\\kennzeichen_detection_dataset\\data.yaml"
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Pfad zur data.yaml aus unify_yolo_labels.py")
    ap.add_argument("--model", default="yolo26n.pt", help="Basismodell (wird bei Bedarf automatisch heruntergeladen)")
    ap.add_argument("--epochs", type=int, default=100, help="Anzahl Trainings-Epochen")
    ap.add_argument("--imgsz", type=int, default=640, help="Trainingsbildgroesse (Pixel, quadratisch)")
    ap.add_argument("--batch", default=16, help="Batch-Groesse. -1 = automatisch ~60%% des GPU-Speichers nutzen")
    ap.add_argument("--device", default=None, help="z.B. '0' fuer erste GPU, 'cpu' fuer CPU. Default: automatisch")
    ap.add_argument("--project", default="runs_stage2", help="Ausgabeordner fuer Trainingsergebnisse/Checkpoints")
    ap.add_argument("--name", default="plate_detection", help="Name dieses Trainingslaufs (Unterordner in --project)")
    ap.add_argument("--patience", type=int, default=20, help="Early Stopping: Abbruch nach so vielen Epochen ohne Verbesserung")
    args = ap.parse_args()

    try:
        batch = int(args.batch)
    except ValueError:
        batch = float(args.batch)

    model = YOLO(args.model)

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

    metrics = model.val()
    print("\n[stage2] Validierungsergebnisse:")
    print(metrics)

    print(
        f"\n[stage2] Fertig. Bestes Modell liegt unter: "
        f"{args.project}/{args.name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
