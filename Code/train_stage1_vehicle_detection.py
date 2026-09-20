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

Fuer einen zeitlich begrenzten Testlauf (z.B. "ueber Nacht, egal wie schnell
der Rechner ist") --time statt/zusaetzlich zu --epochs angeben -- siehe
--time weiter unten.

Wurde ein Lauf unterbrochen (Session-Abriss, GPU-Quota alle, versehentlich
gestoppt) und soll dort WEITERGEMACHT statt neu gestartet werden -- --resume
mit dem Pfad zur last.pt DIESES unterbrochenen Laufs angeben, z.B.:
  python train_stage1_vehicle_detection.py --resume /kaggle/working/runs/detect/runs_stage1/vehicle_detection-6/weights/last.pt
Dabei werden --data/--model/--epochs/--imgsz/... etc. IGNORIERT: Ultralytics
liest alle urspruenglichen Einstellungen (inkl. --time) selbst wieder aus dem
unterbrochenen Lauf ein und haengt im selben Ausgabeordner weiter an, statt
bei Epoche 0 neu zu beginnen. WICHTIG: das setzt voraus, dass last.pt noch
existiert -- auf Kaggle liegt das nur im /kaggle/working/ DERSELBEN Session
(nicht persistent ueber Sessions hinweg, ausser du hast es vorher manuell
heruntergeladen oder per "Save & Run All" committed).

Automatisches Zwischenspeichern nach Kaggle (optional, gegen genau das
"Session ist ploetzlich weg"-Problem): mit --kaggle-checkpoint-dataset
<username>/<dataset-slug> laedt dieses Skript alle --checkpoint-every
Epochen (Default: 10) automatisch best.pt/last.pt/results.csv als neue
Version eines Kaggle-Datasets hoch. Bricht die Session danach unerwartet ab
(Zeitlimit, GPU-Kontingent alle, Absturz), geht so hoechstens der
Fortschritt seit dem letzten Checkpoint verloren -- nicht der ganze Lauf.
Ein fehlgeschlagener Upload (kein Internet, keine Anmeldung, ...) bricht
das Training NICHT ab, sondern gibt nur eine Warnung aus.
  python train_stage1_vehicle_detection.py --data ... --kaggle-checkpoint-dataset deinusername/ba-stage1-checkpoints

WICHTIG zur Authentifizierung: NIE den Kaggle-API-Token direkt im Code, in
einer Notebook-Zelle oder im Chat eintragen (auch nicht als "export ..."-
Befehl) -- der landet sonst im Notebook-Verlauf/Output bzw. Chat-Verlauf
und gilt danach als kompromittiert (auf kaggle.com/settings widerrufen und
neu erzeugen, falls doch mal passiert!). Stattdessen im Kaggle-Notebook-
Editor unter "Add-ons -> Secrets" einen Secret anlegen (z.B. Label
"KAGGLE_API_TOKEN") und ZU BEGINN des Notebooks (nicht in diesem Skript)
so laden:
    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
Laeuft das Notebook direkt auf Kaggle, ist die Kaggle-API haeufig ohnehin
schon automatisch authentifiziert -- einfach ohne Secret testen (z.B.
"!kaggle datasets list") und nur bei Bedarf wie oben nachruesten.
"""

import argparse
import json
import subprocess
from pathlib import Path

from ultralytics import YOLO

# Ausgabe im UEBERORDNER von Code/ (z.B. C:\BA\runs_stage1), nicht in Code/runs/. Ein
# ABSOLUTER Pfad verhindert, dass Ultralytics noch "runs/detect/" davorhaengt.
DEFAULT_PROJECT = str(Path(__file__).resolve().parent.parent / "runs_stage1")
# Vortrainierte Gewichte liegen in C:/BA/models/ (nicht in Code/); fehlen sie dort,
# laedt Ultralytics yolo26n.pt automatisch herunter.
_LOCAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolo26n.pt"
DEFAULT_MODEL = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "yolo26n.pt"

def _make_kaggle_checkpoint_callback(dataset_slug: str, checkpoint_every: int, stage_name: str):
    """Baut einen Ultralytics-Trainings-Callback, der alle `checkpoint_every`
    Epochen den aktuellen Zwischenstand (best.pt/last.pt/results.csv) als
    neue Version eines Kaggle-Datasets hochlaedt.

    Warum das ueberhaupt noetig ist: Eine Kaggle-Session kann jederzeit
    unerwartet enden (Zeitlimit, GPU-Kontingent alle, Absturz) -- der
    Python-Prozess bekommt davon i.d.R. NICHTS mehr mit, es gibt keinen
    zuverlaessigen "kurz-vorher-noch-schnell-speichern"-Haken. Der einzige
    robuste Schutz ist deshalb, waehrend des Trainings regelmaessig zu
    sichern statt erst am Ende zu versuchen, etwas zu retten.

    Setzt voraus, dass die Kaggle-API bereits authentifiziert ist (siehe
    Docstring oben -- NIE den Token hart in diese Funktion eintragen).

    WICHTIG zum gewaehlten Callback-Zeitpunkt: bewusst "on_model_save" statt
    "on_train_epoch_end" -- Ultralytics schreibt best.pt/last.pt erst NACH
    on_train_epoch_end (naemlich nach Validierung, siehe save_model() in
    BaseTrainer). Wuerde hier auf on_train_epoch_end gehoert, wuerden bei
    jedem Checkpoint die Gewichte der VORHERIGEN (nicht der aktuellen)
    Epoche hochgeladen. "on_model_save" feuert direkt NACHDEM die Dateien
    tatsaechlich geschrieben wurden.
    """

    state = {"dataset_exists": False}

    def callback(trainer) -> None:
        epoch = trainer.epoch + 1  # trainer.epoch ist 0-indiziert
        if epoch % checkpoint_every != 0:
            return

        weights_dir = trainer.save_dir / "weights"
        if not weights_dir.exists():
            return

        try:
            ckpt_dir = trainer.save_dir / "_kaggle_checkpoint_upload"
            ckpt_dir.mkdir(exist_ok=True)
            for fname in ("best.pt", "last.pt"):
                src = weights_dir / fname
                if src.exists():
                    (ckpt_dir / fname).write_bytes(src.read_bytes())
            results_csv = trainer.save_dir / "results.csv"
            if results_csv.exists():
                (ckpt_dir / "results.csv").write_bytes(results_csv.read_bytes())

            (ckpt_dir / "dataset-metadata.json").write_text(
                json.dumps({
                    "title": f"BA Checkpoint {stage_name}",
                    "id": dataset_slug,
                    "licenses": [{"name": "CC0-1.0"}],
                }),
                encoding="utf-8",
            )

            if not state["dataset_exists"]:
                # Erster Upload dieser Session: das Dataset existiert vielleicht
                # schon aus einer frueheren Session -- dann schlaegt "create"
                # fehl, das ist hier KEIN echter Fehler, einfach mit "version"
                # weitermachen.
                result = subprocess.run(
                    ["kaggle", "datasets", "create", "-p", str(ckpt_dir), "-q"],
                    capture_output=True, text=True,
                )
                state["dataset_exists"] = True
                if result.returncode != 0:
                    print(f"[{stage_name}] Hinweis: 'create' fehlgeschlagen (Dataset existiert vermutlich schon): {result.stderr.strip()[:200]}, versuche 'version' ...")
                    result = subprocess.run(
                        ["kaggle", "datasets", "version", "-p", str(ckpt_dir), "-m", f"epoch {epoch}", "-q"],
                        capture_output=True, text=True,
                    )
            else:
                result = subprocess.run(
                    ["kaggle", "datasets", "version", "-p", str(ckpt_dir), "-m", f"epoch {epoch}", "-q"],
                    capture_output=True, text=True,
                )

            if result.returncode == 0:
                print(f"[{stage_name}] Epoche {epoch}: Zwischenstand nach Kaggle-Dataset '{dataset_slug}' gesichert.")
            else:
                print(f"[{stage_name}] Warnung: Checkpoint-Upload (Epoche {epoch}) fehlgeschlagen: {result.stderr.strip()[:200]}")
        except Exception as e:
            # Ein fehlgeschlagener Upload darf NIE das Training abbrechen --
            # deshalb hier bewusst breit abgefangen und nur gewarnt.
            print(f"[{stage_name}] Warnung: Checkpoint-Upload (Epoche {epoch}) fehlgeschlagen ({e}), Training laeuft weiter.")

    return callback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="Pfad zur data.yaml aus prepare_visdrone.py (nicht noetig bei --resume)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Basismodell (wird bei Bedarf automatisch heruntergeladen)")
    ap.add_argument("--epochs", type=int, default=100, help="Anzahl Trainings-Epochen (wird von --time ueberschrieben, falls gesetzt)")
    ap.add_argument("--time", type=float, default=None, help="Maximale Trainingsdauer in STUNDEN. Falls gesetzt, ueberschreibt das --epochs: das Training laeuft hoechstens so lange und stoppt danach automatisch -- egal wie schnell/langsam die Hardware ist. Praktisch fuer 'ueber Nacht trainieren' (z.B. --time 8). Ultralytics speichert dabei nach jeder Epoche den aktuellen Stand, das Zeitlimit fuehrt also nie zu einem komplett verlorenen Lauf.")
    ap.add_argument("--imgsz", type=int, default=640, help="Trainingsbildgroesse (Pixel, quadratisch)")
    ap.add_argument("--batch", default=16, help="Batch-Groesse. -1 = automatisch ~60%% des GPU-Speichers nutzen")
    ap.add_argument("--device", default=None, help="z.B. '0' fuer erste GPU, 'cpu' fuer CPU. Default: automatisch")
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="Ausgabeordner fuer Trainingsergebnisse/Checkpoints")
    ap.add_argument("--name", default="vehicle_detection", help="Name dieses Trainingslaufs (Unterordner in --project)")
    ap.add_argument("--patience", type=int, default=20, help="Early Stopping: Abbruch nach so vielen Epochen ohne Verbesserung")
    ap.add_argument("--resume", default=None, help="Pfad zur last.pt eines unterbrochenen Laufs -- macht dort weiter statt neu zu starten (siehe Docstring oben). Wenn gesetzt, werden alle anderen Trainings-Parameter ignoriert.")
    ap.add_argument("--kaggle-checkpoint-dataset", default=None, help="Kaggle-Dataset-Slug (z.B. 'username/ba-stage1-checkpoints'), an das alle --checkpoint-every Epochen automatisch ein Zwischenstand (best.pt/last.pt/results.csv) hochgeladen wird. Ohne Angabe passiert nichts (Standardverhalten unveraendert). Siehe Docstring oben zur sicheren Authentifizierung -- NIE den Token als Argument oder im Code eintragen.")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Nur relevant mit --kaggle-checkpoint-dataset: alle so viele Epochen wird ein Zwischenstand hochgeladen.")
    args = ap.parse_args()

    if args.resume:
        # Weitermachen: last.pt selbst enthaelt schon Modellgewichte,
        # Optimizer-Zustand UND die urspruenglichen Trainings-Argumente
        # (inkl. data/epochs/time/...) -- deshalb hier bewusst NICHTS davon
        # erneut uebergeben, nur resume=True. Ultralytics haengt automatisch
        # im selben Ausgabeordner (--project/--name aus dem unterbrochenen
        # Lauf) an der naechsten Epoche weiter an.
        print(f"[stage1] Setze unterbrochenen Lauf fort: {args.resume}")
        model = YOLO(args.resume)
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage1"),
            )
        model.train(resume=True)
    else:
        if not args.data:
            ap.error("--data ist erforderlich, ausser bei --resume")

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
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage1"),
            )

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
            time=args.time,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
        )

    # Nach dem Training auf dem eigenen Val-Split evaluieren und die
    # Kennzahlen (mAP etc.) ausgeben -- fuer den Ergebnisteil der Arbeit.
    # Val-Statistiken in einen eigenen Ordner neben dem Trainingslauf (<name>_val)
    # schreiben, statt dass Ultralytics sie nach runs/detect/val legt.
    train_dir = model.trainer.save_dir
    metrics = model.val(project=str(train_dir.parent), name=train_dir.name + "_val")
    print("\n[stage1] Validierungsergebnisse:")
    print(metrics)

    # WICHTIG: den tatsaechlich verwendeten Ausgabeordner direkt vom Trainer
    # erfragen (model.trainer.save_dir), statt ihn selbst aus --project/--name
    # zusammenzubauen. Ultralytics haengt je nach Version noch "runs/detect/"
    # davor -- der wirkliche Pfad steht zuverlaessig nur hier.
    save_dir = model.trainer.save_dir
    print(
        f"\n[stage1] Fertig. Bestes Modell liegt unter: "
        f"{save_dir / 'weights' / 'best.pt'}\n"
        f"[stage1] Letzter Trainingsstand: {save_dir / 'weights' / 'last.pt'}\n"
        f"[stage1] Verlauf pro Epoche: {save_dir / 'results.csv'}"
    )


if __name__ == "__main__":
    main()
