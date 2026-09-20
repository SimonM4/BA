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

Fuer einen zeitlich begrenzten Testlauf (z.B. "ueber Nacht, egal wie schnell
der Rechner ist") --time statt/zusaetzlich zu --epochs angeben -- siehe
--time weiter unten.

Wurde ein Lauf unterbrochen (Session-Abriss, GPU-Quota alle, versehentlich
gestoppt) und soll dort WEITERGEMACHT statt neu gestartet werden -- --resume
mit dem Pfad zur last.pt DIESES unterbrochenen Laufs angeben, z.B.:
  python train_stage2_plate_detection.py --resume /kaggle/working/runs/detect/runs_stage2/plate_detection-3/weights/last.pt
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
  python train_stage2_plate_detection.py --data ... --kaggle-checkpoint-dataset deinusername/ba-stage2-checkpoints

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

# Ausgabe im UEBERORDNER von Code/ (z.B. C:\BA\runs_stage2), nicht in Code/runs/. Ein
# ABSOLUTER Pfad verhindert, dass Ultralytics noch "runs/detect/" davorhaengt.
DEFAULT_PROJECT = str(Path(__file__).resolve().parent.parent / "runs_stage2")
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
    ap.add_argument("--data", help="Pfad zur data.yaml aus unify_yolo_labels.py (nicht noetig bei --resume)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Basismodell (wird bei Bedarf automatisch heruntergeladen)")
    ap.add_argument("--epochs", type=int, default=100, help="Anzahl Trainings-Epochen (wird von --time ueberschrieben, falls gesetzt)")
    ap.add_argument("--time", type=float, default=None, help="Maximale Trainingsdauer in STUNDEN. Falls gesetzt, ueberschreibt das --epochs: das Training laeuft hoechstens so lange und stoppt danach automatisch -- egal wie schnell/langsam die Hardware ist. Praktisch fuer 'ueber Nacht trainieren' (z.B. --time 8). Ultralytics speichert dabei nach jeder Epoche den aktuellen Stand, das Zeitlimit fuehrt also nie zu einem komplett verlorenen Lauf.")
    ap.add_argument("--imgsz", type=int, default=640, help="Trainingsbildgroesse (Pixel, quadratisch)")
    ap.add_argument("--batch", default=16, help="Batch-Groesse. -1 = automatisch ~60%% des GPU-Speichers nutzen")
    ap.add_argument("--device", default=None, help="z.B. '0' fuer erste GPU, 'cpu' fuer CPU. Default: automatisch")
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="Ausgabeordner fuer Trainingsergebnisse/Checkpoints")
    ap.add_argument("--name", default="plate_detection", help="Name dieses Trainingslaufs (Unterordner in --project)")
    ap.add_argument("--patience", type=int, default=20, help="Early Stopping: Abbruch nach so vielen Epochen ohne Verbesserung")
    ap.add_argument("--resume", default=None, help="Pfad zur last.pt eines unterbrochenen Laufs -- macht dort weiter statt neu zu starten (siehe Docstring oben). Wenn gesetzt, werden alle anderen Trainings-Parameter ignoriert.")
    ap.add_argument("--kaggle-checkpoint-dataset", default=None, help="Kaggle-Dataset-Slug (z.B. 'username/ba-stage2-checkpoints'), an das alle --checkpoint-every Epochen automatisch ein Zwischenstand (best.pt/last.pt/results.csv) hochgeladen wird. Ohne Angabe passiert nichts (Standardverhalten unveraendert). Siehe Docstring oben zur sicheren Authentifizierung -- NIE den Token als Argument oder im Code eintragen.")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Nur relevant mit --kaggle-checkpoint-dataset: alle so viele Epochen wird ein Zwischenstand hochgeladen.")
    args = ap.parse_args()

    if args.resume:
        # Weitermachen: last.pt selbst enthaelt schon Modellgewichte,
        # Optimizer-Zustand UND die urspruenglichen Trainings-Argumente
        # (inkl. data/epochs/time/...) -- deshalb hier bewusst NICHTS davon
        # erneut uebergeben, nur resume=True. Ultralytics haengt automatisch
        # im selben Ausgabeordner (--project/--name aus dem unterbrochenen
        # Lauf) an der naechsten Epoche weiter an.
        print(f"[stage2] Setze unterbrochenen Lauf fort: {args.resume}")
        model = YOLO(args.resume)
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage2"),
            )
        model.train(resume=True)
    else:
        if not args.data:
            ap.error("--data ist erforderlich, ausser bei --resume")

        try:
            batch = int(args.batch)
        except ValueError:
            batch = float(args.batch)

        model = YOLO(args.model)
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage2"),
            )

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

    # Val-Statistiken in einen eigenen Ordner neben dem Trainingslauf (<name>_val)
    # schreiben, statt dass Ultralytics sie nach runs/detect/val legt.
    train_dir = model.trainer.save_dir
    metrics = model.val(project=str(train_dir.parent), name=train_dir.name + "_val")
    print("\n[stage2] Validierungsergebnisse:")
    print(metrics)

    # WICHTIG: den tatsaechlich verwendeten Ausgabeordner direkt vom Trainer
    # erfragen (model.trainer.save_dir), statt ihn selbst aus --project/--name
    # zusammenzubauen. Ultralytics haengt je nach Version noch "runs/detect/"
    # davor -- der wirkliche Pfad steht zuverlaessig nur hier.
    save_dir = model.trainer.save_dir
    print(
        f"\n[stage2] Fertig. Bestes Modell liegt unter: "
        f"{save_dir / 'weights' / 'best.pt'}\n"
        f"[stage2] Letzter Trainingsstand: {save_dir / 'weights' / 'last.pt'}\n"
        f"[stage2] Verlauf pro Epoche: {save_dir / 'results.csv'}"
    )


if __name__ == "__main__":
    main()
