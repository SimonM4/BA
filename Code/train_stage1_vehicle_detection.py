#!/usr/bin/env python3
"""
train_stage1_vehicle_detection.py  --  TRAINING STUFE 1: Fahrzeuge im Luftbild

===========================================================================
WOZU GIBT ES STUFE 1?
===========================================================================
Die Pipeline arbeitet in drei Stufen, jede mit einem eigenen Modell:

  Stufe 1  ganzes Drohnenbild  ->  Fahrzeug-Boxen         (dieses Skript)
  Stufe 2  Bild/Fahrzeug       ->  Kennzeichen-Box        (train_stage2_...)
  Stufe 3  Kennzeichen         ->  Text, z.B. "LIP-YE 351" (train_stage3_...)

Stufe 1 lernt auf dem VisDrone-Datensatz (echte Drohnenbilder), die 10
VisDrone-Klassen zu finden: pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor. Fuer die Kennzeichenerkennung sind
davon vor allem car, van, truck und bus interessant.

Warum nicht direkt im ganzen Bild nach Kennzeichen suchen?
  Aus der Luft ist ein Kennzeichen winzig (oft nur wenige Pixel hoch).
  Findet Stufe 1 zuerst das Fahrzeug, kann Stufe 2 den Fahrzeug-Ausschnitt
  vergroessert betrachten. Das Kennzeichen ist dann relativ zum Bildausschnitt
  viel groesser und leichter zu finden.

Warum ZWEI getrennte Modelle (Stufe 1 und 2) statt einem gemeinsamen?
  Fahrzeuge im ganzen Luftbild und Kennzeichen auf einem Fahrzeug sind zwei
  verschiedene Groessenordnungen und zwei verschiedene Datensaetze. Zwei
  spezialisierte Modelle lassen sich getrennt trainieren, getrennt bewerten
  und in der Arbeit klar beschreiben.

Aktueller Stand: Ein Stufe-1-Modell gibt es noch NICHT. Die Pipeline
(analyze_video.py) sucht deshalb im ganzen Bild nach Kennzeichen und nutzt
Stufe 1 automatisch, sobald runs_stage1\\...\\weights\\best.pt existiert.

===========================================================================
WIE WIRD TRAINIERT?  (die Methode in Kurzform)
===========================================================================
1. Modell: YOLO26n von Ultralytics. "n" = nano, ca. 2,4 Mio. Parameter.
   Klein genug fuer die kostenlose Kaggle-GPU und spaeter fuer schnelle
   Auswertung von Videos ohne teure Hardware.

2. Transfer Learning (WICHTIG fuer die Arbeit, offen so beschreiben!):
   Das Training startet NICHT bei zufaelligen Gewichten, sondern bei
   Gewichten, die Ultralytics bereits auf dem COCO-Datensatz (allgemeine
   Objekterkennung, 80 Klassen) vortrainiert hat (Datei models\\yolo26n.pt).
   Das Netz kann also schon "Kanten, Formen, Objekte" erkennen und lernt
   auf VisDrone nur noch die Drohnenperspektive und unsere 10 Klassen.
   Die Ausgabeschicht wird dabei automatisch auf 10 Klassen umgebaut.
   Man nennt das Fine-Tuning. Es ist Standard bei begrenzten Daten und
   begrenzter Rechenzeit, muss aber in der Methodik genannt werden.

3. Training mit den Standard-Einstellungen von Ultralytics. Die wichtigsten,
   die automatisch aktiv sind:
     - Mosaic-Augmentierung: 4 Trainingsbilder werden zu einem
       zusammengesetzt. Dadurch sieht das Netz viele kleine Objekte in
       wechselndem Zusammenhang, was zu Luftbildern passt.
     - weitere Zufallsveraenderungen (Spiegeln, Skalieren, Verschieben,
       Farbe/Helligkeit), damit das Netz nicht einzelne Bilder auswendig lernt.
   Die genauen Werte stehen nach dem Training in runs_stage1\\<name>\\args.yaml.
   Diese Datei gehoert in den Anhang der Arbeit (Reproduzierbarkeit).

4. Nach jeder Epoche misst Ultralytics die Qualitaet auf dem val-Split
   (Precision, Recall, mAP50, mAP50-95). Das beste Modell wird als best.pt
   gespeichert, das letzte als last.pt.

5. Early Stopping (--patience 20): Verbessert sich das Modell 20 Epochen
   lang nicht, bricht das Training ab. Das spart GPU-Zeit und verhindert,
   dass man nur noch Overfitting (Auswendiglernen) trainiert.

===========================================================================
EINE UEBERLEGUNG FUER DEINE FORSCHUNGSFRAGE (Aufloesung)
===========================================================================
--imgsz 640 heisst: Jedes Bild wird vor dem Netz so verkleinert, dass die
laengere Seite 640 px hat. Ein 4K-Drohnenbild (3840 px breit) wird also auf
1/6 verkleinert. Ein Fahrzeug, das im Original 60 px breit ist, ist danach
nur noch ca. 10 px breit. Groessere --imgsz (z.B. 1280) helfen bei kleinen
Objekten, kosten aber etwa 4x so viel Rechenzeit. Diese Einstellung ist
damit direkt Teil der Frage "ab welcher Aufloesung funktioniert es?".

===========================================================================
AUFRUF
===========================================================================
Voraussetzung: pip install ultralytics  (siehe requirements.txt)
Vorher:        python prepare_visdrone.py --data-dir "C:\\BA\\Daten"

Erster Testlauf (lokal):
  python train_stage1_vehicle_detection.py --data "C:\\...\\visdrone_yolo\\data.yaml"

Finaler Lauf auf Kaggle mit GPU:
  python train_stage1_vehicle_detection.py --data visdrone_yolo/data.yaml --epochs 100 --device 0

Zeitlich begrenzt (z.B. "ueber Nacht"): --time 8  (Stunden, ersetzt --epochs)

Unterbrochenen Lauf fortsetzen (z.B. Kaggle-Session abgebrochen):
  python train_stage1_vehicle_detection.py --resume <pfad>/weights/last.pt
  Alle anderen Parameter werden dann ignoriert, weil last.pt die
  urspruenglichen Einstellungen selbst enthaelt. Das geht nur, solange
  last.pt noch existiert (auf Kaggle nur innerhalb DERSELBEN Session).

Optional: automatische Zwischensicherung nach Kaggle
  --kaggle-checkpoint-dataset <username>/<dataset-name>
  laedt alle --checkpoint-every Epochen (Standard 10) best.pt, last.pt und
  results.csv als neue Version eines Kaggle-Datasets hoch. Bricht die
  Session ab, verliert man hoechstens die letzten paar Epochen.
  Den Kaggle-API-Token NIE in Code, Notebook-Zelle oder Chat schreiben,
  sondern im Kaggle-Notebook unter "Add-ons -> Secrets" hinterlegen und am
  Anfang des Notebooks laden:
      from kaggle_secrets import UserSecretsClient
      import os
      os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
"""

import argparse
import json
import subprocess
from pathlib import Path

from ultralytics import YOLO  # Bibliothek, die YOLO-Modelle laedt, trainiert und auswertet

# ---------------------------------------------------------------------------
# Feste Pfade
# ---------------------------------------------------------------------------
# Path(__file__).resolve().parent        = C:\BA\Code
# Path(__file__).resolve().parent.parent = C:\BA
#
# Ergebnisse landen in C:\BA\runs_stage1 (nicht in Code\), damit der Ordner
# Code\ nur Quelltext enthaelt. Der Pfad ist ABSOLUT, sonst haengt
# Ultralytics selbst noch "runs/detect/" davor.
DEFAULT_PROJECT = str(Path(__file__).resolve().parent.parent / "runs_stage1")

# Startgewichte (COCO-vortrainiert, siehe Punkt 2 oben). Liegt die Datei in
# C:\BA\models, wird sie benutzt. Sonst laedt Ultralytics sie automatisch
# aus dem Internet herunter (z.B. auf Kaggle).
_LOCAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolo26n.pt"
DEFAULT_MODEL = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "yolo26n.pt"


# ===========================================================================
# HILFSFUNKTION: Zwischenstand nach Kaggle sichern
# ---------------------------------------------------------------------------
# Das ist reine Infrastruktur und gehoert NICHT zur Methode der Arbeit.
# Zum Verstehen des Trainings kannst du diese Funktion ueberspringen und
# direkt bei main() weiterlesen.
# ===========================================================================
def _make_kaggle_checkpoint_callback(dataset_slug: str, checkpoint_every: int, stage_name: str):
    """Baut eine Funktion ("Callback"), die Ultralytics waehrend des
    Trainings automatisch aufruft. Sie laedt alle `checkpoint_every` Epochen
    best.pt, last.pt und results.csv als neue Version eines Kaggle-Datasets
    hoch.

    Was ist ein Callback? Man gibt Ultralytics eine eigene Funktion mit und
    sagt "ruf die bei Ereignis X auf". Hier ist das Ereignis "on_model_save",
    also direkt nachdem best.pt/last.pt auf die Festplatte geschrieben wurden.
    (Bei "on_train_epoch_end" waeren die Dateien noch vom VORHERIGEN
    Speichern, weil Ultralytics erst nach der Validierung speichert.)

    Warum ueberhaupt? Eine Kaggle-Session kann jederzeit ohne Vorwarnung
    enden (Zeitlimit, GPU-Kontingent aufgebraucht). Das Skript bekommt davon
    nichts mit. Der einzige sichere Schutz ist regelmaessiges Sichern
    WAEHREND des Trainings.

    Ein fehlgeschlagener Upload bricht das Training nie ab, es gibt nur
    eine Warnung."""

    # Merkt sich ueber mehrere Aufrufe hinweg, ob das Dataset schon angelegt
    # wurde (beim ersten Mal "create", danach "version").
    state = {"dataset_exists": False}

    def callback(trainer) -> None:
        epoch = trainer.epoch + 1  # trainer.epoch zaehlt ab 0, wir ab 1
        if epoch % checkpoint_every != 0:
            return  # nur jede checkpoint_every-te Epoche hochladen

        weights_dir = trainer.save_dir / "weights"
        if not weights_dir.exists():
            return

        try:
            # Die Kaggle-CLI laedt einen ganzen ORDNER hoch. Deshalb werden die
            # gewuenschten Dateien in einen eigenen Upload-Ordner kopiert.
            ckpt_dir = trainer.save_dir / "_kaggle_checkpoint_upload"
            ckpt_dir.mkdir(exist_ok=True)
            for fname in ("best.pt", "last.pt"):
                src = weights_dir / fname
                if src.exists():
                    (ckpt_dir / fname).write_bytes(src.read_bytes())
            results_csv = trainer.save_dir / "results.csv"
            if results_csv.exists():
                (ckpt_dir / "results.csv").write_bytes(results_csv.read_bytes())

            # Pflichtdatei fuer Kaggle-Datasets: Titel, ID, Lizenz.
            (ckpt_dir / "dataset-metadata.json").write_text(
                json.dumps({
                    "title": f"BA Checkpoint {stage_name}",
                    "id": dataset_slug,
                    "licenses": [{"name": "CC0-1.0"}],
                }),
                encoding="utf-8",
            )

            if not state["dataset_exists"]:
                # Erster Upload dieser Session. Existiert das Dataset schon aus
                # einer frueheren Session, schlaegt "create" fehl. Das ist kein
                # echter Fehler, dann einfach eine neue "version" anlegen.
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
            # Absichtlich JEDEN Fehler abfangen: ein Upload-Problem darf nie
            # ein stundenlanges Training abbrechen.
            print(f"[{stage_name}] Warnung: Checkpoint-Upload (Epoche {epoch}) fehlgeschlagen ({e}), Training laeuft weiter.")

    return callback


# ===========================================================================
# HAUPTPROGRAMM: hier passiert das eigentliche Training
# ===========================================================================
def main() -> None:
    # -----------------------------------------------------------------------
    # SCHRITT 1: Einstellungen von der Kommandozeile lesen
    # -----------------------------------------------------------------------
    # Die wichtigsten Begriffe:
    #   Epoche   = das Netz hat jedes Trainingsbild einmal gesehen
    #   Batch    = so viele Bilder werden gleichzeitig verarbeitet, bevor die
    #              Gewichte einmal angepasst werden (groesser = stabiler,
    #              braucht aber mehr GPU-Speicher)
    #   imgsz    = Bildgroesse, auf die vor dem Netz skaliert wird (siehe oben)
    #   patience = Early Stopping (siehe oben)
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
        # -------------------------------------------------------------------
        # FALL A: unterbrochenes Training fortsetzen
        # -------------------------------------------------------------------
        # last.pt enthaelt alles, was man zum Weitermachen braucht: die
        # Gewichte, den Zustand des Optimierers und die urspruenglichen
        # Einstellungen (data, epochs, imgsz, ...). Deshalb wird hier nur
        # resume=True uebergeben. Ultralytics macht dann an der naechsten
        # Epoche im selben Ausgabeordner weiter.
        print(f"[stage1] Setze unterbrochenen Lauf fort: {args.resume}")
        model = YOLO(args.resume)
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage1"),
            )
        model.train(resume=True)
    else:
        # -------------------------------------------------------------------
        # FALL B: neues Training starten
        # -------------------------------------------------------------------
        if not args.data:
            ap.error("--data ist erforderlich, ausser bei --resume")

        # --batch darf eine ganze Zahl (16 = 16 Bilder pro Schritt) ODER ein
        # Bruch sein (0.7 = so viele Bilder, dass 70 % des GPU-Speichers
        # belegt sind). Deshalb erst als int versuchen, sonst als float.
        try:
            batch = int(args.batch)
        except ValueError:
            batch = float(args.batch)

        # SCHRITT 2: Startmodell laden (COCO-vortrainiert = Transfer Learning,
        # siehe Punkt 2 im Kopf dieser Datei).
        model = YOLO(args.model)
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage1"),
            )

        # SCHRITT 3: Trainieren.
        # Diese eine Zeile erledigt intern die ganze Trainingsschleife:
        #   fuer jede Epoche:
        #     fuer jeden Batch:
        #       Bilder laden + zufaellig veraendern (Augmentierung, z.B. Mosaic)
        #       Vorhersage des Netzes berechnen
        #       Fehler (Loss) gegenueber den Labels berechnen
        #       Gewichte so anpassen, dass der Fehler kleiner wird
        #     auf dem val-Split messen (mAP) und best.pt / last.pt speichern
        #     ggf. Early Stopping
        # Alles, was hier NICHT angegeben ist, laeuft mit den
        # Ultralytics-Standardwerten (stehen danach in args.yaml).
        model.train(
            data=args.data,          # welche Bilder/Labels (data.yaml)
            epochs=args.epochs,      # wie viele Durchlaeufe hoechstens
            time=args.time,          # alternativ: Zeitlimit in Stunden
            imgsz=args.imgsz,        # Eingabegroesse des Netzes
            batch=batch,             # Bilder pro Schritt
            device=args.device,      # GPU oder CPU
            project=args.project,    # Hauptordner fuer Ergebnisse
            name=args.name,          # Unterordner fuer diesen Lauf
            patience=args.patience,  # Early Stopping
        )

    # -----------------------------------------------------------------------
    # SCHRITT 4: Abschliessende Auswertung
    # -----------------------------------------------------------------------
    # model.val() misst Precision, Recall, mAP50 und mAP50-95 auf dem
    # val-Split und speichert Diagramme (PR-Kurve, Konfusionsmatrix) in den
    # Ordner <name>_val neben dem Trainingslauf.
    #
    # Kurz erklaert:
    #   Precision = Anteil der gefundenen Boxen, die wirklich ein Objekt sind
    #   Recall    = Anteil der echten Objekte, die gefunden wurden
    #   mAP50     = mittlere Genauigkeit, wenn eine Box als richtig zaehlt,
    #               sobald sie zu mind. 50 % mit der echten Box ueberlappt (IoU)
    #   mAP50-95  = Mittel ueber strengere Schwellen 50 %, 55 %, ..., 95 %
    #
    # HINWEIS FUER DIE ARBEIT: Der val-Split wurde auch benutzt, um best.pt
    # auszuwaehlen. Die Zahlen hier sind deshalb leicht optimistisch. Fuer
    # das Ergebniskapitel zusaetzlich auf dem test-Split auswerten, z.B.
    #   yolo val model=<...>/best.pt data=<...>/data.yaml split=test
    train_dir = model.trainer.save_dir
    metrics = model.val(project=str(train_dir.parent), name=train_dir.name + "_val")
    print("\n[stage1] Validierungsergebnisse:")
    print(metrics)

    # Den TATSAECHLICHEN Ausgabeordner beim Trainer abfragen, statt ihn aus
    # --project/--name zusammenzubauen. Ultralytics haengt bei einem zweiten
    # Lauf mit gleichem Namen z.B. "2" an (vehicle_detection2).
    save_dir = model.trainer.save_dir
    print(
        f"\n[stage1] Fertig. Bestes Modell liegt unter: "
        f"{save_dir / 'weights' / 'best.pt'}\n"
        f"[stage1] Letzter Trainingsstand: {save_dir / 'weights' / 'last.pt'}\n"
        f"[stage1] Verlauf pro Epoche: {save_dir / 'results.csv'}"
    )


if __name__ == "__main__":
    main()
