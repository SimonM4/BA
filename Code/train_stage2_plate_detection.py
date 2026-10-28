#!/usr/bin/env python3
"""
train_stage2_plate_detection.py  --  TRAINING STUFE 2: Kennzeichen finden

===========================================================================
WOZU GIBT ES STUFE 2?
===========================================================================
  Stufe 1  ganzes Drohnenbild  ->  Fahrzeug-Boxen
  Stufe 2  Bild/Fahrzeug       ->  Kennzeichen-Box        (dieses Skript)
  Stufe 3  Kennzeichen         ->  Text, z.B. "LIP-YE 351"

Stufe 2 lernt, WO im Bild ein Kennzeichen ist (eine Box drumherum). Was
draufsteht, ist hier egal; das liest erst Stufe 3. Es gibt deshalb nur
EINE Klasse: "license_plate".

Trainingsdaten: der mit unify_yolo_labels.py zusammengefuehrte Datensatz
(kennzeichen_detection_dataset), vier Quellen, ueberwiegend Bodenaufnahmen
ganzer Autos. Trainiert wird also auf ganzen Bildern, nicht auf
Fahrzeug-Ausschnitten. In der Pipeline laeuft das Modell entweder auf dem
ganzen Bild oder, sobald Stufe 1 existiert, auf den Fahrzeug-Ausschnitten.

===========================================================================
WIE WIRD TRAINIERT?
===========================================================================
Technisch ist das Skript fast identisch mit train_stage1_vehicle_detection.py.
Das ist Absicht: Beides ist Objekterkennung mit YOLO, nur mit anderen Daten
und 1 statt 10 Klassen. Zwei getrennte, kurze Skripte lassen sich in der
Arbeit leichter beschreiben als ein Skript mit Verzweigungen.

1. Modell: YOLO26n (nano, ca. 2,4 Mio. Parameter).

2. Transfer Learning (in der Methodik offen nennen!):
   Start von den COCO-vortrainierten Gewichten models\\yolo26n.pt, NICHT von
   den Stufe-1-Gewichten. Grund: Stufe 1 hat Fahrzeuge im Luftbild gelernt,
   Stufe 2 soll ein kleines Detail auf (meist bodennahen) Fahrzeugbildern
   finden. Eine Verkettung braechte keinen klaren Vorteil und machte die
   Ergebnisse schwerer nachvollziehbar.

3. Ultralytics-Standard-Augmentierung (Mosaic, Skalieren, Verschieben,
   Farbe/Helligkeit, horizontales Spiegeln mit 50 %). Das Spiegeln dreht
   auch die Schrift um. Fuer Stufe 2 ist das unproblematisch, weil hier nur
   die Lage der Box gelernt wird, nicht der Text. (Fuer Stufe 3 waere es
   falsch, dort wird deshalb nicht gespiegelt.)
   Alle tatsaechlich benutzten Werte stehen danach in
   runs_stage2\\<name>\\args.yaml (gehoert in den Anhang der Arbeit).

4. Nach jeder Epoche Messung auf dem val-Split, bestes Modell -> best.pt.

5. Early Stopping nach 20 Epochen ohne Verbesserung (--patience).

Bisheriger Stand: runs_stage2\\plate_detection_first_test wurde nur 6 von
100 Epochen auf der CPU trainiert (8-h-Zeitlimit): mAP50 0,886,
mAP50-95 0,52. Diese Werte sind ein erster Funktionstest, kein Endergebnis.
Der volle Lauf soll auf der Kaggle-GPU erfolgen.

===========================================================================
BEZUG ZUR FORSCHUNGSFRAGE (Aufloesung, Flughoehe)
===========================================================================
--imgsz 640 heisst: Das Bild wird so verkleinert, dass die laengere Seite
640 px hat. Ein Kennzeichen, das in einem 4K-Drohnenbild (3840 px breit)
60 px breit ist, ist danach nur noch ca. 10 px breit, also an der Grenze
dessen, was YOLO noch findet. Genau deshalb ist Stufe 1 sinnvoll (erst
Fahrzeug ausschneiden, dann im Ausschnitt suchen), und deshalb beeinflusst
--imgsz direkt, bis zu welcher Flughoehe die Erkennung funktioniert.

===========================================================================
AUFRUF
===========================================================================
Voraussetzung: pip install ultralytics
Vorher:        python unify_yolo_labels.py --data-dir "C:\\BA\\Daten"

  python train_stage2_plate_detection.py --data "C:\\BA\\Daten\\kennzeichen_detection_dataset\\data.yaml"

Auf Kaggle mit GPU:   ... --epochs 100 --device 0
Zeitlimit:            ... --time 8      (Stunden, ersetzt --epochs)
Fortsetzen:           ... --resume <pfad>/weights/last.pt
                      (alle anderen Parameter werden dann ignoriert; geht
                      auf Kaggle nur innerhalb DERSELBEN Session)

Optional: automatische Zwischensicherung nach Kaggle
  --kaggle-checkpoint-dataset <username>/<dataset-name>
  laedt alle --checkpoint-every Epochen (Standard 10) best.pt, last.pt und
  results.csv als neue Dataset-Version hoch. Den Kaggle-API-Token NIE in
  Code, Notebook-Zelle oder Chat schreiben, sondern im Kaggle-Notebook unter
  "Add-ons -> Secrets" hinterlegen und am Anfang des Notebooks laden:
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
# Ergebnisse landen in C:\BA\runs_stage2 (nicht in Code\), damit der Ordner
# Code\ nur Quelltext enthaelt. Der Pfad ist ABSOLUT, sonst haengt
# Ultralytics selbst noch "runs/detect/" davor.
DEFAULT_PROJECT = str(Path(__file__).resolve().parent.parent / "runs_stage2")

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
        # -------------------------------------------------------------------
        # FALL A: unterbrochenes Training fortsetzen
        # -------------------------------------------------------------------
        # last.pt enthaelt alles, was man zum Weitermachen braucht: die
        # Gewichte, den Zustand des Optimierers und die urspruenglichen
        # Einstellungen (data, epochs, imgsz, ...). Deshalb wird hier nur
        # resume=True uebergeben. Ultralytics macht dann an der naechsten
        # Epoche im selben Ausgabeordner weiter.
        print(f"[stage2] Setze unterbrochenen Lauf fort: {args.resume}")
        model = YOLO(args.resume)
        if args.kaggle_checkpoint_dataset:
            model.add_callback(
                "on_model_save",
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage2"),
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
                _make_kaggle_checkpoint_callback(args.kaggle_checkpoint_dataset, args.checkpoint_every, "stage2"),
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
    print("\n[stage2] Validierungsergebnisse:")
    print(metrics)

    # Den TATSAECHLICHEN Ausgabeordner beim Trainer abfragen, statt ihn aus
    # --project/--name zusammenzubauen. Ultralytics haengt bei einem zweiten
    # Lauf mit gleichem Namen z.B. "2" an (plate_detection2).
    save_dir = model.trainer.save_dir
    print(
        f"\n[stage2] Fertig. Bestes Modell liegt unter: "
        f"{save_dir / 'weights' / 'best.pt'}\n"
        f"[stage2] Letzter Trainingsstand: {save_dir / 'weights' / 'last.pt'}\n"
        f"[stage2] Verlauf pro Epoche: {save_dir / 'results.csv'}"
    )


if __name__ == "__main__":
    main()
