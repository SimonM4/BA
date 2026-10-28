#!/usr/bin/env python3
"""
poll_and_download_kaggle_output.py

Wartet, bis ein Kaggle-Notebook-Lauf ("Save & Run All" / Commit) fertig ist,
und laedt danach automatisch alle Output-Dateien (Gewichte best.pt/last.pt,
results.csv, Logs, ...) in einen lokalen Ordner herunter.

WICHTIG -- wo dieses Skript laufen muss:
Es muss auf DEINEM PC in einem normalen Terminal laufen (PowerShell/cmd),
NICHT in der Kaggle-Notebook-Zelle selbst. Es braucht lokal installierte
und authentifizierte Kaggle-CLI:
    pip install kaggle
und entweder eine Datei %USERPROFILE%\\.kaggle\\kaggle.json (Kaggle-Website ->
Account -> "Create New Token") oder die Umgebungsvariable KAGGLE_API_TOKEN.
Trage den Token NIRGENDS in dieses Skript, eine Chat-Nachricht oder eine
Notebook-Zelle ein -- nur in die kaggle.json-Datei bzw. als Windows-
Umgebungsvariable auf deinem eigenen PC.

WICHTIG -- Unterschied zum eigentlichen Kaggle-Training:
Das Training selbst laeuft auf Kaggles Servern und ist komplett unabhaengig
von deinem PC (du kannst deinen PC dafuer ausschalten). DIESES Skript hier
ist anders: es fragt von deinem PC aus periodisch den Status ab und muss
deshalb selbst laufen, solange du auf das Ergebnis wartest -- dein PC muss
also an und wach bleiben, waehrend DIESES Skript laeuft. Wenn du den PC
zwischendurch ausschaltest, stoppt nur das Warten/Herunterladen; das
Training auf Kaggle laeuft unbeeinflusst weiter. Du kannst das Skript
jederzeit spaeter erneut starten -- es prueft einfach den aktuellen Stand
und lädt herunter, sobald der Lauf fertig ist.

Verhindert zusaetzlich (auf Windows), dass der PC waehrend des Wartens in
den Standby geht -- trotzdem: Anmeldebildschirm-Sperre/Energiesparplan am
besten vorher selbst kurz pruefen, falls es trotzdem passiert.

Aufruf (Standardwerte passen fuer dieses Projekt):
  python poll_and_download_kaggle_output.py

Mit eigenen Einstellungen:
  python poll_and_download_kaggle_output.py --kernel simon1543/batraining --dest C:\\BA\\kaggle_output --interval 120 --timeout-hours 13
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Status-Strings, bei denen der Kaggle-Lauf FERTIG ist (egal ob erfolgreich
# oder nicht) -- dann wird das Warten beendet und ein Download versucht.
TERMINAL_STATUSES = {"complete", "error", "cancelacknowledged"}


def log(msg: str, log_file: Optional[Path] = None) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # Logdatei ist nur ein Komfort-Extra, darf nie das Skript stoppen.


def prevent_sleep() -> None:
    """Verhindert unter Windows, dass der PC waehrend des Wartens einschlaeft."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        pass  # Falls das fehlschlaegt, laeuft das Skript trotzdem normal weiter.


def allow_sleep_again() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def check_kaggle_cli_available() -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "FEHLER: Das Kommando 'kaggle' wurde nicht gefunden.\n"
            "Installiere die Kaggle-CLI lokal mit:  pip install kaggle\n"
            "und richte deine Zugangsdaten ein (kaggle.json oder KAGGLE_API_TOKEN),\n"
            "bevor du dieses Skript erneut startest."
        )


def get_status(kernel: str) -> Tuple[Optional[str], str]:
    """Fragt den Status per 'kaggle kernels status' ab.

    Rueckgabe: (status_string_lowercase_oder_None, roher_output_fuer_die_anzeige)
    """
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, "Zeitueberschreitung beim Abfragen des Status (Netzwerk?)."
    except FileNotFoundError:
        sys.exit("FEHLER: 'kaggle' CLI nicht gefunden. Siehe Hinweis im Skript-Kopf (pip install kaggle).")

    raw = (result.stdout or "") + (result.stderr or "")
    raw = raw.strip()

    match = re.search(r'has status "([^"]+)"', raw)
    if match:
        # Neuere Kaggle-CLI-Versionen geben den Status als Enum-Namen aus,
        # z.B. "KernelWorkerStatus.COMPLETE" statt "complete" -> Praefix abschneiden.
        status = match.group(1).lower().rsplit(".", 1)[-1]
        return status, raw
    return None, raw


def download_output(kernel: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel, "-p", str(dest), "-o"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True


def main() -> None:
    default_dest = Path(__file__).resolve().parent.parent / "kaggle_output"

    ap = argparse.ArgumentParser(description="Wartet auf einen Kaggle-Kernel-Lauf und laedt danach dessen Output herunter.")
    ap.add_argument("--kernel", default="simon1543/batraining", help="Kaggle-Kernel-Slug im Format 'username/kernel-name' (Standard: simon1543/batraining).")
    ap.add_argument("--dest", default=None, help=f"Zielordner fuer die heruntergeladenen Dateien (Standard: {default_dest}\\<kernel-name>).")
    ap.add_argument("--interval", type=int, default=120, help="Sekunden zwischen den Status-Abfragen (Standard: 120).")
    ap.add_argument("--timeout-hours", type=float, default=13.0, help="Nach so vielen Stunden ohne Ergebnis aufgeben (Standard: 13, etwas ueber Kaggles Sessionlimit).")
    args = ap.parse_args()

    kernel_name = args.kernel.split("/")[-1]
    dest = Path(args.dest) if args.dest else default_dest / kernel_name
    log_file = dest.parent / f"{kernel_name}_poll.log"
    dest.parent.mkdir(parents=True, exist_ok=True)

    check_kaggle_cli_available()

    log(f"Beobachte Kaggle-Kernel '{args.kernel}'. Ziel: {dest}", log_file)
    log(f"Abfrage alle {args.interval}s, Abbruch nach {args.timeout_hours}h ohne Ergebnis.", log_file)

    prevent_sleep()
    start = time.monotonic()
    consecutive_failures = 0

    try:
        while True:
            elapsed_h = (time.monotonic() - start) / 3600
            if elapsed_h > args.timeout_hours:
                log(f"Zeitlimit von {args.timeout_hours}h erreicht, ohne dass der Lauf fertig wurde. Breche ab.", log_file)
                log("Du kannst den aktuellen Stand manuell pruefen mit: kaggle kernels status " + args.kernel, log_file)
                sys.exit(1)

            status, raw = get_status(args.kernel)

            if status is None:
                consecutive_failures += 1
                log(f"Konnte Status nicht auslesen ({consecutive_failures}. Versuch in Folge): {raw}", log_file)
                if consecutive_failures >= 20:
                    log("Zu viele fehlgeschlagene Status-Abfragen in Folge. Breche ab -- pruefe Internetverbindung/Kaggle-Login.", log_file)
                    sys.exit(1)
                time.sleep(args.interval)
                continue

            consecutive_failures = 0

            if status not in TERMINAL_STATUSES:
                log(f"Status: {status} -- laeuft noch, warte {args.interval}s ...", log_file)
                time.sleep(args.interval)
                continue

            # Terminal-Status erreicht.
            if status == "complete":
                log("Lauf ist fertig (complete). Lade Output herunter ...", log_file)
            elif status == "error":
                log("Lauf ist mit einem FEHLER beendet (error). Pruefe das Notebook-Log auf Kaggle! Lade trotzdem vorhandenen Output herunter ...", log_file)
            else:
                log(f"Lauf wurde abgebrochen (Status: {status}). Lade trotzdem vorhandenen Output herunter ...", log_file)

            ok = download_output(args.kernel, dest)
            if ok:
                files = sorted(p.relative_to(dest) for p in dest.rglob("*") if p.is_file())
                log(f"Download abgeschlossen. {len(files)} Datei(en) in {dest}:", log_file)
                for f in files:
                    log(f"  - {f}", log_file)
            else:
                log("Download ist fehlgeschlagen -- siehe Fehlermeldung oben.", log_file)
                sys.exit(1)
            break
    except KeyboardInterrupt:
        log("Manuell abgebrochen (Strg+C).", log_file)
    finally:
        allow_sleep_again()


if __name__ == "__main__":
    main()
