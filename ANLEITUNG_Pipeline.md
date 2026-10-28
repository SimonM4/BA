# Kennzeichenerkennung: Pipeline benutzen

## Video analysieren
- **Drag & Drop:** Eine oder mehrere Videodateien auf `C:\BA\VIDEO_ANALYSIEREN.bat` ziehen.
- **Ordner:** Videos nach `C:\BA\Videos\Eingang` legen und `VIDEO_ANALYSIEREN.bat` doppelklicken.
- **Live-Stream der Drohne:** `LIVESTREAM_ANALYSIEREN.bat` starten und die RTSP-URL eingeben (Fenster mit `q` schließen).

Am Ende öffnet sich der Ergebnisordner `C:\BA\Ergebnisse\<video>_<datum_zeit>\`:

| Datei | Inhalt |
|---|---|
| `kennzeichen.csv` | eine Zeile pro erkanntem Kennzeichen: Text, Konfidenz, Übereinstimmung über die Frames, erste/letzte Sichtung (s), plausibel ja/nein |
| `detektionen.csv` | Rohdaten: jede Kennzeichen-Detektion in jedem analysierten Frame |
| `<video>_annotiert.mp4` | Video mit Boxen und gelesenem Kennzeichen |
| `crops\` | bester Ausschnitt je Kennzeichen + Bild der Zeichen-Segmentierung |
| `zusammenfassung.json` | verwendete Modelle, Parameter, Laufzeit |

CSV-Dateien sind mit `;` getrennt und öffnen sich direkt in Excel.

## Ablauf (für die Arbeit)
1. **Stufe 1 – Fahrzeuge** (optional, YOLO auf VisDrone): Gibt es ein Modell unter `runs_stage1\...\weights\best.pt`, sucht Stufe 2 nur in Fahrzeug-Ausschnitten. Sonst sucht sie im ganzen Bild.
2. **Stufe 2 – Kennzeichen** (YOLO, `runs_stage2\...\weights\best.pt`): findet die Kennzeichen.
3. **Stufe 3 – Zeichen** (`Code\plate_ocr_utils.py`, `Code\char_cnn.py`, Modell `runs_stage3\char_ocr_model.pt`):
   - Segmentierung mit klassischer Bildverarbeitung (adaptive Binarisierung, Connected Components, Textzeilen-Suche, Schräglagenkorrektur),
   - Klassifikation jedes Zeichens mit dem eigenen CNN (0–9, A–Z, „_“ = kein Zeichen),
   - syntaxgestützte Dekodierung gegen deutsche Kennzeichenformate (`LIP-YE 351`, `Y-881 695`, `LIP-06431`).
4. **Tracking und Abstimmung:** Dasselbe Kennzeichen wird über mehrere Frames verfolgt. Die Lesungen werden gewichtet zusammengefasst, sodass einzelne unscharfe Frames das Ergebnis nicht verfälschen.

## Nützliche Optionen (`Code\analyze_video.py`)
- `--stride 2`: nur jeden 2. Frame analysieren (doppelt so schnell)
- `--imgsz 1280`: Eingabegröße für die Kennzeichensuche. Höher hilft bei kleinen Kennzeichen aus großer Höhe, ist aber langsamer.
- `--plate-conf 0.35`: Mindestkonfidenz für Kennzeichen-Detektionen
- `--show`: Live-Fenster
- `--plate-model / --vehicle-model / --ocr-model`: Modelle explizit angeben. Ohne Angabe werden die neuesten trainierten Modelle in `C:\BA` automatisch gefunden.

## Training / Auswertung
- `Start\02_stage3_trainieren.bat`: Stufe 3 neu trainieren (Ausgabe: `runs_stage3\`, inkl. Trainingskurven und Konfusionsmatrix)
- `Start\04_stage3_auf_kennzeichen_testen.bat`: Stufe 3 auf echten Kennzeichen-Ausschnitten bewerten (Ground Truth: `Daten\kennzeichen_text_gt.csv`, von Hand abgelesen. Bitte prüfen und erweitern.)
- `Code\make_test_video.py`: erzeugt ein Testvideo aus Einzelbildern (`Videos\testvideo_kennzeichen.mp4`)
