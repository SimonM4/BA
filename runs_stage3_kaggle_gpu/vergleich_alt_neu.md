# Vergleich Stufe 3: altes vs. neues Zeichen-CNN (28.10.2026)

- ALT: `C:\BA\runs_stage3\char_ocr_model.pt` (CPU, 15 Epochen)
- NEU: `C:\BA\runs_stage3_kaggle_gpu\char_ocr_model.pt` (Kaggle GPU T4, 60 Epochen, beste Epoche 57)
- Testdaten: nur deutsche Kennzeichen, 13 Testbilder mit Klartext (`Daten\kennzeichen_text_gt.csv`)
- Ausgefuehrt im Kaggle-Notebook notebook967f00d352 (ohne GPU), Stufe 2 = `runs_stage2\plate_detection\weights\best.pt`

## Validierung (einzelne Zeichenbilder, aus dem Training)

| | ALT | NEU |
|---|---|---|
| Val sauber | 99,8 % | 99,9 % |
| Val mit Drohnen-Augmentierung | 99,0 % | 99,6 % |
| "kein Zeichen" (Negativbeispiele) | 95,8 % | 97,6 % |

## Echte deutsche Kennzeichen (13 Stueck)

| | ALT | NEU |
|---|---|---|
| Stufe 3 mit perfekter Box: exakt | 5/13 (38 %) | 4/13 (31 %) |
| Stufe 3 mit perfekter Box: CER | 19,0 % | 24,0 % |
| Ganze Pipeline (Stufe 2 + 3): exakt | 3/13 | 3/13 |

Beispielbild roboflow_001121.jpg (wahr: LIP-YE 351): Pipeline ALT "LIP-KE 35", NEU "LIP-YE 35".

## Einordnung

- Auf den Validierungsbildern ist NEU besser, auf echten Kennzeichen nicht. Der Unterschied
  (1 Kennzeichen, ca. 5 Zeichen) liegt bei n = 13 im Rauschen; eine Verbesserung ist nicht nachweisbar.
- Typischer Fehler bei NEU: das schmale "I" (und teils "L") am Anfang von "LIP" wird als
  "kein Zeichen" verworfen (z.B. roh "L_PT5555" -> "P-T 5555"). Die staerker gelernte
  Klasse "_" (97,6 % statt 95,8 %) geht hier zulasten schmaler Zeichen.
- Engpass bleibt die Segmentierung und der Unterschied zwischen synthetischen Trainingszeichen
  und echten Kennzeichen, nicht die Trainingsdauer.
- Empfehlung: ALT bleibt Standard in der Pipeline (`runs_stage3`). Fuer belastbare Aussagen
  mehr Kennzeichen mit Klartext annotieren.
