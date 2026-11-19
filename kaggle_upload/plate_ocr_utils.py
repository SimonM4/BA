#!/usr/bin/env python3
"""
plate_ocr_utils.py

Gemeinsame Hilfsfunktionen fuer STUFE 3 (Zeichenerkennung), die sowohl beim
Training (train_stage3_char_ocr.py) als auch bei der Inferenz
(analyze_video.py) benutzt werden. Wichtig: Training und Inferenz muessen die
Zeichen-Bilder EXAKT gleich vorverarbeiten, sonst sieht das Netz im Einsatz
andere Bilder als im Training -- deshalb liegt das hier an einer Stelle.

Enthaelt nur numpy/OpenCV-Code (kein torch), damit es auch ohne GPU-/Torch-
Installation testbar ist.

Inhalt:
  1. normalize_char_image():   Zeichen-Ausschnitt -> einheitliches Graustufen-
                               bild (dunkles Zeichen auf hellem Grund, feste
                               Groesse). Gleicht die zwei sehr unterschiedlichen
                               Trainingsquellen an (archive(4): echte graue
                               Ausschnitte; DeutschEinzelneZahlen: weisses
                               Zeichen auf schwarzem Grund).
  2. segment_characters():     Kennzeichen-Crop aus Stufe 2 -> Liste von
                               Zeichen-Boxen (klassische Bildverarbeitung:
                               adaptive Binarisierung + Connected Components +
                               geometrische Filter). Bewusst ohne weiteres
                               neuronales Netz -- gut erklaerbar in der Arbeit.
  3. decode_german_plate():    Zeichen-Wahrscheinlichkeiten des CNN + Luecken
                               -> "LIP-YE 351": syntaxgestuetzte Dekodierung
                               gegen die erlaubten deutschen Kennzeichenformate.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Seitenverhaeltnis (Breite/Hoehe) der Zeichen-Bilder in den Trainingsdaten
# (75x100 bzw. 80x110). Ausschnitte bei der Inferenz werden auf dasselbe
# Verhaeltnis gepolstert, bevor sie auf CHAR_SIZE skaliert werden.
CHAR_ASPECT = 0.75
CHAR_SIZE = 48  # Netz-Eingabe: CHAR_SIZE x CHAR_SIZE, 1 Kanal
CHAR_FILL_MIN_ASPECT = 0.45  # schmale Zeichen (I, 1) nur bis zu diesem Verhaeltnis auffuellen


# ---------------------------------------------------------------------------
# 1. Vorverarbeitung einzelner Zeichen
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_char_image(img: np.ndarray, size: int = CHAR_SIZE, auto_polarity: bool = True) -> np.ndarray:
    """Zeichen-Ausschnitt (BGR oder Graustufen, beliebige Groesse) ->
    uint8-Graustufenbild size x size, dunkles Zeichen auf hellem Grund,
    Kontrast auf 0..255 gestreckt.

    auto_polarity=True (Training): Polaritaet automatisch bestimmen (die
    Quelle DeutschEinzelneZahlen ist weiss auf schwarz). Bei der Inferenz ist
    die Polaritaet aus der Segmentierung schon bekannt -> False, denn
    angeschnittene Nachbarzeichen am Rand koennen die Automatik taeuschen.

    WARUM diese Funktion so wichtig ist: Sie laeuft beim Training (auf jedem
    Trainingsbild) UND im Einsatz (auf jedem Zeichen vom Kennzeichen). Nur
    wenn beide Male exakt gleich vorverarbeitet wird, sieht das Netz im
    Einsatz dieselbe Art von Bild, die es gelernt hat.

    Die drei Schritte:
      1. Graustufen + auf size x size (48 x 48) skalieren
      2. Polaritaet vereinheitlichen: immer dunkles Zeichen auf hellem Grund
      3. Kontrast strecken: dunkelste Pixel -> 0, hellste -> 255"""
    # Schritt 1: Farbe entfernen (Zeichenform reicht) und auf Netzgroesse bringen.
    g = to_gray(img)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    if auto_polarity:
        # Schritt 2: Ist das Zeichen hell auf dunkel, Bild invertieren.
        # Otsu waehlt automatisch die Schwelle, die Zeichen und Hintergrund
        # am besten trennt (-> Schwarz-Weiss-Bild bw). Dann wird geprueft,
        # ob die obersten und untersten 2 Zeilen ueberwiegend weiss sind.
        # Diese Zeilen sind fast immer Hintergrund (links/rechts koennen
        # dagegen Reste von Nachbarzeichen stehen). Sind sie ueberwiegend
        # schwarz, ist der Hintergrund dunkel -> invertieren (255 - g).
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([bw[:2, :].ravel(), bw[-2:, :].ravel()])
        if np.mean(border > 0) < 0.5:
            g = 255 - g
    # Schritt 3: Kontrast strecken. Statt des absolut dunkelsten/hellsten
    # Pixels werden das 2-%- und das 98-%-Perzentil genommen. So koennen
    # einzelne Ausreisser (ein Rauschpixel, ein Glanzpunkt) die Streckung
    # nicht verderben. Ergebnis: flaue Drohnenbilder und kontrastreiche
    # Trainingsbilder landen auf derselben Helligkeitsskala.
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi - lo < 1:
        hi = lo + 1   # einfarbiges Bild: Division durch 0 vermeiden
    g = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return g


# ---------------------------------------------------------------------------
# 2. Segmentierung des Kennzeichens in einzelne Zeichen
# ---------------------------------------------------------------------------

@dataclass
class CharBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self):
        return self.x + self.w

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


PLATE_WORK_HEIGHT = 80  # Crops werden fuer die Segmentierung auf diese Hoehe skaliert


def _blue_mask(bgr: np.ndarray) -> np.ndarray:
    """Maske fuer das blaue EU-Feld (wird bei der Segmentierung ignoriert).
    Nur EINE zusammenhaengende, hohe, schmale blaue Flaeche am linken Rand
    zaehlt -- sonst wuerden z.B. blaeulich-dunkle Zeichen im Schatten
    faelschlich ausgeblendet."""
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 120, 60), (130, 255, 255))
    m[:, int(0.22 * W):] = 0
    out = np.zeros_like(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if h >= 0.35 * H and w <= 0.2 * W and area >= 0.3 * w * h:
            out[lab == i] = 255
    # etwas erweitern, damit der dunkle Uebergang EU-Feld/weisser Grund nicht
    # mit dem ersten Buchstaben verschmilzt
    return cv2.dilate(out, np.ones((5, 5), np.uint8))


def _binarize(gray: np.ndarray, block: int, c: int) -> np.ndarray:
    """Zeichen = weiss (255) im Ergebnis. Erwartet dunkle Schrift auf hellem
    Grund (Standard bei deutschen Kennzeichen)."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)


def _remove_frame_lines(binary: np.ndarray) -> np.ndarray:
    """Lange waagerechte/senkrechte Linien (Kennzeichenrahmen, Stossstange)
    entfernen, damit sie nicht mit Zeichen zu einer Komponente verschmelzen."""
    H, W = binary.shape
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 6), 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.72 * H))))
    return cv2.subtract(binary, cv2.bitwise_or(horiz, vert))


def _candidate_boxes(binary: np.ndarray) -> List[CharBox]:
    H, W = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.2 * H or h > 0.9 * H or h < 8:
            continue
        if w < 2 or w > 1.3 * h:
            continue
        fill = area / float(w * h)
        if fill < 0.12 or fill > 0.92:
            continue
        # Komponenten, die oben UND unten am Rand kleben, sind meist Rahmen
        if y <= 1 and y + h >= H - 1:
            continue
        boxes.append(CharBox(int(x), int(y), int(w), int(h)))
    return boxes


def _select_text_line(boxes: List[CharBox]) -> List[CharBox]:
    """Sucht die groesste Gruppe von Boxen, die eine gemeinsame Textzeile
    bilden: aehnliche Hoehe (+-25 %), gleiche vertikale Lage und ohne grosse
    horizontale Spruenge zwischen Nachbarn. Stoerkomponenten (Rahmen,
    Stossstange, Schmutz, Plakette) fallen so heraus."""
    if len(boxes) < 2:
        return boxes
    best: List[CharBox] = []
    best_key = (0, 0.0)
    for seed in boxes:
        hs = seed.h
        grp = [b for b in boxes if abs(b.h - hs) <= 0.2 * hs and abs(b.cy - seed.cy) <= 0.2 * hs]
        grp.sort(key=lambda b: b.x)
        # in zusammenhaengende Ketten aufteilen (Luecke < 1.6 * Zeichenhoehe)
        chains, cur = [], [grp[0]]
        for b in grp[1:]:
            if b.x - cur[-1].x2 < 1.6 * hs:
                cur.append(b)
            else:
                chains.append(cur)
                cur = [b]
        chains.append(cur)
        for ch in chains:
            # Gruppen mit mehr Zeichen bevorzugen, bei Gleichstand die hoeheren
            key = (min(len(ch), 9), float(np.median([b.h for b in ch])))
            if key > best_key:
                best, best_key = ch, key
    return best


def _split_wide(boxes: List[CharBox], binary: np.ndarray) -> List[CharBox]:
    """Zusammengewachsene Zeichen (eine Komponente, die deutlich breiter als
    ein normales Zeichen ist) an der duennsten Spalte auftrennen."""
    if not boxes:
        return boxes
    mh = float(np.median([b.h for b in boxes]))
    # typische Zeichenbreite: Median der Breiten (schmale Zeichen wie I/1
    # ausgenommen), begrenzt auf den fuer FE-Schrift plausiblen Bereich
    ws = [b.w for b in boxes if 0.3 * mh < b.w < 0.85 * mh]
    typical_w = float(np.clip(np.median(ws) if ws else 0.55 * mh, 0.45 * mh, 0.68 * mh))
    out = []
    for b in boxes:
        n_parts = int(round(b.w / typical_w))
        # breiter als ein einzelnes Zeichen sein kann (auch M/W < 0.85*h)
        if b.w > max(1.45 * typical_w, 0.85 * b.h) and n_parts >= 2:
            col = binary[b.y:b.y + b.h, b.x:b.x2].sum(axis=0).astype(np.float32)
            cuts = []
            for k in range(1, n_parts):
                center = int(k * b.w / n_parts)
                lo, hi = max(1, center - int(0.2 * typical_w)), min(b.w - 1, center + int(0.2 * typical_w))
                cuts.append(lo + int(np.argmin(col[lo:hi])) if hi > lo else center)
            edges = [0] + cuts + [b.w]
            for a, e in zip(edges[:-1], edges[1:]):
                if e - a >= 2:
                    out.append(CharBox(b.x + a, b.y, e - a, b.h))
        else:
            out.append(b)
    return out


def _score(boxes: List[CharBox]) -> float:
    """Bewertung einer Segmentierung: deutsche Kennzeichen haben 4-8 Zeichen
    mit sehr aehnlicher Hoehe."""
    n = len(boxes)
    if n == 0:
        return -1.0
    hs = np.array([b.h for b in boxes], dtype=np.float32)
    height_consistency = 1.0 - float(np.std(hs) / (np.mean(hs) + 1e-6))
    count_score = 1.0 if 4 <= n <= 8 else max(0.0, 1.0 - 0.3 * min(abs(n - 4), abs(n - 8)))
    return count_score * 2 + height_consistency + 0.05 * min(n, 8)


def _segment_candidates(bgr: np.ndarray):
    """Alle Segmentierungs-Kandidaten (verschiedene Binarisierungs-Parameter
    und beide Polaritaeten) -> Liste (heuristischer Score, Boxen, Graustufenbild
    dunkel-auf-hell). Doppelte Kandidaten werden entfernt."""
    gray = to_gray(bgr)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 8)).apply(gray)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    blue = _blue_mask(bgr)
    H = gray.shape[0]
    out, seen = [], set()
    for polarity in (False, True):  # True: helle Schrift auf dunklem Grund
        g = 255 - gray_blur if polarity else gray_blur
        for block in (int(H * 0.6) | 1, int(H * 1.0) | 1, int(H * 1.6) | 1):
            for c in (5, 10, 15):
                binary = _binarize(g, block, c)
                binary[blue > 0] = 0
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                binary = _remove_frame_lines(binary)
                boxes = _select_text_line(_candidate_boxes(binary))
                boxes = _split_wide(boxes, binary)
                boxes.sort(key=lambda b: b.x)
                key = (polarity, tuple((b.x // 3, b.w // 3) for b in boxes))
                if not boxes or key in seen:
                    continue
                seen.add(key)
                sc = _score(boxes) - (0.3 if polarity else 0.0)
                out.append((sc, boxes, 255 - gray if polarity else gray))
    out.sort(key=lambda t: -t[0])
    return out


def segment_characters(plate_bgr: np.ndarray, scorer=None, max_candidates: int = 6
                       ) -> Tuple[List[np.ndarray], List[CharBox], np.ndarray]:
    """Kennzeichen-Crop (BGR) -> (Liste Zeichenbilder (Graustufen, dunkel auf
    hell), Liste CharBox (Koordinaten im hochskalierten Crop), hochskalierter
    Graustufen-Crop).

    Probiert mehrere Binarisierungs-Parameter (und eine Schraeglagenkorrektur).
    Ohne `scorer` gewinnt die geometrisch plausibelste Segmentierung (_score).
    Mit `scorer` (Funktion: Liste Zeichenbilder, Boxen -> Zahl, hoeher = besser;
    in der Pipeline: Konfidenz der Kennzeichen-Dekodierung mit dem CNN) werden
    die besten `max_candidates` Kandidaten vom Erkenner selbst bewertet
    ("erkennungsgestuetzte Segmentierung")."""
    if plate_bgr is None or plate_bgr.size == 0 or plate_bgr.shape[0] < 5 or plate_bgr.shape[1] < 10:
        return [], [], np.zeros((1, 1), np.uint8)

    # Detektions-Boxen sind oft sehr knapp (Zeichen beruehren den Rand) ->
    # etwas Rand anfuegen, damit die Zeichen als eigene Komponenten erscheinen.
    # Konstanter Rand in der typischen Randfarbe (nicht BORDER_REPLICATE: das
    # verschmiert Randpixel zu Streifen, die wie Zeichen aussehen koennen).
    ph, pw = plate_bgr.shape[:2]
    edge = np.concatenate([plate_bgr[0], plate_bgr[-1], plate_bgr[:, 0], plate_bgr[:, -1]])
    fill = [int(v) for v in np.median(edge, axis=0)]
    plate_bgr = cv2.copyMakeBorder(plate_bgr, max(2, int(0.12 * ph)), max(2, int(0.12 * ph)),
                                   max(2, int(0.04 * pw)), max(2, int(0.04 * pw)), cv2.BORDER_CONSTANT, value=fill)
    scale = PLATE_WORK_HEIGHT / plate_bgr.shape[0]
    bgr = cv2.resize(plate_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    cands = _segment_candidates(bgr)
    if not cands:
        return [], [], to_gray(bgr)
    # Schraeglage korrigieren: Gerade durch die Zeichen-Mittelpunkte des
    # besten Kandidaten legen, Bild drehen und erneut segmentieren.
    best = cands[0][1]
    if len(best) >= 3:
        xs = np.array([b.cx for b in best]); ys = np.array([b.cy for b in best])
        angle = float(np.degrees(np.arctan(float(np.polyfit(xs, ys, 1)[0]))))
        if 1.5 < abs(angle) < 25:
            Hh, Ww = bgr.shape[:2]
            M = cv2.getRotationMatrix2D((Ww / 2, Hh / 2), angle, 1.0)
            rot = cv2.warpAffine(bgr, M, (Ww, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
            # gedrehte Kandidaten bekommen einen kleinen Bonus (Schraeglage behoben)
            cands += [(sc + 0.02, bx, g) for sc, bx, g in _segment_candidates(rot)]
            cands.sort(key=lambda t: -t[0])

    if scorer is None:
        _, boxes, gray = cands[0]
        return [crop_char(gray, b) for b in boxes], boxes, gray

    best_val, best_res = None, None
    for _, boxes, gray in cands[:max_candidates]:
        chars = [crop_char(gray, b) for b in boxes]
        val = scorer(chars, boxes)
        if best_val is None or val > best_val:
            best_val, best_res = val, (chars, boxes, gray)
    return best_res


def crop_char(gray: np.ndarray, b: CharBox) -> np.ndarray:
    """Zeichen aus dem Graustufen-Crop (dunkel auf hell) schneiden: knapper
    Rand, dann mit HINTERGRUNDFARBE auf das Seitenverhaeltnis der
    Trainingsbilder (CHAR_ASPECT) auffuellen. Bewusst nicht mit den
    Nachbarpixeln auffuellen: angeschnittene Nachbarzeichen links/rechts
    sehen fuer das Netz sonst wie die Balken eines "H" aus."""
    H, W = gray.shape
    my = int(round(0.08 * b.h))
    mx = max(1, int(round(0.06 * b.h)))
    x0, x1 = max(0, b.x - mx), min(W, b.x2 + mx)
    y0, y1 = max(0, b.y - my), min(H, b.y + b.h + my)
    c = gray[y0:y1, x0:x1]
    if c.size == 0:
        return np.full((int(b.h / CHAR_ASPECT), b.h), 255, np.uint8)
    bg = int(np.percentile(c, 90))
    h, w = c.shape
    # Die Trainingszeichen fuellen ihr Bild in der Breite fast ganz aus;
    # Kennzeichen-Schrift (FE-Schrift) ist dagegen schmal. Deshalb wird der
    # knappe Ausschnitt nur bis CHAR_FILL_MIN_ASPECT aufgefuellt und danach
    # (beim Skalieren auf CHAR_SIZE x CHAR_SIZE) in die Breite gezogen.
    target_w = int(round(h * CHAR_FILL_MIN_ASPECT))
    if w < target_w:
        l = (target_w - w) // 2
        c = cv2.copyMakeBorder(c, 0, 0, l, target_w - w - l, cv2.BORDER_CONSTANT, value=bg)
    elif w > h * CHAR_ASPECT:  # breiter als die Trainingsbilder -> oben/unten auffuellen
        target_h = int(round(w / CHAR_ASPECT))
        t = (target_h - h) // 2
        c = cv2.copyMakeBorder(c, t, target_h - h - t, 0, 0, cv2.BORDER_CONSTANT, value=bg)
    return c


def gap_positions(boxes: List[CharBox]) -> List[float]:
    """Luecke zwischen Zeichen i und i+1, relativ zur mittleren Zeichenhoehe."""
    if len(boxes) < 2:
        return []
    mh = float(np.median([b.h for b in boxes]))
    return [(boxes[i + 1].x - boxes[i].x2) / mh for i in range(len(boxes) - 1)]


# ---------------------------------------------------------------------------
# 3. Syntaxgestuetzte Dekodierung deutscher Kennzeichen
# ---------------------------------------------------------------------------
#
# Statt die Einzelvorhersagen nur aneinanderzuhaengen, wird die Zeichenfolge
# gegen die erlaubten Formate deutscher Kennzeichen dekodiert:
#   Standard:      U{1,3}-B{1,2} Z{1,4}[E|H]   (z.B. LIP-YE 351, M-AB 1234E)
#   Bundeswehr:    Y-ZZZ ZZZ                   (z.B. Y-881 695)
#   rote Nummer:   U{1,3}-0Z{3,5}              (z.B. LIP-06431)
# Fuer jedes Format und jede Aufteilung waehlt man je Position das
# wahrscheinlichste ERLAUBTE Zeichen (Buchstabe bzw. Ziffer) aus den
# Softmax-Wahrscheinlichkeiten des CNN. Ganz links/rechts duerfen bis zu zwei
# Ausschnitte als Stoerobjekt ("_": EU-Feld, Plakette, Rahmen) verworfen werden.
# Gewonnen hat die Variante mit der hoechsten mittleren Log-Wahrscheinlichkeit.
# Liefert zusaetzlich einen Konfidenzwert (geometrisches Mittel der
# Zeichenwahrscheinlichkeiten).

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PLAUSIBLE_MIN_CONF = 0.6
DIGITS = set("0123456789")

GERMAN_PLATE_RE = re.compile(r"^(?:[A-Z]{1,3}-[A-Z]{1,2} [1-9][0-9]{0,3}[EH]?|Y-[0-9]{3} [0-9]{3}|[A-Z]{1,3}-0[0-9]{3,5})$")


@dataclass
class PlateReading:
    text: str            # formatiert, z.B. "LIP-YE 351" (leer, wenn nichts lesbar)
    confidence: float    # 0..1, geometrisches Mittel der Zeichenwahrscheinlichkeiten
    plausible: bool      # passt zu einem deutschen Kennzeichenformat
    raw: str             # unformatierte Einzelvorhersagen (Argmax je Ausschnitt)


def _templates(n: int):
    """Alle Formate (Liste von Segmenten (Zeichentyp, Laenge)) mit Gesamtlaenge n."""
    out = []
    for a in (1, 2, 3):
        for b in (1, 2):
            for c in (1, 2, 3, 4):
                if a + b + c == n:
                    out.append(("std", [("L", a), ("L", b), ("D", c)]))
                if a + b + c + 1 == n:
                    out.append(("stdEH", [("L", a), ("L", b), ("D", c), ("EH", 1)]))
        for c in (4, 5, 6):
            if a + c == n:
                out.append(("rot", [("L", a), ("D0", c)]))
    if n == 7:
        out.append(("bw", [("Y", 1), ("D", 6)]))
    return out


def decode_german_plate(probs: np.ndarray, classes: List[str], gaps: Optional[List[float]] = None) -> PlateReading:
    """probs: N x C Softmax-Wahrscheinlichkeiten je Zeichen-Ausschnitt (links->rechts),
    classes: Klassennamen des Modells, gaps: gap_positions() der Ausschnitte."""
    n = len(probs)
    raw = "".join(classes[int(i)] for i in probs.argmax(1)) if n else ""
    if n < 3:
        return PlateReading("", 0.0, False, raw)
    # O/0 und I/1 sind bei geringer Aufloesung praktisch nicht zu
    # unterscheiden (und "O" ist in den Trainingsdaten selten). Deshalb zaehlt
    # an Buchstaben-Positionen die Wahrscheinlichkeit von "0" auch fuer "O"
    # (bzw. "1" fuer "I") und an Ziffern-Positionen umgekehrt.
    probs = probs.copy()
    ci = {c: i for i, c in enumerate(classes)}
    for a_, b_ in (("O", "0"), ("I", "1")):
        if a_ in ci and b_ in ci:
            both = np.maximum(probs[:, ci[a_]], probs[:, ci[b_]])
            probs[:, ci[a_]] = both
            probs[:, ci[b_]] = both
    lp = np.log(np.clip(probs, 1e-6, 1.0))
    L_idx = [i for i, c in enumerate(classes) if c in LETTERS]
    D_idx = [i for i, c in enumerate(classes) if c in DIGITS]
    allowed = {
        "L": L_idx, "D": D_idx,
        "EH": [i for i, c in enumerate(classes) if c in ("E", "H")],
        "Y": [i for i, c in enumerate(classes) if c == "Y"],
        "D0": D_idx,
    }
    nochar = classes.index("_") if "_" in classes else None
    gaps = list(gaps) if gaps else [0.0] * (n - 1)
    med_gap = float(np.median(gaps)) if gaps else 0.0

    best = None
    for dl in range(0, 3):
        for dr in range(0, 3):
            m = n - dl - dr
            if m < 3:
                continue
            # Kosten fuers Verwerfen: wie sicher ist der Ausschnitt "kein Zeichen"?
            drop_cost = 0.0
            for k in list(range(dl)) + list(range(n - dr, n)):
                p_no = probs[k, nochar] if nochar is not None else 0.0
                drop_cost += float(np.log(max(p_no, 0.05)))
            for kind, segs in _templates(m):
                pos, total, chars, bounds = dl, 0.0, [], []
                for typ, length in segs:
                    for j in range(length):
                        idx = allowed[typ]
                        if typ == "D0" and j == 0:
                            idx = [i for i in D_idx if classes[i] == "0"]
                        if not idx:
                            total = -1e9
                            break
                        k = idx[int(np.argmax(lp[pos, idx]))]
                        total += lp[pos, k]
                        chars.append(classes[k])
                        pos += 1
                    bounds.append(pos)
                if total <= -1e8:
                    continue
                mean_lp = total / m
                # Luecken-Bonus: Trennstellen sollten an groesseren Luecken liegen
                bonus = 0.0
                if kind.startswith("std"):
                    gi = bounds[0] - 1  # Luecke zwischen Unterscheidungszeichen und Buchstaben
                    g = gaps[gi] if 0 <= gi < len(gaps) else med_gap
                    bonus += 0.15 * float(np.clip(g - med_gap, -1.0, 1.0))
                score = mean_lp + bonus + drop_cost / n
                if best is None or score > best[0]:
                    best = (score, kind, segs, chars, mean_lp)

    if best is None:
        return PlateReading("", 0.0, False, raw)
    _, kind, segs, chars, mean_lp = best
    s = "".join(chars)
    if kind in ("std", "stdEH"):
        a, b = segs[0][1], segs[1][1]
        text = s[:a] + "-" + s[a:a + b] + " " + s[a + b:]
    elif kind == "rot":
        a = segs[0][1]
        text = s[:a] + "-" + s[a:]
    else:  # Bundeswehr
        text = s[0] + "-" + s[1:4] + " " + s[4:]
    conf = float(np.exp(mean_lp))
    # "plausibel" = passt zu einem Format UND das CNN war sich dabei
    # ausreichend sicher (das Format allein erzwingt die Dekodierung ja immer)
    return PlateReading(text, conf, bool(GERMAN_PLATE_RE.match(text)) and conf >= PLAUSIBLE_MIN_CONF, raw)
