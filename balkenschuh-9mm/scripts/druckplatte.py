#!/usr/bin/env python3
"""
Druckplatte Balkenschuh 9 mm: 4 Stück, angewinkelt, mit fertigen Supports
==========================================================================
Anycubic Photon Mono M7 Pro (223 x 126 mm). Je Variante (wand/holz) eine
Platte mit 4 Teilen in 2 x 2.

Ausrichtung: Die lange Achse steht 60° zur Platte (30° aus der Senkrechten
gekippt, Hinweis aus dem Druckprofil: dünne senkrechte Flächen nie nur an
der Unterkante stützen). Wandseite und Boden zeigen schräg nach unten und
tragen die Supports; Tasche, Vorderkanten und Senkungen zeigen nach oben
und bleiben frei von Stützpunkten.

Supports nach Profil „Claude No1“: Kontakt 0,6 mm, Kopf 1,0 mm, Stamm
1,8 mm, Kopf in Flächennormale angesetzt; Stämme in jeder Reihe diagonal
verstrebt, unten auf Raft-Schienen.
"""

import argparse
import os
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Point

import balkenschuh_9mm as bs
from balkenschuh_9mm import kegelstumpf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "werkzeuge"))
import topopt  # noqa: E402

# ----------------------------------------------------------------------------
# Parameter [mm]
# ----------------------------------------------------------------------------
PLATTE = (223.0, 126.0)
THETA = np.radians(60.0)     # Winkel lange Achse zur Platte
LIFT = 6.0                   # tiefster Punkt über der Platte
SPALT = 5.0                  # Abstand zwischen den Teilen

RAFT_H = 0.8                 # Höhe der Raft-Schienen
SCHIENE_B = 3.5              # Breite der Raft-Schienen
KONTAKT_R = 0.3              # Kontaktpunkt Ø 0,6
KOPF_R = 0.5                 # Kopf Ø 1,0
KOPF_L = 2.0                 # Länge Kopf entlang der Flächennormale
EMBED = 0.3                  # Spitze taucht so weit ins Teil
UEBERGANG_L = 2.0            # Kegel Kopf -> Stamm
STAMM_R = 0.9                # Stamm Ø 1,8
FUSS_R = 1.8                 # Fußkegel auf der Schiene
FUSS_H = 1.0
STREBE_R = 0.6               # Diagonalstreben Ø 1,2
STREBE_ABSTAND = 18.0        # Höhenabstand der Streben
RASTER_Z = 11.0              # Stützpunkt-Abstand entlang der Teilhöhe
UEBERHANG = 40.0             # optimierte Form: Schnittflächen flacher als 40°
UEBERHANG_RASTER = 4.0       # zur Waagerechten bekommen Stützpunkte in diesem Abstand

here = os.path.dirname(__file__)
stl_dir = os.path.abspath(os.path.join(here, "..", "stl"))


# ----------------------------------------------------------------------------
# Lage im Druckraum
# ----------------------------------------------------------------------------
def druck_matrix():
    """Konstruktion (x, y, z) -> flach (X = z, Y = x, Z = y) -> um Y gekippt."""
    perm = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    kipp = trimesh.transformations.rotation_matrix(-THETA, [0, 1, 0])
    return kipp @ perm


def stuetzpunkte(variante, teil=None):
    """Kontaktpunkte (Konstruktions-Koordinaten) mit Außennormale. Mit teil
    (optimierte Form) fallen Punkte weg, unter denen kein Material mehr ist."""
    tl = bs.VARIANTEN[variante]["t_lasche"]
    loch_r = (bs.VARIANTEN[variante]["senkung"] or bs.VARIANTEN[variante]["loch"]) / 2
    lp = bs.lasche_profil().buffer(-1.2)
    punkte = []

    # Wandseite (y = 0): drei Reihen je Seite (Wangenkante, innen, außen)
    zs = np.arange(2.5, bs.H - 2.0, RASTER_Z)
    for sx in (1, -1):
        for x in (bs.xi + 1.5, bs.xa + 4.0, bs.xl - 4.0):
            for z in zs:
                if not lp.contains(Point(x, z)) and x > bs.xa:
                    continue
                if any(np.hypot(x - bs.LOCH_X, z - zl) < loch_r + 1.2 for zl in bs.WAND_LOCH_Z):
                    continue
                punkte.append(((sx * x, 0.0, z), (0, -1, 0)))

    # Boden (z = 0): Raster unter dem Taschenboden, dazu die Laschenfüße
    for x in (-3.8, 0.0, 3.8):
        for y in np.linspace(3.0, bs.TIEFE - 2.0, 5):
            punkte.append(((x, y, 0.0), (0, 0, -1)))
    for sx in (1, -1):
        punkte.append(((sx * (bs.xa + 4.0), tl / 2, 0.0), (0, 0, -1)))
    if teil is not None:
        punkte = [pn for pn in punkte if traegt(teil, *pn)]
    return punkte


def traegt(teil, p, n, r=0.8, tiefe=0.4):
    """Ist unter dem Kontaktpunkt (Mitte und Kreis r) Material?"""
    p, n = np.asarray(p, float), np.asarray(n, float)
    a = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    proben = [p - tiefe * n] + [p - tiefe * n + r * (np.cos(w) * a + np.sin(w) * b)
                                for w in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    return topopt.punkte_innen(teil, proben).all()


def kugel(p, r):
    k = trimesh.creation.icosphere(subdivisions=1, radius=r)
    k.apply_translation(p)
    return k


def stuetze(p, n):
    """Ein Support: Kopf entlang der Normale, dann senkrechter Stamm zur Schiene."""
    p, n = np.asarray(p), np.asarray(n)
    q = p + KOPF_L * n
    oben = q - [0, 0, UEBERGANG_L]
    teile = [
        kegelstumpf(p - EMBED * n, q, KONTAKT_R, KOPF_R, 12),
        kugel(q, KOPF_R),
        kegelstumpf(q, oben, KOPF_R, STAMM_R, 12),
        kegelstumpf(oben, (q[0], q[1], RAFT_H + FUSS_H), STAMM_R, STAMM_R, 12),
        kegelstumpf((q[0], q[1], RAFT_H - 0.1), (q[0], q[1], RAFT_H + FUSS_H), FUSS_R, STAMM_R, 12),
    ]
    assert oben[2] > RAFT_H + FUSS_H + 0.5, f"Stamm zu kurz bei {p}"
    return teile, (q[0], q[1], oben[2])


def streben(staemme):
    """Diagonale Zickzack-Streben zwischen Nachbarstämmen (in X- und Y-Richtung)."""
    teile = []

    def verbinde(a, b):
        d = np.hypot(b[0] - a[0], b[1] - a[1])
        if d < 1.0 or d > 13.0:
            return
        top = min(a[2], b[2]) - 1.0
        h, k = RAFT_H + FUSS_H + 2.0, 0
        while h + d < top:
            u, v = (a, b) if k % 2 == 0 else (b, a)
            teile.append(kegelstumpf((u[0], u[1], h), (v[0], v[1], h + d),
                                     STREBE_R, STREBE_R, 8))
            h += STREBE_ABSTAND
            k += 1

    # Reihen gleicher Y-Lage, Nachbarn in X
    reihen = {}
    for s in staemme:
        reihen.setdefault(round(s[1], 2), []).append(s)
    for r in reihen.values():
        r.sort(key=lambda s: s[0])
        for a, b in zip(r, r[1:]):
            verbinde(a, b)
    # Nachbarreihen in Y, Stämme gleicher X-Lage
    ys = sorted(reihen)
    for y0, y1 in zip(ys, ys[1:]):
        for a in reihen[y0]:
            for b in reihen[y1]:
                if abs(a[0] - b[0]) < 0.5:
                    verbinde(a, b)
    return teile, reihen


def schienen(reihen):
    """Raft-Schienen unter jeder Reihe, dazu Querschienen als Verbund."""
    teile = []
    for y, r in reihen.items():
        x0, x1 = min(s[0] for s in r) - FUSS_R, max(s[0] for s in r) + FUSS_R
        teile.append(trimesh.creation.box(bounds=[[x0, y - SCHIENE_B / 2, 0],
                                                  [x1, y + SCHIENE_B / 2, RAFT_H]]))
    xs = [s[0] for r in reihen.values() for s in r]
    for xq in np.linspace(min(xs), max(xs), 6):
        drin = [y for y, r in reihen.items()
                if min(s[0] for s in r) - 0.1 <= xq <= max(s[0] for s in r) + 0.1]
        if len(drin) > 1:
            teile.append(trimesh.creation.box(bounds=[[xq - SCHIENE_B / 2, min(drin), 0],
                                                      [xq + SCHIENE_B / 2, max(drin), RAFT_H]]))
    return teile


def ueberhang_punkte(variante, form, teil, rot, dz, vorhanden):
    """Zusätzliche Stützpunkte für Schnittflächen der optimierten Form, die in
    Drucklage flach nach unten zeigen (v. a. Oberkanten der Fenster).
    Bewertet wird die über 1,5 mm gemittelte Normale, sonst zählen die
    Voxelstufen schräger Flächen als Überhang. Rückgabe: Liste (p, n, ziel),
    ziel = None für Stützen bis zur Platte, sonst die Höhe in Drucklage, auf
    der die Stütze auf dem Teil selbst aufsetzt (Stamm träfe sonst das Teil)."""
    orig = bs.balkenschuh(variante)
    alle_n, flaeche = form.face_normals, form.area_faces
    kand = np.nonzero((alle_n @ rot.T)[:, 2] < -np.cos(np.radians(UEBERHANG)))[0]
    c = form.triangles_center[kand]
    baum = cKDTree(form.triangles_center)
    n = np.array([(alle_n[i] * flaeche[i, None]).sum(0) for i in baum.query_ball_point(c, 1.5)])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    flach = (n @ rot.T)[:, 2] < -np.cos(np.radians(UEBERHANG))
    c, n = c[flach], n[flach]
    # neue Flächen: Schnitt ins Original (davor war Material) oder ganz außerhalb
    # des Originals (Bauraum-Modus); Originalflächen tragen schon das Grundraster
    neu_flaeche = (topopt.punkte_innen(orig, c + 0.3 * n)
                   | ~topopt.punkte_innen(orig, c - 0.3 * n))
    c, n = c[neu_flaeche], n[neu_flaeche]
    hoehe = (c @ rot.T)[:, 2]
    belegt = [np.asarray(p, float) for p, _ in vorhanden]
    gewaehlt = []
    for i in np.argsort(hoehe):                             # tiefste zuerst
        if all(np.linalg.norm(c[i] - b) >= UEBERHANG_RASTER for b in belegt):
            gewaehlt.append((tuple(c[i]), tuple(n[i])))
            belegt.append(c[i])
    if not gewaehlt:
        return []

    # Belegung des Teils in Drucklage (0,4-mm-Raster) für den Freigang der Stämme
    lo, hi = teil.bounds
    ach = [np.arange(lo[a] - 0.2, hi[a] + 0.6, 0.4) for a in range(3)]
    voll = topopt.innen_maske(teil, ach)

    def belegt_an(punkte):
        idx = [np.clip(np.searchsorted(ach[a], punkte[:, a]) - 1, 0, len(ach[a]) - 2)
               for a in range(3)]
        drin = np.all([(punkte[:, a] > ach[a][0]) & (punkte[:, a] < ach[a][-1]) for a in range(3)], 0)
        return voll[tuple(idx)] & drin

    aus = []
    for p, nn in gewaehlt:
        pp = rot @ np.asarray(p) + [0, 0, dz]
        nd = rot @ np.asarray(nn)
        q = pp + KOPF_L * nd
        if belegt_an(pp + np.linspace(0.8, KOPF_L, 6)[:, None] * nd).any():
            continue                                         # Kopf steckt im Teil
        zs = np.arange(q[2] - 0.4, RAFT_H + FUSS_H, -0.2)
        treffer = belegt_an(np.column_stack([np.full(len(zs), q[0]), np.full(len(zs), q[1]), zs]))
        if not treffer.any():
            if q[2] - UEBERGANG_L > RAFT_H + FUSS_H + 0.5:
                aus.append((p, nn, None))
        elif q[2] - zs[np.argmax(treffer)] >= 1.5:          # Platz für eine Stütze aufs Teil
            aus.append((p, nn, zs[np.argmax(treffer)] + 0.2))
    return aus


def stuetze_auf_teil(p, n, ziel):
    """Kurze Stütze, die auf dem Teil selbst aufsetzt (Kopf wie üblich, Stamm
    dünn, unten wieder eine Kontaktspitze)."""
    p, n = np.asarray(p), np.asarray(n)
    q = p + KOPF_L * n
    fuss = (q[0], q[1], ziel + 1.0)
    return [
        kegelstumpf(p - EMBED * n, q, KONTAKT_R, KOPF_R, 12),
        kugel(q, KOPF_R),
        kegelstumpf(q, fuss, KOPF_R, KOPF_R, 12),
        kegelstumpf(fuss, (q[0], q[1], ziel - EMBED), KOPF_R, KONTAKT_R, 12),
    ]


def teil_mit_stuetzen(variante, form=None):
    """Ein Teil in Drucklage samt Supports, Ursprung links unten auf der Platte.
    form: optimierte Form in Konstruktionslage (sonst das Original)."""
    m = druck_matrix()
    teil = bs.balkenschuh(variante) if form is None else form.copy()
    punkte = [(p, n, None) for p, n in stuetzpunkte(variante, form)]
    teil.apply_transform(m)
    dz = LIFT - teil.bounds[0][2]
    teil.apply_translation([0, 0, dz])

    rot = m[:3, :3]
    if form is not None:
        extra = ueberhang_punkte(variante, form, teil, rot, dz, [(p, n) for p, n, _ in punkte])
        auf_teil = sum(z is not None for _, _, z in extra)
        print(f"  {len(punkte)} Stützpunkte vom Original übernommen, {len(extra)} an "
              f"Überhängen der optimierten Form ergänzt ({auf_teil} davon auf dem Teil)")
        punkte = punkte + extra
    sup, staemme = [], []
    for p, n, ziel in punkte:
        pp = rot @ np.asarray(p, float) + [0, 0, dz]
        nd = rot @ np.asarray(n, float)
        if ziel is None:
            tt, stamm = stuetze(pp, nd)
            staemme.append(stamm)
        else:
            tt = stuetze_auf_teil(pp, nd, ziel)
        sup += tt
    st, reihen = streben(staemme)
    # check_volume aus: Berührkanten der optimierten Form verträgt manifold3d
    alles = trimesh.boolean.union([teil] + sup + st + schienen(reihen), engine="manifold",
                                  check_volume=False)
    alles.apply_translation([-alles.bounds[0][0], -alles.bounds[0][1], 0])
    return alles


def druckplatte(variante, form=None):
    einzel = teil_mit_stuetzen(variante, form)
    fx, fy = einzel.extents[:2]
    gx, gy = 2 * fx + SPALT, 2 * fy + SPALT
    assert gx <= PLATTE[0] - 4 and gy <= PLATTE[1] - 4, f"passt nicht: {gx:.1f} x {gy:.1f}"
    x0, y0 = (PLATTE[0] - gx) / 2, (PLATTE[1] - gy) / 2
    kopien = []
    for i in range(2):
        for j in range(2):
            k = einzel.copy()
            k.apply_translation([x0 + i * (fx + SPALT), y0 + j * (fy + SPALT), 0])
            kopien.append(k)
    return trimesh.util.concatenate(kopien), einzel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimiert", type=float, metavar="VOL",
                    help="optimierte Form mit diesem Volumenanteil (aus optimieren.py)")
    ap.add_argument("varianten", nargs="*", default=list(bs.VARIANTEN))
    a = ap.parse_args()
    os.makedirs(stl_dir, exist_ok=True)
    for name in a.varianten:
        form, zusatz = None, ""
        if a.optimiert:
            tag = f"{name}_{round(100 * a.optimiert)}"
            form = trimesh.load(os.path.join(stl_dir, "..", "optimierung", f"konstruktion_{tag}.stl"))
            zusatz = f"_{round(100 * a.optimiert)}_optimiert"
            print(f"{name}: {len(stuetzpunkte(name))} Stützpunkte am Original, "
                  f"{len(stuetzpunkte(name, form))} am optimierten Teil")
        platte, einzel = druckplatte(name, form)
        out = os.path.join(stl_dir, f"druckplatte_balkenschuh_9mm_{name}{zusatz}_4x.stl")
        platte.export(out)
        lo, hi = platte.bounds
        print(out)
        print(f"  Teil+Supports: wasserdicht {einzel.is_watertight}, "
              f"{einzel.volume / 1000:.1f} ml, Grundfläche {einzel.extents[0]:.1f} x "
              f"{einzel.extents[1]:.1f} mm")
        print(f"  Platte: X {lo[0]:.1f}..{hi[0]:.1f}, Y {lo[1]:.1f}..{hi[1]:.1f}, "
              f"Höhe {hi[2]:.1f} mm, Harz gesamt ca. {4 * einzel.volume / 1000:.0f} ml, "
              f"{len(platte.faces)} Dreiecke, Schichten bei 0,05 mm: {hi[2] / 0.05:.0f}")
