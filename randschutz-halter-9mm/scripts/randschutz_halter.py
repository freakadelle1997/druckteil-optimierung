#!/usr/bin/env python3
"""
Randschutz-Halter für 9-mm-Trassenbretter
=========================================
Kurzer U-Clip, der seitlich auf die Kante eines liegenden 9-mm-Trassenbretts
geschoben und von oben mit einer Schraube gesichert wird. Direkt vor der
Brettkante hat er einen senkrechten Schlitz, in den ein 3-mm-MDF-Streifen
hochkant eingesteckt wird (Absturzschutz für Loks).

Für Bögen: Der Clip ist entlang der Kante nur 20 mm lang. Auf 20 mm weicht
ein Bogen mit R 250 mm nur 0,2 mm von der Geraden ab, das schluckt die Luft
im Schlitz und an der Brettkante. Der MDF-Streifen läuft ungeteilt durch
alle Clips und biegt sich dazwischen. Eine Klemmrippe in der Schlitzmitte
hält ihn auch im Bogen (Dreipunktklemmung).

Konstruktions-Koordinaten: x entlang der Brettkante, y quer zur Kante
(Brett bei y < 0, Kante bei y = 0), z nach oben (Brettunterseite z = 0).
"""

import os
import numpy as np
import trimesh
import manifold3d
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

# ----------------------------------------------------------------------------
# Parameter [mm]
# ----------------------------------------------------------------------------
L = 20.0                     # Länge entlang der Brettkante
BRETT = 9.0                  # Trassenbrett
MAUL = BRETT + 0.4           # lichte Weite zwischen den Backen
T_OBEN = 3.0                 # obere Backe (liegt auf dem Brett)
T_UNTEN = 3.0                # untere Backe (Führung beim Aufschieben)
TIEFE_OBEN = 15.0            # obere Backe greift so weit aufs Brett
TIEFE_UNTEN = 6.0            # untere Backe kürzer, damit die Schraube vorbeigeht
STEG = 3.0                   # Wand zwischen Brettkante und MDF

MDF = 3.0                    # MDF-Streifen
SCHLITZ = MDF + 0.4          # lichte Weite des Schlitzes
KLEMM = 2.9                  # Restweite an der Klemmrippe
AUSSEN = 2.5                 # äußere Schlitzwand
BODEN = BRETT                # Schlitzboden = Streifen-Unterkante, bündig mit Brettoberkante
OBEN = BODEN + 12.0          # Oberkante Schlitzwände (12 mm Stecktiefe, 12 mm über dem Brett)

LOCH = 3.8                   # für 3,5er Holzschraube (oder 3er mit Scheibe)
LOCH_Y = -9.5                # Lochmitte ab Brettkante
FASE = 1.0                   # Einführfasen an Backen und Schlitz
FREI_R = 0.6                 # Freistich für die Brettkanten
RUND = 0.4                   # Außenecken
RIPPE_R = 1.0                # Klemmrippe (Halbzylinder, senkrecht)

Z_UNTEN = -T_UNTEN
Z_BACKE = MAUL               # Unterseite obere Backe
Y_SCHLITZ = (STEG, STEG + SCHLITZ)
Y_AUSSEN = STEG + SCHLITZ + AUSSEN

here = os.path.dirname(os.path.abspath(__file__))
stl_dir = os.path.abspath(os.path.join(here, "..", "stl"))


def profil():
    """Querschnitt in (y, z)."""
    teile = [
        box(0, Z_UNTEN, STEG, OBEN),                                   # Steg
        box(-TIEFE_OBEN, Z_BACKE, 0, Z_BACKE + T_OBEN),                # obere Backe
        box(-TIEFE_UNTEN, Z_UNTEN, 0, 0),                              # untere Backe
        box(Y_SCHLITZ[0], Z_UNTEN, Y_SCHLITZ[1], BODEN),               # Schlitzboden
        box(Y_SCHLITZ[1], Z_UNTEN, Y_AUSSEN, OBEN),                    # Außenwand
        Polygon([(-2.5, Z_BACKE + T_OBEN), (0, Z_BACKE + T_OBEN),      # Kehle Backe/Steg
                 (0, Z_BACKE + T_OBEN + 2.5)]),
    ]
    p = unary_union(teile)
    weg = [
        # Einführfasen an den Backenspitzen (brettseitig)
        Polygon([(-TIEFE_OBEN - 0.1, Z_BACKE - 0.1), (-TIEFE_OBEN + FASE, Z_BACKE - 0.1),
                 (-TIEFE_OBEN - 0.1, Z_BACKE + FASE + 0.1)]),
        Polygon([(-TIEFE_UNTEN - 0.1, 0.1), (-TIEFE_UNTEN + FASE, 0.1),
                 (-TIEFE_UNTEN - 0.1, -FASE - 0.1)]),
        # Einführfasen oben am Schlitz
        Polygon([(Y_SCHLITZ[0] + 0.1, OBEN + 0.1), (Y_SCHLITZ[0] - FASE, OBEN + 0.1),
                 (Y_SCHLITZ[0] + 0.1, OBEN - FASE - 0.1)]),
        Polygon([(Y_SCHLITZ[1] - 0.1, OBEN + 0.1), (Y_SCHLITZ[1] + FASE, OBEN + 0.1),
                 (Y_SCHLITZ[1] - 0.1, OBEN - FASE - 0.1)]),
    ]
    p = p.difference(unary_union(weg))
    p = p.buffer(-RUND, quad_segs=6).buffer(RUND, quad_segs=6)      # Außenecken runden
    # Freistiche in den Innenecken, damit die Brettkanten satt anliegen
    p = p.difference(Point(0, 0).buffer(FREI_R, quad_segs=6))
    p = p.difference(Point(0, Z_BACKE).buffer(FREI_R, quad_segs=6))
    # fast doppelte Punkte aus den Buffern weg, sonst ist die Extrusion nicht dicht
    return p.simplify(0.01)


def zylinder(r, p0, p1, seg=32):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    c = trimesh.creation.cylinder(radius=r, height=np.linalg.norm(d), sections=seg)
    tf = trimesh.geometry.align_vectors([0, 0, 1], d / np.linalg.norm(d))
    tf[:3, 3] = (p0 + p1) / 2
    c.apply_transform(tf)
    return c


def kegel(r0, r1, p0, p1, seg=32):
    """Kegelstumpf von p0 (Radius r0) nach p1 (Radius r1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    h = np.linalg.norm(d)
    mm = manifold3d.Manifold.cylinder(h, r0, r1, seg).to_mesh()
    m = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts, process=False)
    tf = trimesh.geometry.align_vectors([0, 0, 1], d / h)
    tf[:3, 3] = p0
    m.apply_transform(tf)
    return m


def halter():
    """Clip in Einbaulage (Konstruktions-Koordinaten)."""
    m = trimesh.creation.extrude_polygon(profil(), L)
    # lokal (u = y, v = z, w = x) -> (x, y, z), zyklisch, ohne Spiegelung
    rot = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    m.apply_transform(rot)

    # Klemmrippe an der Innenseite der Außenwand, oben mit Anlaufkegel
    yr = Y_SCHLITZ[0] + KLEMM + RIPPE_R
    z_kegel = OBEN - FASE - 3.0
    rippe = [zylinder(RIPPE_R, (L / 2, yr, BODEN - 0.5), (L / 2, yr, z_kegel)),
             kegel(RIPPE_R, Y_SCHLITZ[1] - yr - 0.05, (L / 2, yr, z_kegel - 0.01),
                   (L / 2, yr, z_kegel + 2.5))]
    m = trimesh.boolean.union([m] + rippe, engine="manifold")

    # Schraubenloch durch die obere Backe
    loch = zylinder(LOCH / 2, (L / 2, LOCH_Y, Z_BACKE - 1), (L / 2, LOCH_Y, Z_BACKE + T_OBEN + 1))
    m = trimesh.boolean.difference([m, loch], engine="manifold")
    return m


if __name__ == "__main__":
    os.makedirs(stl_dir, exist_ok=True)
    m = halter()
    out = os.path.join(stl_dir, "randschutz_halter_9mm.stl")
    m.export(out)
    lo, hi = m.bounds
    print(out)
    print(f"  wasserdicht {m.is_watertight}, Volumen {m.volume / 1000:.2f} cm³, "
          f"{hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} x {hi[2] - lo[2]:.1f} mm")
