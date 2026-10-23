#!/usr/bin/env python3
"""
Balkenschuh Typ A (außenliegende Laschen) für ein 9-mm-Brett
=============================================================
Wandhalter für ein hochkant stehendes Brett (9 mm dick, 130 mm hoch), das
wie ein Spant im rechten Winkel absteht (T-Verbindung). Form nach dem
handelsüblichen Balkenschuh Typ A: U-Tasche aus Boden und zwei Wangen,
außen zwei Laschen, die unten schräg zur Wange auslaufen. Das Brett wird
von vorne eingeschoben, steht auf dem Boden auf und wird seitlich durch
die Wangen verschraubt.

Zwei Varianten, gleiche Tasche:
  - "wand": je Lasche 2 Löcher 5,3 mm für 5er Schraube mit 6er Dübel
    (mit Unterlegscheibe, kein Senkkopf),
  - "holz": für das andere Brettende an Möbel/Holz, je Lasche 2 Löcher
    3,8 mm mit 90°-Senkung für 3,5er Senkkopf-Holzschrauben, Lasche 4 mm.
Beide: 2 Löcher 3,6 mm für das Brett, durchgehend durch beide Wangen.

Drucker: Anycubic Photon Mono M7 Pro (Resin, ABS-Like 3.0), Bauplatte
223 x 126 mm. Diese Datei erzeugt die Einzelteile flach (Wandseite auf der
Platte, ohne Supports); die fertige Druckplatte mit Supports baut
druckplatte.py.

Konstruktions-Koordinaten: x quer, y von der Wand weg, z nach oben.
"""

import os
import numpy as np
import trimesh
import manifold3d
from shapely.geometry import Polygon, box as shp_box, Point

# ----------------------------------------------------------------------------
# Parameter [mm]
# ----------------------------------------------------------------------------
BRETT_D = 9.0                # Brettstärke
BRETT_H = 130.0              # Bretthöhe
SPIEL = 0.4                  # Luft gesamt (0,2 je Seite)
W_IN = BRETT_D + SPIEL       # lichte Breite der Tasche

T = 3.0                      # Wandstärke Wangen
T_BODEN = 3.0                # Bodenstärke
TIEFE = 35.0                 # wie weit die Wangen von der Wand abstehen
H = T_BODEN + BRETT_H        # Gesamthöhe (Brett schließt oben bündig ab)

LASCHE_B = 20.0              # Laschenbreite neben der Wange
LASCHE_SCHRAEG_H = 45.0      # Höhe der unteren Schräge (wie beim Original)
LASCHE_FUSS = 5.0            # Restbreite der Lasche am Fuß
R_ECKE = 2.0                 # Eckenradius an den Laschen
R_BIEGE = 3.0                # Rundung außen zwischen Boden und Wange
KEHLE = 4.0                  # Hohlkehle zwischen Wange und Lasche

WAND_LOCH_Z = (60.0, H - 20.0)   # Lochmitten Laschen, ab Unterkante
BRETT_LOCH = 3.6             # für 3,5er Holzschraube oder M3
BRETT_LOCH_Y = 20.0          # Lochmitte ab Wand
BRETT_LOCH_Z = (30.0, 100.0) # Lochmitte ab Taschenboden

VARIANTEN = {
    "wand": dict(t_lasche=3.0, loch=5.3, senkung=None),
    "holz": dict(t_lasche=4.0, loch=3.8, senkung=7.4),
}

E = 0.01                     # Überstand für saubere Booleans

xi = W_IN / 2                # Innenfläche Wange
xa = xi + T                  # Außenfläche Wange
xl = xa + LASCHE_B           # Außenkante Lasche
LOCH_X = xa + LASCHE_B / 2   # Lochmitte quer

here = os.path.dirname(__file__)
stl_dir = os.path.abspath(os.path.join(here, "..", "stl"))


# ----------------------------------------------------------------------------
# Helfer
# ----------------------------------------------------------------------------
def prism_xy(poly, z0, z1):
    """Profil in (x, y), extrudiert entlang z."""
    m = trimesh.creation.extrude_polygon(poly, z1 - z0)
    m.apply_translation([0, 0, z0])
    return m


def prism_yz(poly, x0, x1):
    """Profil in (y, z), extrudiert entlang x."""
    m = trimesh.creation.extrude_polygon(poly, x1 - x0)
    # lokal (u, v, w) -> (x = w, y = u, z = v), zyklisch, also ohne Spiegelung
    rot = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    m.apply_transform(rot)
    m.apply_translation([x0, 0, 0])
    return m


def prism_xz(poly, y0, y1):
    """Profil in (x, z), extrudiert entlang y."""
    m = trimesh.creation.extrude_polygon(poly, y1 - y0)
    # Rx(+90°): (u, v, w) -> (x = u, y = -w, z = v)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, y1, 0])
    return m


def kegelstumpf(p0, p1, r0, r1, seg=32):
    """Kegelstumpf von p0 (Radius r0) nach p1 (Radius r1), beliebige Richtung."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    h = np.linalg.norm(d)
    mm = manifold3d.Manifold.cylinder(h, r0, r1, seg).to_mesh()
    m = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts, process=False)
    tf = trimesh.geometry.align_vectors([0, 0, 1], d / h)
    tf[:3, 3] = p0
    m.apply_transform(tf)
    return m


def rund(poly, r):
    """Konvexe Ecken eines Profils runden."""
    return poly.buffer(-r, quad_segs=8).buffer(r, quad_segs=8)


def lasche_profil():
    """Rechte Lasche in (x, z): oben volle Breite, unten schräg zur Wange."""
    p = Polygon([(xi, 0), (xa + LASCHE_FUSS, 0), (xl, LASCHE_SCHRAEG_H),
                 (xl, H), (xi, H)])
    return rund(p, R_ECKE)


def spiegeln(m):
    s = m.copy()
    s.apply_transform(trimesh.transformations.reflection_matrix([0, 0, 0], [1, 0, 0]))
    return s


# ----------------------------------------------------------------------------
# Bauteil
# ----------------------------------------------------------------------------
def balkenschuh(variante="wand"):
    v = VARIANTEN[variante]
    tl = v["t_lasche"]

    # U-Tasche aus Boden und Wangen, Profil in (x, z), außen unten gerundet
    aussen = rund(shp_box(-xa, 0, xa, H + R_BIEGE), R_BIEGE).intersection(
        shp_box(-xa, 0, xa, H))
    u = aussen.difference(shp_box(-xi, T_BODEN, xi, H + 1))
    teile = [prism_xz(u, 0, TIEFE)]

    rechts = [prism_xz(lasche_profil(), 0, tl)]
    kehle = shp_box(xa - E, tl - E, xa + KEHLE, tl + KEHLE).difference(
        Point(xa + KEHLE, tl + KEHLE).buffer(KEHLE, quad_segs=16))
    rechts.append(prism_xy(kehle, 0, H))
    rechts = trimesh.boolean.union(rechts, engine="manifold")
    teile += [rechts, spiegeln(rechts)]
    koerper = trimesh.boolean.union(teile, engine="manifold")

    abzug = []
    for sx in (1, -1):
        for z in WAND_LOCH_Z:
            p = (sx * LOCH_X, -1, z)
            abzug.append(kegelstumpf(p, (sx * LOCH_X, tl + 1, z), v["loch"] / 2, v["loch"] / 2))
            if v["senkung"]:
                tiefe = (v["senkung"] - v["loch"]) / 2          # 90°-Senkung
                abzug.append(kegelstumpf((sx * LOCH_X, tl - tiefe, z),
                                         (sx * LOCH_X, tl + E, z),
                                         v["loch"] / 2, v["senkung"] / 2 + E))
    for z in BRETT_LOCH_Z:
        abzug.append(kegelstumpf((-xa - 1, BRETT_LOCH_Y, T_BODEN + z),
                                 (xa + 1, BRETT_LOCH_Y, T_BODEN + z),
                                 BRETT_LOCH / 2, BRETT_LOCH / 2))

    return trimesh.boolean.difference([koerper] + abzug, engine="manifold")


def druckausrichtung_flach(m):
    """Wandseite (y = 0) auf die Bauplatte, Höhe entlang der langen Plattenachse."""
    d = m.copy()
    # (x, y, z) -> (X = z, Y = x, Z = y), zyklisch, also ohne Spiegelung
    rot = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    d.apply_transform(rot)
    d.apply_translation(-d.bounds[0])
    return d


if __name__ == "__main__":
    os.makedirs(stl_dir, exist_ok=True)
    for name in VARIANTEN:
        teil = druckausrichtung_flach(balkenschuh(name))
        out = os.path.join(stl_dir, f"balkenschuh_9mm_{name}_einzeln.stl")
        teil.export(out)
        ext = teil.extents
        print(out)
        print(f"  wasserdicht: {teil.is_watertight}, Volumen {teil.volume / 1000:.1f} ml, "
              f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm")
