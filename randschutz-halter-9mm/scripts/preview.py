#!/usr/bin/env python3
"""Kontrollbild: Schnitt in Einbaulage, 3D-Ansicht, Draufsicht im Bogen."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPoly, Arc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import randschutz_halter as rh

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.abspath(os.path.join(here, "..", "previews", "randschutz_halter_9mm.png"))

HOLZ, MDF_F, TEIL = "#d9b77e", "#8c6e4f", "#e08a3c"
STREIFEN = 30.0              # Beispiel: 30 mm breiter MDF-Streifen

fig = plt.figure(figsize=(17, 6.5))

# --- Schnitt -----------------------------------------------------------------
a = fig.add_subplot(1, 3, 1)
a.add_patch(Rectangle((-45, 0), 45, rh.BRETT, fc=HOLZ, ec="k", lw=0.6))
a.text(-30, rh.BRETT / 2, "Trassenbrett 9 mm", ha="center", va="center", fontsize=8)
y0 = rh.Y_SCHLITZ[0] + (rh.SCHLITZ - rh.MDF) / 2
a.add_patch(Rectangle((y0, rh.BODEN), rh.MDF, STREIFEN, fc=MDF_F, ec="k", lw=0.6))
a.text(y0 + rh.MDF / 2, rh.BODEN + STREIFEN - 6, "MDF 3 mm", rotation=90,
       ha="center", va="center", fontsize=8, color="w")
p = rh.profil()
a.add_patch(MPoly(np.array(p.exterior.coords), fc=TEIL, ec="k", lw=0.8, alpha=0.95))
# Schraube 3,5 x 10 mit Linsenkopf
zk = rh.Z_BACKE + rh.T_OBEN
a.add_patch(Rectangle((rh.LOCH_Y - 1.75, zk - 10), 3.5, 10, fc="#9aa3ad", ec="k", lw=0.5))
a.add_patch(Rectangle((rh.LOCH_Y - 3.5, zk), 7, 2.2, fc="#9aa3ad", ec="k", lw=0.5))
a.annotate("", (rh.Y_AUSSEN + 3, rh.BODEN), (rh.Y_AUSSEN + 3, rh.OBEN),
           arrowprops=dict(arrowstyle="<->", lw=0.7))
a.text(rh.Y_AUSSEN + 4, (rh.BODEN + rh.OBEN) / 2, f"{rh.OBEN - rh.BODEN:.0f} mm\nim Schlitz",
       fontsize=7, va="center")
a.annotate("", (-rh.TIEFE_OBEN, zk + 6), (0, zk + 6), arrowprops=dict(arrowstyle="<->", lw=0.7))
a.text(-rh.TIEFE_OBEN / 2, zk + 7, f"{rh.TIEFE_OBEN:.0f} mm aufs Brett", ha="center", fontsize=7)
a.text(rh.LOCH_Y, zk + 3, "3,5 × 10", ha="center", fontsize=7)
a.text(-44, rh.BODEN + STREIFEN - 2,
       f"Beispiel: Streifen {STREIFEN:.0f} mm breit\n→ {STREIFEN + rh.BODEN - rh.BRETT:.0f} mm über Brettoberkante",
       fontsize=8, va="top")
a.set_xlim(-46, 22)
a.set_ylim(-6, rh.BODEN + STREIFEN + 2)
a.set_aspect("equal")
a.set_title("Schnitt quer zur Brettkante (Einbaulage)")
a.set_xlabel("mm")

# --- 3D ----------------------------------------------------------------------
m = rh.halter()
a3 = fig.add_subplot(1, 3, 2, projection="3d")
pc = Poly3DCollection(m.triangles, alpha=1.0)
licht = np.clip(m.face_normals @ np.array([0.4, -0.5, 0.75]), 0.15, 1)
pc.set_facecolor(np.c_[np.outer(licht, [0.88, 0.54, 0.24]), np.ones(len(licht))])
pc.set_edgecolor("none")
a3.add_collection3d(pc)
lo, hi = m.bounds
c, r = (lo + hi) / 2, (hi - lo).max() / 2
a3.set_xlim(c[0] - r, c[0] + r)
a3.set_ylim(c[1] - r, c[1] + r)
a3.set_zlim(c[2] - r, c[2] + r)
a3.set_box_aspect((1, 1, 1))
a3.view_init(elev=28, azim=-125)
a3.set_axis_off()
a3.set_title(f"Halter {rh.L:.0f} × {hi[1] - lo[1]:.0f} × {hi[2] - lo[2]:.0f} mm, {m.volume / 1000:.1f} cm³")

# --- Draufsicht Bogen --------------------------------------------------------
ab = fig.add_subplot(1, 3, 3)
R_in, R_out = 250.0, 340.0
w = np.radians(np.linspace(0, 60, 200))
ring = np.r_[np.c_[R_out * np.cos(w), R_out * np.sin(w)],
             np.c_[R_in * np.cos(w[::-1]), R_in * np.sin(w[::-1])]]
ab.add_patch(MPoly(ring, fc=HOLZ, ec="k", lw=0.6))
for R, aussen in ((R_out, True), (R_in, False)):
    s = 1 if aussen else -1
    rm = R + s * (rh.Y_SCHLITZ[0] + rh.SCHLITZ / 2)
    ab.add_patch(Arc((0, 0), 2 * rm, 2 * rm, theta1=0, theta2=60, color=MDF_F, lw=2.2))
    for wk in np.radians(np.arange(5, 60, 12)):
        u = np.array([np.cos(wk), np.sin(wk)])
        t = np.array([-u[1], u[0]])
        ecken = []
        for dy, dx in ((-rh.TIEFE_OBEN, -rh.L / 2), (rh.Y_AUSSEN, -rh.L / 2),
                       (rh.Y_AUSSEN, rh.L / 2), (-rh.TIEFE_OBEN, rh.L / 2)):
            ecken.append((R + s * dy) * u + dx * t)
        ab.add_patch(MPoly(np.array(ecken), fc=TEIL, ec="k", lw=0.5))
ab.text(0.5 * (R_in + R_out) * np.cos(np.radians(30)), 0.5 * (R_in + R_out) * np.sin(np.radians(30)),
        "Trassenbrett\nim Bogen", ha="center", va="center", fontsize=8, rotation=-60)
ab.set_aspect("equal")
ab.autoscale()
ab.set_title(f"Bogen innen R {R_in:.0f} / außen R {R_out:.0f}: Clips an beiden Kanten")
ab.set_xlabel("mm")

fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)
