#!/usr/bin/env python3
"""Vorschau: beide Varianten, Einbau mit Brett, Maßskizze und Druckplatten."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

import balkenschuh_9mm as bs

here = os.path.dirname(__file__)
prev_dir = os.path.abspath(os.path.join(here, "..", "previews"))
stl_dir = os.path.abspath(os.path.join(here, "..", "stl"))
LICHT = np.array([0.4, -0.7, 0.6]) / np.linalg.norm([0.4, -0.7, 0.6])
STAHL = np.array([0.72, 0.74, 0.78])
HOLZ = np.array([0.85, 0.65, 0.40])
GRAU = np.array([0.80, 0.80, 0.80])


def shade(mesh, rgb, alpha=1.0):
    k = 0.35 + 0.65 * np.clip(mesh.face_normals @ LICHT, 0, 1)
    c = np.clip(np.outer(k, rgb), 0, 1)
    return np.column_stack([c, np.full(len(c), alpha)])


def zeichne(ax, teile, elev, azim, titel, zoom=1.0):
    for mesh, rgb, alpha in teile:
        if len(mesh.faces) < 20000:
            # große Dreiecke teilen, sonst sortiert matplotlib die Tiefe falsch
            mesh = mesh.subdivide_to_size(max_edge=4.0, max_iter=12)
        ax.add_collection3d(Poly3DCollection(mesh.triangles, edgecolors="none",
                                             facecolors=shade(mesh, rgb, alpha)))
    pts = np.vstack([m.vertices for m, _, _ in teile])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid, r = (lo + hi) / 2, (hi - lo).max() / 2 / zoom
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(titel, fontsize=11)


def ansicht():
    wand = bs.balkenschuh("wand")
    holz = bs.balkenschuh("holz")
    brett = trimesh.creation.box(bounds=[[-bs.BRETT_D / 2, 0, bs.T_BODEN],
                                         [bs.BRETT_D / 2, 110, bs.H]])
    wand_m = trimesh.creation.box(bounds=[[-55, -4, -15], [55, 0, bs.H + 15]])

    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    zeichne(ax, [(wand, STAHL, 1.0)], 20, 55, "Variante Wand (Dübellöcher 5,3 mm)", 1.25)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    zeichne(ax, [(holz, STAHL, 1.0)], 20, 55, "Variante Holz (Senklöcher 3,8/7,4 mm)", 1.25)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    zeichne(ax, [(wand_m, GRAU, 0.25), (wand, STAHL, 1.0), (brett, HOLZ, 0.55)],
            22, 40, "Einbau: Brett 9 x 130 mm von vorne eingeschoben")
    fig.tight_layout()
    out = os.path.join(prev_dir, "balkenschuh_9mm_ansicht.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


def masse():
    teil = bs.balkenschuh("wand")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8),
                                 gridspec_kw={"width_ratios": [1, 1.3]})

    # Draufsicht: Schnitt auf halber Brett-Höhe als Liniensegmente
    seg = trimesh.intersections.mesh_plane(teil, [0, 0, 1], [0, 0, bs.T_BODEN + 65])
    for s in seg:
        a1.plot(s[:, 0], s[:, 1], color="k", lw=1)
    a1.add_patch(plt.Rectangle((-bs.BRETT_D / 2, 0), bs.BRETT_D, 50, color=HOLZ, alpha=0.5))
    a1.axhline(0, color="gray", lw=3)
    a1.text(0, -2, "Wand / Holz", ha="center", va="top", color="gray")
    a1.annotate("", (-bs.xi, 44), (bs.xi, 44), arrowprops=dict(arrowstyle="<->"))
    a1.text(0, 46, f"{bs.W_IN:.1f} innen", ha="center")
    a1.annotate("", (-bs.xl, -8), (bs.xl, -8), arrowprops=dict(arrowstyle="<->"))
    a1.text(0, -10, f"{2 * bs.xl:.1f}", ha="center", va="top")
    a1.annotate("", (bs.xa + 8, 0), (bs.xa + 8, bs.TIEFE), arrowprops=dict(arrowstyle="<->"))
    a1.text(bs.xa + 9, bs.TIEFE / 2, f"{bs.TIEFE:.0f}", va="center")
    a1.set_aspect("equal")
    a1.set_title("Draufsicht (Schnitt), Maße in mm")
    a1.set_ylim(-16, 54)
    a1.axis("off")

    # Vorderansicht: Laschenprofil, Löcher, Brett
    lp = bs.lasche_profil()
    for sx in (1, -1):
        x, z = lp.exterior.xy
        a2.fill(sx * np.asarray(x), z, color=STAHL, ec="k")
        for z0 in bs.WAND_LOCH_Z:
            a2.add_patch(plt.Circle((sx * bs.LOCH_X, z0), 2.65, color="white", ec="k"))
    a2.add_patch(plt.Rectangle((-bs.xi, 0), 2 * bs.xi, bs.T_BODEN, color=STAHL, ec="k"))
    a2.add_patch(plt.Rectangle((-bs.xi, bs.T_BODEN), 2 * bs.xi, bs.BRETT_H,
                               color=HOLZ, alpha=0.5))
    for z0 in bs.WAND_LOCH_Z:
        a2.annotate(f"{z0:.0f} ab unten", (bs.LOCH_X, z0), (bs.xl + 8, z0),
                    va="center", arrowprops=dict(arrowstyle="-"))
    a2.annotate("", (bs.xl + 4, 0), (bs.xl + 4, bs.H), arrowprops=dict(arrowstyle="<->"))
    a2.text(bs.xl + 5, bs.H / 2, f"{bs.H:.0f}", va="center")
    for z0 in bs.BRETT_LOCH_Z:
        a2.plot([-bs.xl, bs.xl], [bs.T_BODEN + z0] * 2, ls=":", color="k", lw=0.8)
        a2.text(-bs.xl - 2, bs.T_BODEN + z0,
                f"Brettloch Ø {bs.BRETT_LOCH}\n{z0:.0f} über Boden,\n"
                f"{bs.BRETT_LOCH_Y:.0f} vor der Wand", ha="right", va="center", fontsize=9)
    a2.text(0, -4, "Wand: Ø 5,3   |   Holz: Ø 3,8 mit Senkung Ø 7,4",
            ha="center", va="top", fontsize=9)
    a2.set_xlim(-bs.xl - 45, bs.xl + 40)
    a2.set_ylim(-10, bs.H + 5)
    a2.set_aspect("equal")
    a2.set_title("Vorderansicht (Brettlöcher quer durch beide Wangen)")
    a2.axis("off")
    fig.tight_layout()
    out = os.path.join(prev_dir, "balkenschuh_9mm_masse.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


def druckplatten():
    import druckplatte as dp
    platte = trimesh.creation.box(bounds=[[0, 0, -2], [dp.PLATTE[0], dp.PLATTE[1], 0]])
    for name in bs.VARIANTEN:
        stl = os.path.join(stl_dir, f"druckplatte_balkenschuh_9mm_{name}_4x.stl")
        m = trimesh.load(stl)
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        zeichne(ax, [(platte, np.array([0.55, 0.55, 0.6]), 0.3), (m, STAHL, 1.0)],
                24, -70, f"Druckplatte {name}: 4 Stück, {np.degrees(dp.THETA):.0f}° angewinkelt, "
                f"Supports an Wandseite und Boden")
        fig.tight_layout()
        out = os.path.join(prev_dir, f"druckplatte_{name}_4x.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(out)

    # Nahansicht eines Teils mit Supports (Seitenansicht)
    m = dp.teil_mit_stuetzen("wand")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    zeichne(ax, [(m, STAHL, 1.0)], 8, -100, "Ein Teil mit Supports (Wandseite unten)")
    fig.tight_layout()
    out = os.path.join(prev_dir, "druckplatte_teil_nah.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    os.makedirs(prev_dir, exist_ok=True)
    was = sys.argv[1:] or ["ansicht", "masse", "druckplatten"]
    for w in was:
        globals()[w]()
