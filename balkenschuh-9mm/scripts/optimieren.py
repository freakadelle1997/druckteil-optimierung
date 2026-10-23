#!/usr/bin/env python3
"""
Balkenschuh 9 mm: Topologie-Optimierung (Resin sparen)
======================================================
Einsatz: Spant der Modellbahnanlage, Multiplex 9 x 130 x ca. 1000 mm, ein
Ende an der Wand (Variante wand), das andere an Möbel/Holz (Variante holz).
Das Brett wiegt etwa 0,8 kg (Birke-Multiplex ~680 kg/m³), dazu kommt der
Anteil der Anlage, die auf dem Spant liegt.

Lastannahme je Schuh (einstellbar unten):
  - Auflage:   150 N senkrecht auf den Taschenboden (Brett steht auf)
  - Schrauben: 150 N senkrecht über die beiden Brettschrauben (falls das
               Brett nicht aufsitzt)
  - seitlich:   30 N quer (Anstoßen), vom verschraubten Brett über die
               Innenseite der Brettschrauben-Ringe in eine Wange; die
               Gegenrichtung deckt die Spiegelsymmetrie ab. (Nicht über die
               ganze Wangenfläche verteilen: dann baut die Optimierung zu
               jedem Lastpunkt eine eigene Rippe, ein Kamm aus Zinken.)
  - Zug:        50 N nach vorne über die Brettschrauben (halbes Gewicht)
  - Führung:    20 N quer gegen die Innenseite der Stege an Vorder- und
               Oberkante der Wange. Ohne eigene Last tragen die Stege nichts,
               und die Optimierung lässt sie unverbunden (oberer Steg hing
               nur vorne, vorderer nur am Boden).

Lagerung: Die Schraubenköpfe mit Unterlegscheibe (wand: Ø 10) bzw. die
Senkköpfe (holz: Ø 7,4) klemmen die Lasche fest; die übrige Wandseite liegt
nur auf Druck an (hebt oben ab, wenn das Brett nach unten zieht).

Schutzbereiche (bleiben massiv): Taschenboden, Ringe um Wand- und
Brettschrauben, 3-mm-Steg an Vorder- und Oberkante der Wangen (führt das
Brett, keine ausfransenden Kanten).

Aufruf (Python 3.13):
  python3.13 optimieren.py wand            Optimieren, STL + Bilder
  python3.13 optimieren.py holz --vol 0.45
  python3.13 optimieren.py wand --nur-mesh Fläche aus gespeicherter Dichte neu
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

import balkenschuh_9mm as bs
from preview import zeichne, STAHL

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "werkzeuge"))
import topopt as to  # noqa: E402

# ----------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------
F_SENK = 150.0               # N je Schuh, senkrecht
F_SEIT = 30.0                # N je Schuh, quer
F_ZUG = 50.0                 # N je Schuh, nach vorne
F_FUEHRUNG = 20.0            # N je Schuh, quer gegen die Stege
H_RASTER = 1.0               # mm Elementgröße (Wände liegen exakt auf dem Raster)
RMIN = 2.0                   # mm Filterradius, kleinste Strebe etwa 4 mm
VOLUMEN = 0.5                # Anteil des Originalvolumens, der bleibt

KLEMM_R = {"wand": 5.0, "holz": 3.7}   # Scheibe Ø 10 / Senkkopf Ø 7,4
RING = 2.0                   # Schutzring um die Klemmfläche
BRETT_RING = 2.5             # Schutzring um die Brettlöcher
RAND = 3.0                   # Steg an Vorder- und Oberkante der Wangen
SITZE = 4                    # Teile je Druckplatte

opt_dir = os.path.abspath(os.path.join(here, "..", "optimierung"))
stl_dir = os.path.abspath(os.path.join(here, "..", "stl"))
prev_dir = os.path.abspath(os.path.join(here, "..", "previews"))


def randbedingungen(m, variante):
    tl = bs.VARIANTEN[variante]["t_lasche"]
    rk = KLEMM_R[variante]
    eps = 1e-6
    h = m.h

    # Lagerung: Klemmbereich der Wandschrauben fest, Wandseite nur auf Druck
    for sx in (1, -1):
        for z in bs.WAND_LOCH_Z:
            m.fest(m.knoten_wo(lambda X, Y, Z: (np.hypot(X - sx * bs.LOCH_X, Z - z) <= rk)
                               & (Y <= tl + eps)))
    m.kontakt(m.knoten_wo(lambda X, Y, Z: Y < eps), "y", +1)

    # Schutzbereiche
    m.behalten(to.quader([-bs.xa - 1, -1, -1], [bs.xa + 1, bs.TIEFE + 1, bs.T_BODEN]))
    for sx in (1, -1):
        for z in bs.WAND_LOCH_Z:
            m.behalten(to.zylinder((sx * bs.LOCH_X, -1, z), (sx * bs.LOCH_X, tl + 1, z), rk + RING))
    rb = bs.BRETT_LOCH / 2
    for z in bs.BRETT_LOCH_Z:
        m.behalten(to.zylinder((-bs.xa - 1, bs.BRETT_LOCH_Y, bs.T_BODEN + z),
                               (bs.xa + 1, bs.BRETT_LOCH_Y, bs.T_BODEN + z), rb + BRETT_RING))
    m.behalten(to.quader([-bs.xa - 1, bs.TIEFE - RAND, -1], [bs.xa + 1, bs.TIEFE + 1, bs.H + 1]))
    m.behalten(to.quader([-bs.xa - 1, -1, bs.H - RAND], [bs.xa + 1, bs.TIEFE + 1, bs.H + 1]))

    # Lasten
    m.last("Auflage", m.knoten_wo(lambda X, Y, Z: (np.abs(Z - bs.T_BODEN) < eps)
                                  & (np.abs(X) <= bs.xi + eps)), (0, 0, -F_SENK))

    def lochrand(X, Y, Z, zc):
        r = np.hypot(Y - bs.BRETT_LOCH_Y, Z - zc)
        return ((np.abs(X) >= bs.xi - eps) & (np.abs(X) <= bs.xa + eps)
                & (r >= rb - 0.8 * h) & (r <= rb + 0.8 * h))

    unten = np.concatenate([m.knoten_wo(lambda X, Y, Z: lochrand(X, Y, Z, bs.T_BODEN + z)
                                        & (Z <= bs.T_BODEN + z), True) for z in bs.BRETT_LOCH_Z])
    vorne = np.concatenate([m.knoten_wo(lambda X, Y, Z: lochrand(X, Y, Z, bs.T_BODEN + z)
                                        & (Y >= bs.BRETT_LOCH_Y), True) for z in bs.BRETT_LOCH_Z])
    m.last("Schrauben", unten, (0, 0, -F_SENK))
    m.last("Zug", vorne, (0, F_ZUG, 0), gewicht=0.5)
    ring = np.concatenate([m.knoten_wo(
        lambda X, Y, Z: (np.abs(X - bs.xi) < eps)
        & (np.hypot(Y - bs.BRETT_LOCH_Y, Z - bs.T_BODEN - z) <= rb + BRETT_RING), True)
        for z in bs.BRETT_LOCH_Z])
    m.last("seitlich", ring, (F_SEIT, 0, 0))
    steg = m.knoten_wo(lambda X, Y, Z: (np.abs(X - bs.xi) < eps) & (Z >= bs.T_BODEN - eps)
                       & ((Y >= bs.TIEFE - RAND - eps) | (Z >= bs.H - RAND - eps)), True)
    m.last("Führung", steg, (F_FUEHRUNG, 0, 0))


def zusammenfassung(name, m, r):
    zeilen = [f"{name}: Volumen {m.mesh.volume / 1000:.1f} cm³"]
    for fall, d in r.items():
        F = np.linalg.norm(m.lastfaelle[fall][0].reshape(-1, 3).sum(0))
        zeilen.append(
            f"  {fall:10s} {F:5.0f} N: Weg {d['weg_last']:.3f} mm (max {d['weg_max']:.3f}), "
            f"Zug max {d['s1_max']:5.1f} MPa (99 % {d['s1_p99']:4.1f}), "
            f"Mises max {d['vm_max']:5.1f}, Sicherheit {m.ws['zug'] / d['s1_max']:5.1f}, "
            f"Zug max inkl. Schutzbereiche {d['s1_max_alle']:5.1f}, "
            f"Wandkontakt {d['kontakt_aktiv']}/{d['kontakt_gesamt']}")
    print("\n".join(zeilen))
    return {fall: {k: float(v) for k, v in d.items() if np.isscalar(v)} for fall, d in r.items()}


def spannungsbild(ax, m, r, fall, vmax, titel, elev=18, azim=50):
    quads, el, nrm = m.aussenflaechen()
    vm = r[fall]["vm"][el]
    licht = np.array([0.4, -0.7, 0.6]) / np.linalg.norm([0.4, -0.7, 0.6])
    k = 0.45 + 0.55 * np.clip(nrm @ licht, 0, 1)
    farbe = plt.cm.turbo(np.clip(vm / vmax, 0, 1))
    farbe[:, :3] *= k[:, None]
    ax.add_collection3d(Poly3DCollection(quads, facecolors=farbe, edgecolors="none"))
    lo, hi = m.mesh.bounds
    mid, rr = (lo + hi) / 2, (hi - lo).max() / 2 / 1.25
    for a, f in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        f(mid[a] - rr, mid[a] + rr)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(titel, fontsize=11)


def bilder(tag, orig, neu, m0, r0, m1, r1):
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    zeichne(ax, [(orig, STAHL, 1.0)], 20, 55,
            f"Original {orig.volume / 1000:.1f} cm³", 1.25)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    zeichne(ax, [(neu, STAHL, 1.0)], 20, 55,
            f"Optimiert {neu.volume / 1000:.1f} cm³ ({neu.volume / orig.volume - 1:+.0%})", 1.25)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    zeichne(ax, [(neu, STAHL, 1.0)], 15, -125, "Optimiert, Wandseite", 1.25)
    fig.tight_layout()
    out = os.path.join(prev_dir, f"optimiert_{tag}_ansicht.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)

    vmax = max(np.percentile(r0["Auflage"]["vm"], 99.5), np.percentile(r1["Auflage"]["vm"], 99.5))
    fig = plt.figure(figsize=(14, 7))
    for i, (m, r, t) in enumerate(((m0, r0, "Original"), (m1, r1, "Optimiert"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        spannungsbild(ax, m, r, "Auflage", vmax,
                      f"{t}: Vergleichsspannung bei {F_SENK:.0f} N Auflage")
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="MPa (von Mises)")
    out = os.path.join(prev_dir, f"optimiert_{tag}_spannung.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variante", choices=list(bs.VARIANTEN))
    ap.add_argument("--vol", type=float, default=VOLUMEN)
    ap.add_argument("--h", type=float, default=H_RASTER)
    ap.add_argument("--rmin", type=float, default=RMIN)
    ap.add_argument("--nur-mesh", action="store_true")
    ap.add_argument("--ausfuehrlich", action="store_true", help="CG-Schritte je Lastfall zeigen")
    a = ap.parse_args()
    for d in (opt_dir, stl_dir, prev_dir):
        os.makedirs(d, exist_ok=True)
    v = a.variante
    tag = f"{v}_{round(100 * a.vol)}"
    dichte_datei = os.path.join(opt_dir, f"dichte_{tag}.npz")

    orig = bs.balkenschuh(v)
    m0 = to.Modell(orig, a.h, symmetrie_x=True)
    m0.ausfuehrlich = a.ausfuehrlich
    randbedingungen(m0, v)
    t0 = time.time()
    r0 = m0.pruefen()
    s0 = zusammenfassung("Original", m0, r0)
    if a.nur_mesh:
        xP = np.load(dichte_datei)["xP"]
        verlauf = []
    else:
        erg = m0.optimieren(a.vol, rmin=a.rmin, beta_max=16, stufe=20, max_iter=120)
        xP, verlauf = erg["xP"], erg["verlauf"]
        np.savez_compressed(dichte_datei, xP=xP)
    neu = m0.ergebnis_mesh(xP)
    m0.beenden()
    print(f"Fläche: wasserdicht {neu.is_watertight}, {len(neu.faces)} Dreiecke, "
          f"{neu.volume / 1000:.1f} cm³")

    m1 = to.Modell(neu, a.h, symmetrie_x=True)
    randbedingungen(m1, v)
    r1 = m1.pruefen()
    m1.beenden()
    s1 = zusammenfassung("Optimiert", m1, r1)

    spar = (orig.volume - neu.volume) / 1000
    print(f"Ersparnis: {spar:.1f} ml je Teil ({spar / orig.volume * 1000:.0%}), "
          f"{SITZE * spar:.0f} ml je Druckplatte ({SITZE} Stück), ohne Supports")

    neu.export(os.path.join(opt_dir, f"konstruktion_{tag}.stl"))   # für druckplatte.py
    flach = bs.druckausrichtung_flach(neu)
    out = os.path.join(stl_dir, f"balkenschuh_9mm_{tag}_optimiert.stl")
    flach.export(out)
    print(out)
    with open(os.path.join(opt_dir, f"bericht_{tag}.json"), "w") as f:
        json.dump(dict(variante=v, volumenanteil=a.vol, h=a.h, rmin=a.rmin,
                       lasten=dict(senkrecht=F_SENK, seitlich=F_SEIT, zug=F_ZUG,
                                   fuehrung=F_FUEHRUNG),
                       werkstoff=to.ABS_LIKE_3,
                       volumen_cm3=dict(original=orig.volume / 1000, optimiert=neu.volume / 1000),
                       original=s0, optimiert=s1, verlauf=verlauf,
                       dauer_min=(time.time() - t0) / 60), f, indent=1, default=float)
    bilder(tag, orig, neu, m0, r0, m1, r1)


if __name__ == "__main__":
    main()
