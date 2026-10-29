#!/usr/bin/env python3
"""
Randschutz-Halter 9 mm: Topologie-Optimierung (Resin sparen)
============================================================
Einsatz: Clip auf der Kante eines liegenden 9-mm-Trassenbretts, mit einer
Schraube durch die obere Backe gesichert. Im Schlitz steckt ein 3-mm-MDF-
Streifen hochkant als Absturzschutz für Loks.

Lastannahme je Clip (einstellbar unten):
  - Anstoßen außen: 10 N quer nach außen an der Oberkante des Streifens
               (Lok kippt dagegen), 30 mm über dem Schlitzboden. Der Streifen
               hebelt im 12 mm tiefen Schlitz: oben drückt er mit 25 N gegen
               die Außenwand, unten mit 15 N zurück gegen den Steg. Eine H0-Lok
               wiegt 3 bis 5 N und drückt gekippt nur einen Teil davon quer;
               der Streifen verteilt das zudem auf die Nachbarhalter. Mit 20 N
               hätte schon das Original nur Sicherheit 1,6 (Fuß der 2,5-mm-
               Außenwand).
  - Anstoßen innen: dasselbe von außen (Hand, Ärmel): oben gegen den Steg,
               unten gegen die Außenwand.
  - Einstecken: 30 N nach unten auf den Schlitzboden (Streifen an der
               Klemmrippe vorbeidrücken).
  - Klemmung:   10 N Dauerlast. Die Rippe in der Mitte drückt den Streifen
               gegen die beiden Enden des Stegs (Dreipunktklemmung), die
               Außenwand wird an der Rippe nach außen gedrückt.
  - Führung:    10 N nach unten auf die Spitze der unteren Backe (Brett
               drückt beim Aufschieben oder Hebeln dagegen).

Lagerung: Der Schraubenkopf zieht die obere Backe aufs Brett; um die
Schraube ist die Backenunterseite fest. Die übrige Backenunterseite liegt
nur auf Druck auf dem Brett, der Steg nur auf Druck an der Brettkante.

Schutzbereiche (bleiben massiv): Ring um die Schraube (Kopfauflage),
obere 3 mm beider Schlitzwände (führen den Streifen, Einführfasen),
Schlitzboden, Außenwand hinter der Klemmrippe, Spitze der unteren Backe.

Kein Bauraum-Modus: Der Clip füllt seinen Bauraum schon fast aus (Brett,
Schlitz, Schraubenzugang und Gleisseite müssen frei bleiben), ein Block
drumherum wäre kaum größer als das Original.

Aufruf (Python 3.13):
  python3.13 optimieren.py --vol 0.4          Optimieren, STL + Bilder
  python3.13 optimieren.py --vol 0.4 --nur-mesh   Fläche aus gespeicherter Dichte neu
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

import randschutz_halter as rh

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "werkzeuge"))
import topopt as to  # noqa: E402

# ----------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------
F_ANSTOSS = 10.0             # N quer an der Oberkante des Streifens (kurzzeitig)
STREIFEN_H = 30.0            # mm Hebelarm: Streifenoberkante über dem Schlitzboden
F_EINSTECK = 30.0            # N nach unten auf den Schlitzboden (kurzzeitig)
F_KLEMM = 10.0               # N Klemmkraft der Rippe (Dauerlast)
F_FUEHRUNG = 10.0            # N auf die Spitze der unteren Backe (kurzzeitig)
BAND = 2.0                   # mm Höhe der Kontaktbänder oben/unten im Schlitz

H_RASTER = 0.6               # mm Elementgröße (Außenwand nur 2,5 mm)
RMIN = 2.0                   # mm Filterradius, kleinste Strebe etwa 4 mm
VOLUMEN = 0.4                # Anteil des Originalvolumens, der bleibt
RUNDUNG = 1.0                # mm Weichzeichnen der Fläche (kleines Teil: weniger als beim Balkenschuh)
MAX_ITER = 120

KLEMM_R = 4.0                # Klemmfläche unter dem Schraubenkopf (Linsenkopf Ø 7)
RING = 4.5                   # Schutzring um die Schraube
RAND = 3.0                   # Führungssteg oben an den Schlitzwänden
SPITZE = 3.0                 # geschützte Spitze der unteren Backe
BODEN_T = 2.0                # geschützter Schlitzboden
STUECK = 10                  # Bezugsmenge für die Ersparnis (Druckplatte folgt)

DAUER = {"Klemmung"}         # Sicherheit >= 3, alle anderen kurzzeitig (>= 2)

opt_dir = os.path.abspath(os.path.join(here, "..", "optimierung"))
stl_dir = os.path.abspath(os.path.join(here, "..", "stl"))
prev_dir = os.path.abspath(os.path.join(here, "..", "previews"))

X0 = rh.L / 2                # Modell um x = 0 zentriert (Spiegelsymmetrie)
TEIL_FARBE = (0.88, 0.54, 0.24)


def original():
    m = rh.halter()
    m.apply_translation((-X0, 0, 0))
    return m


def randbedingungen(m):
    # Toleranz: das gerundete Profil verschiebt einzelne Flächen um wenige µm
    # (Schlitz bei y = 6,397 statt 6,4); das Raster folgt dem Netz
    eps = 0.02
    ys0, ys1 = rh.Y_SCHLITZ
    zb, zo = rh.BODEN, rh.OBEN
    zbk = rh.Z_BACKE

    # Lagerung
    m.fest(m.knoten_wo(lambda X, Y, Z: (np.abs(Z - zbk) < eps)
                       & (np.hypot(X, Y - rh.LOCH_Y) <= KLEMM_R)))
    m.kontakt(m.knoten_wo(lambda X, Y, Z: (np.abs(Z - zbk) < eps) & (Y < eps)), "z", +1)
    m.kontakt(m.knoten_wo(lambda X, Y, Z: (np.abs(Y) < eps) & (Z > -eps) & (Z < zbk + eps)), "y", +1)

    # Schutzbereiche
    m.behalten(to.zylinder((0, rh.LOCH_Y, zbk - 1), (0, rh.LOCH_Y, zbk + rh.T_OBEN + 1), RING))
    m.behalten(to.quader([-X0 - 1, -1, zo - RAND], [X0 + 1, rh.Y_AUSSEN + 1, zo + 1]))
    m.behalten(to.quader([-X0 - 1, ys0 - 0.5, zb - BODEN_T], [X0 + 1, ys1 + 0.5, zb]))
    m.behalten(to.quader([-2.5, ys1 - 1, zb - 1], [2.5, rh.Y_AUSSEN + 1, zo + 1]))
    m.behalten(to.quader([-X0 - 1, -rh.TIEFE_UNTEN - 1, rh.Z_UNTEN - 1],
                         [X0 + 1, -rh.TIEFE_UNTEN + SPITZE, 0]))

    # Hebel des Streifens im Schlitz: oben und unten je ein Kontaktband
    tiefe = zo - zb
    r_oben = F_ANSTOSS * STREIFEN_H / tiefe
    r_unten = r_oben - F_ANSTOSS

    def wand(y, z0, z1, x=lambda X: np.ones_like(X, bool)):
        return m.knoten_wo(lambda X, Y, Z: (np.abs(Y - y) < eps) & (Z >= z0 - eps)
                           & (Z <= z1 + eps) & x(X), True)

    oben = (zo - RAND, zo - rh.FASE - 0.2)
    unten = (zb, zb + BAND)
    m.last("Anstoßen außen", wand(ys1, *oben), (0, r_oben, 0))
    m.last("Anstoßen außen", wand(ys0, *unten), (0, -r_unten, 0))
    m.last("Anstoßen innen", wand(ys0, *oben), (0, -r_oben, 0))
    m.last("Anstoßen innen", wand(ys1, *unten), (0, r_unten, 0))

    m.last("Einstecken", m.knoten_wo(lambda X, Y, Z: (np.abs(Z - zb) < eps) & (Y > ys0 - eps)
                                     & (Y < ys1 + eps), True), (0, 0, -F_EINSTECK))

    m.last("Klemmung", wand(ys1, zb, zo - rh.FASE, lambda X: np.abs(X) <= 2.0),
           (0, F_KLEMM, 0), gewicht=0.5)
    m.last("Klemmung", wand(ys0, zb, zo - rh.FASE, lambda X: np.abs(X) >= X0 - 3.0),
           (0, -F_KLEMM, 0), gewicht=0.5)

    m.last("Führung", m.knoten_wo(lambda X, Y, Z: (np.abs(Z) < eps)
                                  & (Y <= -rh.TIEFE_UNTEN + SPITZE + eps), True),
           (0, 0, -F_FUEHRUNG), gewicht=0.5)


def zusammenfassung(name, m, r):
    zeilen = [f"{name}: Volumen {m.mesh.volume / 1000:.2f} cm³"]
    for fall, d in r.items():
        f = m.lastfaelle[fall][0]
        F = np.abs(f.reshape(-1, 3)).sum(0).max()
        weg = f @ d["u"] / F                  # arbeitsgleich, auch für Kräftepaare
        soll = 3 if fall in DAUER else 2
        sich = m.ws["zug"] / d["s1_max"]
        zeilen.append(
            f"  {fall:15s} {F:5.0f} N: Weg {weg:.3f} mm (max {d['weg_max']:.3f}), "
            f"Zug max {d['s1_max']:5.1f} MPa (99 % {d['s1_p99']:4.1f}), "
            f"Mises max {d['vm_max']:5.1f}, Sicherheit {sich:5.1f} (soll {soll}), "
            f"Zug max inkl. Schutzbereiche {d['s1_max_alle']:5.1f}, "
            f"Kontakt {d['kontakt_aktiv']}/{d['kontakt_gesamt']}")
    print("\n".join(zeilen))
    return {fall: {k: float(v) for k, v in d.items() if np.isscalar(v)} for fall, d in r.items()}


def zeichne(ax, mesh, elev, azim, titel):
    pc = Poly3DCollection(mesh.triangles)
    licht = np.clip(mesh.face_normals @ np.array([0.4, -0.5, 0.75]), 0.15, 1)
    pc.set_facecolor(np.c_[np.outer(licht, TEIL_FARBE), np.ones(len(licht))])
    pc.set_edgecolor("none")
    ax.add_collection3d(pc)
    lo, hi = mesh.bounds
    c, rr = (lo + hi) / 2, (hi - lo).max() / 2
    for a, f in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        f(c[a] - rr, c[a] + rr)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(titel, fontsize=11)


def spannungsbild(ax, m, r, fall, vmax, titel, elev=25, azim=-125):
    quads, el, nrm = m.aussenflaechen()
    vm = r[fall]["vm"][el]
    licht = np.array([0.4, -0.7, 0.6]) / np.linalg.norm([0.4, -0.7, 0.6])
    k = 0.45 + 0.55 * np.clip(nrm @ licht, 0, 1)
    farbe = plt.cm.turbo(np.clip(vm / vmax, 0, 1))
    farbe[:, :3] *= k[:, None]
    ax.add_collection3d(Poly3DCollection(quads, facecolors=farbe, edgecolors="none"))
    lo, hi = m.mesh.bounds
    mid, rr = (lo + hi) / 2, (hi - lo).max() / 2
    for a, f in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        f(mid[a] - rr, mid[a] + rr)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(titel, fontsize=11)


def bilder(tag, orig, neu, m0, r0, m1, r1):
    fig = plt.figure(figsize=(18, 7))
    zeichne(fig.add_subplot(1, 3, 1, projection="3d"), orig, 28, -125,
            f"Original {orig.volume / 1000:.2f} cm³")
    zeichne(fig.add_subplot(1, 3, 2, projection="3d"), neu, 28, -125,
            f"Optimiert {neu.volume / 1000:.2f} cm³ ({neu.volume / orig.volume - 1:+.0%})")
    zeichne(fig.add_subplot(1, 3, 3, projection="3d"), neu, 20, 55, "Optimiert, Schlitzseite")
    fig.tight_layout()
    out = os.path.join(prev_dir, f"optimiert_{tag}_ansicht.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)

    fall = "Anstoßen außen"
    vmax = max(np.percentile(r0[fall]["vm"], 99.5), np.percentile(r1[fall]["vm"], 99.5))
    fig = plt.figure(figsize=(14, 7))
    for i, (m, r, t) in enumerate(((m0, r0, "Original"), (m1, r1, "Optimiert"))):
        spannungsbild(fig.add_subplot(1, 2, i + 1, projection="3d"), m, r, fall, vmax,
                      f"{t}: Vergleichsspannung, {F_ANSTOSS:.0f} N {fall}")
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="MPa (von Mises)")
    out = os.path.join(prev_dir, f"optimiert_{tag}_spannung.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", type=float, default=VOLUMEN)
    ap.add_argument("--h", type=float, default=H_RASTER)
    ap.add_argument("--rmin", type=float, default=RMIN)
    ap.add_argument("--rundung", type=float, default=RUNDUNG)
    ap.add_argument("--iter", type=int, default=MAX_ITER, help="höchstens so viele Schritte")
    ap.add_argument("--nur-mesh", action="store_true",
                    help="Fläche, Nachrechnung und Bilder aus gespeicherter Dichte (optimierung/dichte_*.npz)")
    ap.add_argument("--ausfuehrlich", action="store_true", help="CG-Schritte je Lastfall zeigen")
    a = ap.parse_args()
    for d in (opt_dir, stl_dir, prev_dir):
        os.makedirs(d, exist_ok=True)
    tag = f"v{round(100 * a.vol)}"
    dichte_datei = os.path.join(opt_dir, f"dichte_{tag}.npz")

    orig = original()
    t0 = time.time()
    m0 = to.Modell(orig, a.h, symmetrie_x=True)
    m0.ausfuehrlich = a.ausfuehrlich
    randbedingungen(m0)
    r0 = m0.pruefen()
    s0 = zusammenfassung("Original", m0, r0)
    if a.nur_mesh:
        xP = np.load(dichte_datei)["xP"]
        verlauf = []
    else:
        erg = m0.optimieren(a.vol, rmin=a.rmin, beta_max=16, stufe=20, max_iter=a.iter)
        xP, verlauf = erg["xP"], erg["verlauf"]
        np.savez_compressed(dichte_datei, xP=xP)
    neu = m0.ergebnis_mesh(xP, rundung=a.rundung)
    m0.beenden()
    print(f"Fläche: wasserdicht {neu.is_watertight}, {len(neu.faces)} Dreiecke, "
          f"{neu.volume / 1000:.2f} cm³")

    m1 = to.Modell(neu, a.h, symmetrie_x=True)
    randbedingungen(m1)
    r1 = m1.pruefen()
    m1.beenden()
    s1 = zusammenfassung("Optimiert", m1, r1)

    spar = (orig.volume - neu.volume) / 1000
    print(f"Ersparnis: {spar:.2f} ml je Teil ({spar / orig.volume * 1000:.0%}), "
          f"{STUECK * spar:.1f} ml je {STUECK} Stück, ohne Supports")

    einbau = neu.copy()
    einbau.apply_translation((X0, 0, 0))          # wieder wie randschutz_halter_9mm.stl
    einbau.export(os.path.join(opt_dir, f"konstruktion_{tag}.stl"))
    out = os.path.join(stl_dir, f"randschutz_halter_9mm_{tag}_optimiert.stl")
    einbau.export(out)
    print(out)
    with open(os.path.join(opt_dir, f"bericht_{tag}.json"), "w") as f:
        json.dump(dict(volumenanteil=a.vol, h=a.h, rmin=a.rmin, rundung=a.rundung,
                       lasten=dict(anstoss=F_ANSTOSS, streifen_h=STREIFEN_H, einstecken=F_EINSTECK,
                                   klemmung=F_KLEMM, fuehrung=F_FUEHRUNG),
                       dauerlast=sorted(DAUER), werkstoff=to.ABS_LIKE_3,
                       volumen_cm3=dict(original=orig.volume / 1000, optimiert=neu.volume / 1000),
                       original=s0, optimiert=s1, verlauf=verlauf,
                       dauer_min=(time.time() - t0) / 60), f, indent=1, default=float)
    bilder(tag, orig, neu, m0, r0, m1, r1)


if __name__ == "__main__":
    main()
