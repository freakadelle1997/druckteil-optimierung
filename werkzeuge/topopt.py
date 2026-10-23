#!/usr/bin/env python3
"""
Topologie-Optimierung für Druckteile (Voxel-FEM, SIMP)
======================================================
Nimmt ein fertiges Bauteil (trimesh, wasserdicht, Konstruktionslage in mm),
legt ein achsparalleles Quaderraster darüber und entfernt Material dort, wo
es kaum Last trägt. Ablauf:

  1. Lager und Lasten:  fest(), kontakt(), last()
  2. Schutzbereiche:    behalten()  (Löcher, Auflagen bleiben massiv)
  3. Optimieren:        optimieren(volumenanteil), nur der Lastpfad bleibt
  4. Geometrie:         ergebnis_mesh(), Original ∩ Dichtefeld ∪ Schutzbereiche,
                        Originalflächen bleiben also exakt, wo Material bleibt
  5. Nachrechnen:       pruefen() am fertigen Teil, mit Wandkontakt

Rechenmodell: lineare Elastizität, Quader-Elemente (Hex8 mit inkompatiblen
Moden, damit dünne Wände auch mit 3 Elementen über die Dicke richtig biegen),
Wände liegen exakt auf Rasterlinien. Wandkontakt nur auf Druck (Federn, die
bei Abheben abgeschaltet werden). Löser: CG mit algebraischem Mehrgitter
(pyamg). Spannungen an Elementmitten, also Näherung, kein Festigkeitsnachweis.

Einheiten: mm, N, MPa.
Python: /opt/homebrew/bin/python3.13 (numpy, scipy, trimesh, manifold3d,
pyamg, scikit-image).

Selbsttest (Kragbalken gegen Handrechnung):  python3.13 topopt.py
"""

import atexit
import itertools
import os
import time

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import manifold3d
import pyamg
import trimesh
from skimage.measure import marching_cubes

# Herstellerwerte Anycubic ABS-Like Resin 3.0: Zugfestigkeit 38-45 MPa,
# E-Modul 1300-1700 MPa, Bruchdehnung 22-30 %. Gerechnet wird mit den
# unteren Werten; Querkontraktion geschätzt (übliche Kunststoffe 0,35-0,40).
ABS_LIKE_3 = dict(name="Anycubic ABS-Like 3.0", E=1300.0, nu=0.38, zug=38.0)

ECKEN = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
ACHSE = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------------------
# Element
# ----------------------------------------------------------------------------
def b_matrix(dN):
    """Verzerrungs-Verschiebungs-Matrix, Voigt [xx, yy, zz, yz, xz, xy]."""
    n = len(dN)
    B = np.zeros((6, 3 * n))
    for i, (x, y, z) in enumerate(dN):
        B[0, 3 * i], B[1, 3 * i + 1], B[2, 3 * i + 2] = x, y, z
        B[3, 3 * i + 1], B[3, 3 * i + 2] = z, y
        B[4, 3 * i], B[4, 3 * i + 2] = z, x
        B[5, 3 * i], B[5, 3 * i + 1] = y, x
    return B


def stoffmatrix(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    return D


def hex8(a, b, c, D):
    """Quader a x b x c: Steifigkeit (24x24, mit 9 inkompatiblen Moden
    statisch kondensiert) und B-Matrix in der Mitte (6x24)."""
    s = 2.0 * ECKEN - 1.0
    J = np.array([a, b, c]) / 2
    g = 1 / np.sqrt(3)
    Kuu, Kua, Kaa = np.zeros((24, 24)), np.zeros((24, 9)), np.zeros((9, 9))

    def ableitungen(xi):
        f = 1 + s * xi
        dN = np.column_stack([s[:, 0] * f[:, 1] * f[:, 2],
                              f[:, 0] * s[:, 1] * f[:, 2],
                              f[:, 0] * f[:, 1] * s[:, 2]]) / 8
        return dN / J

    for xi in itertools.product((-g, g), repeat=3):
        xi = np.array(xi)
        B = b_matrix(ableitungen(xi))
        Ba = b_matrix(np.diag(-2 * xi / J))      # Moden 1 - xi_m^2
        w = a * b * c / 8
        Kuu += B.T @ D @ B * w
        Kua += B.T @ D @ Ba * w
        Kaa += Ba.T @ D @ Ba * w
    K = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
    return 0.5 * (K + K.T), b_matrix(ableitungen(np.zeros(3)))


# ----------------------------------------------------------------------------
# Raster
# ----------------------------------------------------------------------------
def raster_achsen(mesh, h, min_abstand=None, symmetrie_x=False):
    """Rasterlinien je Achse: achsparallele Flächen liegen genau auf einer
    Linie (Wände behalten ihre echte Dicke), dazwischen gleichmäßig <= h.
    symmetrie_x: Raster spiegelgleich zu x = 0 (für symmetrische Teile)."""
    min_abstand = 0.4 * h if min_abstand is None else min_abstand

    def verdichten(werte):
        bp = [werte[0]]
        for w in werte[1:-1]:
            if w - bp[-1] >= min_abstand and werte[-1] - w >= min_abstand:
                bp.append(w)
        bp.append(werte[-1])
        return np.array(bp)

    achsen = []
    for ax in range(3):
        eben = np.abs(np.abs(mesh.face_normals[:, ax]) - 1) < 1e-6
        werte = np.unique(np.round(np.r_[mesh.triangles[eben][:, 0, ax],
                                         mesh.bounds[:, ax]], 4))
        if ax == 0 and symmetrie_x:
            pos = np.unique(np.abs(werte))
            bp = verdichten(np.r_[0.0, pos[pos > 0]])
            bp = np.r_[-bp[:0:-1], bp]
        else:
            bp = verdichten(werte)
        linien = [bp[0]]
        for a, b in zip(bp[:-1], bp[1:]):
            n = max(1, int(np.ceil((b - a) / h - 1e-6)))
            linien.extend(np.linspace(a, b, n + 1)[1:])
        achsen.append(np.array(linien))
    return achsen


def innen_maske(mesh, achsen):
    """Welche Zellmitten liegen im (wasserdichten) Mesh? Strahl entlang z,
    Schnittpunkte zählen."""
    xs, ys, zs = [0.5 * (a[1:] + a[:-1]) for a in achsen]
    px, py = xs + 1.37e-5, ys + 2.71e-5          # nie genau auf einer Kante
    nx, ny, nz = len(xs), len(ys), len(zs)
    zaehl = np.zeros((nx * ny, nz + 1), np.int32)
    tri = mesh.triangles
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    D = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    lo = np.minimum(np.minimum(A, B), C)
    hi = np.maximum(np.maximum(A, B), C)
    i0, i1 = np.searchsorted(px, lo[:, 0]), np.searchsorted(px, hi[:, 0])
    j0, j1 = np.searchsorted(py, lo[:, 1]), np.searchsorted(py, hi[:, 1])
    for t in np.nonzero((np.abs(D) > 1e-12) & (i1 > i0) & (j1 > j0))[0]:
        a, b, c, d = A[t], B[t], C[t], D[t]
        X, Y = np.meshgrid(px[i0[t]:i1[t]], py[j0[t]:j1[t]], indexing="ij")
        w0 = ((b[0] - X) * (c[1] - Y) - (b[1] - Y) * (c[0] - X)) / d
        w1 = ((c[0] - X) * (a[1] - Y) - (c[1] - Y) * (a[0] - X)) / d
        w2 = 1 - w0 - w1
        drin = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not drin.any():
            continue
        z = (w0 * a[2] + w1 * b[2] + w2 * c[2])[drin]
        ii, jj = np.nonzero(drin)
        np.add.at(zaehl, ((ii + i0[t]) * ny + jj + j0[t], np.searchsorted(zs, z)), 1)
    return (np.cumsum(zaehl, axis=1)[:, :nz] % 2 == 1).reshape(nx, ny, nz)


def punkte_innen(mesh, P):
    """Liegen die Punkte P (n, 3) im wasserdichten Mesh? Strahl nach +z,
    Schnittpunkte zählen (ohne rtree/embree)."""
    P = np.asarray(P, float) + [1.37e-5, 2.71e-5, 0]
    tri = mesh.triangles
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    D = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    ok = np.abs(D) > 1e-12
    A, B, C, D = A[ok], B[ok], C[ok], D[ok]
    drin = np.zeros(len(P), bool)
    for k, (x, y, z) in enumerate(P):
        w0 = ((B[:, 0] - x) * (C[:, 1] - y) - (B[:, 1] - y) * (C[:, 0] - x)) / D
        w1 = ((C[:, 0] - x) * (A[:, 1] - y) - (C[:, 1] - y) * (A[:, 0] - x)) / D
        w2 = 1 - w0 - w1
        t = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        zs = w0[t] * A[t, 2] + w1[t] * B[t, 2] + w2[t] * C[t, 2]
        drin[k] = (zs > z).sum() % 2 == 1
    return drin


def vereinfachen(mesh, toleranz=0.02):
    """Dreiecke einsparen, ohne die Form um mehr als toleranz [mm] zu ändern
    (Fläche aus dem Dichtefeld ist sonst unnötig fein, STL ~4x größer)."""
    m = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    o = m.simplify(toleranz).to_mesh()
    return trimesh.Trimesh(o.vert_properties[:, :3], o.tri_verts, process=False)


def zylinder(p0, p1, r, seg=48):
    """Zylinder von p0 nach p1 (für Schutzbereiche)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    z = trimesh.creation.cylinder(radius=r, height=np.linalg.norm(d), sections=seg)
    tf = trimesh.geometry.align_vectors([0, 0, 1], d / np.linalg.norm(d))
    tf[:3, 3] = (p0 + p1) / 2
    z.apply_transform(tf)
    return z


def quader(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


# ----------------------------------------------------------------------------
# Löser
# ----------------------------------------------------------------------------
def _matrix(d, spalten, indptr, ndof):
    return sp.bsr_matrix((d.reshape(-1, 3, 3), spalten, indptr), shape=(ndof, ndof))


def _cg(A, b, x0, tol, c, B, aktiv):
    """CG mit Mehrgitter-Vorkonditionierer c (Cache je Lastfall). Neu aufbauen,
    wenn CG damit deutlich langsamer wird als direkt nach dem Aufbau oder sich
    der Wandkontakt nennenswert geändert hat."""
    if (c is None or c["schritte"] > 1.5 * c["start"] + 10
            or (c["maske"] != aktiv).mean() > 0.005):
        ml = pyamg.smoothed_aggregation_solver(
            A, B=B, max_coarse=1000,
            smooth=("energy", {"krylov": "cg", "maxiter": 2, "degree": 1, "weighting": "local"}))
        c = dict(M=ml.aspreconditioner(cycle="V"), start=None, schritte=0, maske=aktiv.copy())
    res = []
    u, _ = pyamg.krylov.cg(A, b, x0=x0, tol=tol, maxiter=1000, M=c["M"], residuals=res)
    c["schritte"] = len(res)
    c["start"] = c["start"] or len(res)
    return u, c, len(res), res[-1] / np.linalg.norm(b)


def _arbeiter(verbindung, shm_name, n, ndof, spalten, indptr, B, k_idx, k_pen):
    """Eigener Prozess je Lastfall: liest die Grundmatrix aus gemeinsamem
    Speicher, ergänzt seine Kontaktfedern und löst."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    basis = np.ndarray((n,), np.float64, buffer=shm.buf)
    c = None
    try:
        while (auftrag := verbindung.recv()) is not None:
            aktiv, b, x0, tol = auftrag
            d = basis.copy()
            np.add.at(d, k_idx, k_pen * aktiv)
            u, c, schritte, rest = _cg(_matrix(d, spalten, indptr, ndof), b, x0, tol, c, B, aktiv)
            verbindung.send((u, schritte, rest))
    finally:
        shm.close()


# ----------------------------------------------------------------------------
# Modell
# ----------------------------------------------------------------------------
class Modell:
    """Voxel-FE-Modell eines Bauteils auf einem achsparallelen Raster."""

    def __init__(self, mesh, h, werkstoff=ABS_LIKE_3, symmetrie_x=False, parallel=None,
                 protokoll=print):
        """symmetrie_x: Teil ist spiegelsymmetrisch zu x = 0; dann wird auch
        das Raster symmetrisch und optimieren() hält das Ergebnis symmetrisch.
        parallel: Lastfälle in eigenen Prozessen lösen (braucht je Lastfall
        einige 100 MB); Vorgabe aus der Umgebungsvariable TOPOPT_PARALLEL=1."""
        t0 = time.time()
        self.parallel = os.environ.get("TOPOPT_PARALLEL") == "1" if parallel is None else parallel
        self._prozesse = {}
        self.mesh, self.h, self.ws, self.log = mesh, h, werkstoff, protokoll
        self.symmetrie_x = symmetrie_x
        self.achsen = raster_achsen(mesh, h, symmetrie_x=symmetrie_x)
        self.innen = self._maske(mesh)
        nx, ny, nz = self.innen.shape
        self.ijk = np.argwhere(self.innen)
        nel = len(self.ijk)
        d = [np.diff(a) for a in self.achsen]
        groesse = np.column_stack([d[a][self.ijk[:, a]] for a in range(3)])
        self.vol = groesse.prod(1)
        mitte = [0.5 * (a[1:] + a[:-1]) for a in self.achsen]
        self.zentren = np.column_stack([mitte[a][self.ijk[:, a]] for a in range(3)])

        self.D = stoffmatrix(werkstoff["E"], werkstoff["nu"])
        typen, self.typ = np.unique(np.round(groesse, 5), axis=0, return_inverse=True)
        self.typ = self.typ.ravel()
        ke, b0 = zip(*(hex8(*t, self.D / werkstoff["E"]) for t in typen))
        self.KE, self.B0 = np.array(ke), np.array(b0)          # für E = 1
        self.typ_idx = [np.nonzero(self.typ == t)[0] for t in range(len(typen))]

        nijk = self.ijk[:, None, :] + ECKEN[None]
        gid = (nijk[..., 0] * (ny + 1) + nijk[..., 1]) * (nz + 1) + nijk[..., 2]
        uid, inv = np.unique(gid, return_inverse=True)
        self.enode = inv.reshape(nel, 8)
        k = uid % (nz + 1)
        j = (uid // (nz + 1)) % (ny + 1)
        i = uid // ((nz + 1) * (ny + 1))
        self.knoten = np.column_stack([self.achsen[0][i], self.achsen[1][j], self.achsen[2][k]])
        self.edof = (3 * self.enode[:, :, None] + np.arange(3)).reshape(nel, 24)
        self.ndof = 3 * len(uid)
        self.nachbarn = np.bincount(self.enode.ravel(), minlength=len(uid))

        self.fest_dof = np.zeros(self.ndof, bool)
        self.kontakte = []
        self.lastfaelle = {}
        self.passiv = np.zeros(nel, bool)
        self.schutz = []
        self.ausfuehrlich = False
        self._bereit = False
        self.log(f"Modell: {nel} Elemente ({len(typen)} Größen), {self.ndof} Freiheitsgrade, "
                 f"Raster {nx}x{ny}x{nz}, Volumen {self.vol.sum() / 1000:.2f} cm³ "
                 f"(Mesh {mesh.volume / 1000:.2f}), {time.time() - t0:.1f} s")

    def _maske(self, mesh):
        """Zellbelegung; bei Symmetrie spiegelgleich (Zellmitten genau auf
        schrägen Flächen fallen sonst links und rechts verschieden aus)."""
        m = innen_maske(mesh, self.achsen)
        return m & m[::-1] if self.symmetrie_x else m

    # --- Randbedingungen ----------------------------------------------------
    def knoten_wo(self, bedingung, nur_oberflaeche=False):
        """Knoten-Indizes, für die bedingung(X, Y, Z) wahr ist."""
        X, Y, Z = self.knoten.T
        m = np.asarray(bedingung(X, Y, Z), bool)
        if nur_oberflaeche:
            m &= self.nachbarn < 8
        return np.nonzero(m)[0]

    def fest(self, knoten, richtungen="xyz"):
        """Feste Lagerung (z. B. verschraubte Bereiche)."""
        for r in richtungen:
            self.fest_dof[3 * knoten + ACHSE[r]] = True
        self._bereit = False

    def kontakt(self, knoten, achse, vorzeichen=+1):
        """Auflage nur auf Druck, z. B. an der Wand. vorzeichen: Richtung, in
        die die Auflage das Teil drückt (Wand bei y = 0, Teil bei y > 0: +1)."""
        self.kontakte.append((3 * knoten + ACHSE[achse], vorzeichen))
        self._bereit = False

    def last(self, fall, knoten, kraft, gewicht=1.0):
        """Gesamtkraft [N] als Vektor, gleichmäßig auf die Knoten verteilt.
        Mehrere Aufrufe mit demselben Fall addieren sich."""
        if len(knoten) == 0:
            raise ValueError(f"Lastfall {fall}: keine Knoten gefunden")
        f = self.lastfaelle.setdefault(fall, [np.zeros(self.ndof), gewicht])[0]
        for a in range(3):
            np.add.at(f, 3 * knoten + a, kraft[a] / len(knoten))

    def behalten(self, koerper):
        """Schutzbereich (trimesh-Volumen): bleibt massiv."""
        # nicht einzeln spiegeln: ein Ring rechts hat sein Gegenstück in einem
        # anderen Körper; die Symmetrie des Ganzen stellt optimieren() her
        maske = innen_maske(koerper, self.achsen)[tuple(self.ijk.T)]
        self.passiv |= maske
        self.schutz.append(koerper)
        self._schutz_masken = getattr(self, "_schutz_masken", []) + [maske]

    # --- Gleichungssystem -----------------------------------------------------
    def _vorbereiten(self):
        """Matrixmuster als 3x3-Blöcke je Knotenpaar über alle Freiheitsgrade;
        feste Freiheitsgrade werden Einheitszeilen. Mit Blöcken und
        Starrkörpermoden konvergiert das Mehrgitter bei dünnen Wänden etwa
        doppelt so schnell wie mit gestrichenen Zeilen."""
        t0 = time.time()
        nn, nel = self.ndof // 3, len(self.enode)
        zk = np.repeat(self.enode, 8, axis=1)
        sk = np.tile(self.enode, (1, 8))
        uk, inv = np.unique(zk * nn + sk, return_inverse=True)
        inv = inv.reshape(nel, 64)
        del zk, sk
        zeilen, self._bl_spalten = uk // nn, (uk % nn).astype(np.int32)
        self._indptr = np.searchsorted(zeilen, np.arange(nn + 1)).astype(np.int32)
        self._nbl = len(uk)
        lk, lr = np.divmod(np.arange(24), 3)
        paar = (lk[:, None] * 8 + lk[None, :]).ravel()
        pos = (lr[:, None] * 3 + lr[None, :]).ravel()
        self._idx = (inv[:, paar] * 9 + pos).astype(np.int32)

        z = self.fest_dof[3 * zeilen[:, None, None] + np.arange(3)[None, :, None]]
        s = self.fest_dof[3 * self._bl_spalten[:, None, None] + np.arange(3)[None, None, :]]
        self._null = np.nonzero((z | s).ravel())[0]
        diag_bl = np.searchsorted(uk, np.arange(nn) * (nn + 1))
        self._diag = (diag_bl[:, None] * 9 + 4 * np.arange(3)).ravel()

        if self.kontakte:
            kd = np.concatenate([d for d, _ in self.kontakte])
            kv = np.concatenate([np.full(len(d), v) for d, v in self.kontakte])
        else:
            kd, kv = np.zeros(0, int), np.zeros(0)
        ok = ~self.fest_dof[kd]
        self.k_dof, self.k_vz = kd[ok], kv[ok]
        voll = self._daten(np.full(nel, self.ws["E"]))
        self._d0 = np.median(voll[self._diag[~self.fest_dof]])
        # Federsteifigkeit 10 x typische Knotensteifigkeit: Eindringen < 0,001 mm
        self.k_pen = 10 * self._d0

        x, y, z = (self.knoten - self.knoten.mean(0)).T
        B = np.zeros((self.ndof, 6))
        B[0::3, 0] = B[1::3, 1] = B[2::3, 2] = 1
        B[1::3, 3], B[2::3, 3] = -z, y
        B[0::3, 4], B[2::3, 4] = z, -x
        B[0::3, 5], B[1::3, 5] = -y, x
        self._B = B
        self._amg = {}
        if self.parallel:
            self._prozesse_starten()
        self._bereit = True
        self.log(f"Gleichungssystem: {self.ndof - self.fest_dof.sum()} Unbekannte, "
                 f"{self._nbl * 9 / 1e6:.1f} Mio. Einträge, {len(self.k_dof)} Kontaktfedern, "
                 f"{time.time() - t0:.1f} s")

    def _daten(self, E):
        werte = self.KE.reshape(len(self.KE), -1)[self.typ] * E[:, None]
        return np.bincount(self._idx.ravel(), weights=werte.ravel(), minlength=self._nbl * 9)

    def _prozesse_starten(self):
        from multiprocessing import get_context, shared_memory
        self.beenden()
        n = self._nbl * 9
        self._shm = shared_memory.SharedMemory(create=True, size=n * 8)
        self._basis = np.ndarray((n,), np.float64, buffer=self._shm.buf)
        ctx = get_context("spawn")
        for fall in self.lastfaelle:
            hier, dort = ctx.Pipe()
            p = ctx.Process(target=_arbeiter, daemon=True,
                            args=(dort, self._shm.name, n, self.ndof, self._bl_spalten,
                                  self._indptr, self._B, self._diag[self.k_dof], self.k_pen))
            p.start()
            self._prozesse[fall] = (hier, p)
        atexit.register(self.beenden)
        self.log(f"  {len(self._prozesse)} Löser-Prozesse gestartet")

    def beenden(self):
        """Löser-Prozesse und gemeinsamen Speicher freigeben."""
        for hier, p in self._prozesse.values():
            try:
                hier.send(None)
            except (BrokenPipeError, OSError):
                pass
            p.join(timeout=10)
        self._prozesse = {}
        if getattr(self, "_shm", None) is not None:
            self._basis = None
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def loesen(self, E, aktiv, u0=None, tol=1e-5):
        """Verschiebungen je Lastfall; aktiv[fall] = Maske der Kontaktfedern."""
        if not self._bereit:
            self._vorbereiten()
        basis = self._daten(E)
        basis[self._null] = 0
        basis[self._diag[self.fest_dof]] = self._d0
        t0 = time.time()
        rechte = {fall: np.where(self.fest_dof, 0.0, f) for fall, (f, _) in self.lastfaelle.items()}
        if self.parallel:
            self._basis[:] = basis
            for fall, (hier, _) in self._prozesse.items():
                hier.send((aktiv[fall], rechte[fall], None if u0 is None else u0[fall], tol))
            antworten = {fall: hier.recv() for fall, (hier, _) in self._prozesse.items()}
        else:
            antworten = {}
            for fall in self.lastfaelle:
                d = basis.copy()
                np.add.at(d, self._diag[self.k_dof], self.k_pen * aktiv[fall])
                A = _matrix(d, self._bl_spalten, self._indptr, self.ndof)
                u, self._amg[fall], n, rest = _cg(A, rechte[fall], None if u0 is None else u0[fall],
                                                  tol, self._amg.get(fall), self._B, aktiv[fall])
                antworten[fall] = (u, n, rest)
        U = {}
        for fall, (u, n, rest) in antworten.items():
            u[self.fest_dof] = 0
            if self.ausfuehrlich:
                self.log(f"    {fall}: {n} CG-Schritte, Rest {rest:.1e}, "
                         f"Kontakt {int(aktiv[fall].sum())}/{len(aktiv[fall])}")
            if rest > 10 * tol:
                self.log(f"  Warnung: Löser für {fall} nicht konvergiert ({n} Schritte, Rest {rest:.1e})")
            U[fall] = u
        if self.ausfuehrlich:
            self.log(f"    Lösen gesamt {time.time() - t0:.1f} s")
        return U

    def kontakt_update(self, u, aktiv):
        """Abhebende Federn aus, eindringende ein."""
        un = self.k_vz * u[self.k_dof]
        neu = aktiv.copy()
        neu[aktiv & (un > 0)] = False
        neu[~aktiv & (un < 0)] = True
        return neu

    def energie(self, u):
        """u_e^T K_e u_e je Element (für E = 1)."""
        ue = u[self.edof]
        ce = np.empty(len(ue))
        for t, idx in enumerate(self.typ_idx):
            v = ue[idx]
            ce[idx] = np.einsum("ij,jk,ik->i", v, self.KE[t], v)
        return ce

    def spannungen(self, u, E=None):
        """Von-Mises- und größte Hauptspannung je Element (Elementmitte)."""
        ue = u[self.edof]
        s = np.empty((len(ue), 6))
        for t, idx in enumerate(self.typ_idx):
            s[idx] = ue[idx] @ (self.D @ self.B0[t]).T
        if E is not None:
            s *= (E / self.ws["E"])[:, None]
        xx, yy, zz, yz, xz, xy = s.T
        vm = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                     + 3 * (yz ** 2 + xz ** 2 + xy ** 2))
        T = np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1),
                      np.stack([xz, yz, zz], -1)], -2)
        return vm, np.linalg.eigvalsh(T)[:, -1]

    # --- Optimierung ------------------------------------------------------------
    def optimieren(self, volumenanteil, rmin=2.0, penal=3.0, max_iter=100,
                   beta_max=8.0, stufe=25, kontakt_alle=1):
        """SIMP mit Dichtefilter (Radius rmin, kleinste Strebe etwa 2 x rmin)
        und Heaviside-Projektion. Ziel: steifstes Teil (Summe der normierten
        Nachgiebigkeiten aller Lastfälle) bei volumenanteil des Originals."""
        t_start = time.time()
        nel, E0 = len(self.vol), self.ws["E"]
        Emin = 1e-6 * E0
        tree = cKDTree(self.zentren)
        paare = tree.query_pairs(rmin, output_type="ndarray")
        w = rmin - np.linalg.norm(self.zentren[paare[:, 0]] - self.zentren[paare[:, 1]], axis=1)
        i = np.r_[paare[:, 0], paare[:, 1], np.arange(nel)]
        j = np.r_[paare[:, 1], paare[:, 0], np.arange(nel)]
        H = sp.csr_matrix((np.r_[w, w, np.full(nel, rmin)], (i, j)), shape=(nel, nel))
        Hv = (H @ sp.diags(self.vol)).tocsr()
        HvT = Hv.T.tocsr()
        Hs = np.asarray(Hv.sum(1)).ravel()

        spiegel = None
        if self.symmetrie_x:
            dist, idx = tree.query(self.zentren * [-1, 1, 1])
            if dist.max() < 1e-3:
                spiegel = idx
                self.passiv |= self.passiv[spiegel]
            else:
                self.log("Hinweis: Teil nicht spiegelsymmetrisch in x, Symmetrie aus")

        eta = 0.5

        def proj(xT, beta):
            d = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
            return (np.tanh(beta * eta) + np.tanh(beta * (xT - eta))) / d

        def dproj(xT, beta):
            d = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
            return beta * (1 - np.tanh(beta * (xT - eta)) ** 2) / d

        def physisch(x, beta):
            xP = proj(Hv @ x / Hs, beta)
            xP[self.passiv] = 1
            return xP

        # Schutzbereich ohne Last und ohne Lagerung trägt nichts: die
        # Optimierung bindet ihn dann nicht an (er hängt am Ende frei)
        belastet = np.zeros(self.ndof // 3, bool)
        for f, _ in self.lastfaelle.values():
            belastet |= np.abs(f.reshape(-1, 3)).sum(1) > 0
        belastet |= self.fest_dof.reshape(-1, 3).any(1)
        for i, maske in enumerate(getattr(self, "_schutz_masken", [])):
            if not belastet[np.unique(self.enode[maske])].any():
                lo, hi = self.schutz[i].bounds
                self.log(f"Warnung: Schutzbereich {i + 1} ({np.round(lo, 1)} bis {np.round(hi, 1)}) "
                         f"hat weder Last noch Lagerung und wird wohl nicht angebunden")

        V_ziel = volumenanteil * self.vol.sum()
        V_pass = self.vol[self.passiv].sum()
        if V_pass >= V_ziel:
            raise ValueError(f"Schutzbereiche haben schon {V_pass / self.vol.sum():.0%} "
                             f"des Volumens, Ziel {volumenanteil:.0%} nicht erreichbar")
        x = np.where(self.passiv, 1.0, (V_ziel - V_pass) / self.vol[~self.passiv].sum())
        if not self._bereit:
            self._vorbereiten()
        # Kontaktzustand aus der Nachrechnung des Originals übernehmen, falls da
        aktiv = {f: self._aktiv[f].copy() if hasattr(self, "_aktiv") else
                 np.ones(len(self.k_dof), bool) for f in self.lastfaelle}
        U, c0, beta, it_beta = None, None, 1.0, 0
        verlauf = []
        self.log(f"Optimierung: Ziel {volumenanteil:.0%} Volumen "
                 f"(Schutzbereiche {V_pass / self.vol.sum():.0%}), rmin {rmin} mm")
        it = 0
        for it in range(1, max_iter + 1):
            t0 = time.time()
            xT = Hv @ x / Hs
            xP = proj(xT, beta)
            xP[self.passiv] = 1
            E = Emin + xP ** penal * (E0 - Emin)
            U = self.loesen(E, aktiv, U)
            if c0 is None:
                c0 = {f: self.lastfaelle[f][0] @ u for f, u in U.items()}
            c, dc = 0.0, np.zeros(nel)
            for fall, u in U.items():
                ce = self.energie(u)
                g = self.lastfaelle[fall][1] / c0[fall]
                c += g * (E * ce).sum()
                dc -= g * penal * xP ** (penal - 1) * (E0 - Emin) * ce
                if it % kontakt_alle == 0:
                    aktiv[fall] = self.kontakt_update(u, aktiv[fall])
            dP = dproj(xT, beta)
            dc = HvT @ (dc * dP / Hs)
            dv = HvT @ (self.vol * dP / Hs)
            if spiegel is not None:
                dc, dv = 0.5 * (dc + dc[spiegel]), 0.5 * (dv + dv[spiegel])

            ratio = np.where(dv > 0, np.maximum(-dc, 0) / np.maximum(dv, 1e-30), 0)
            ratio /= max(np.median(ratio[~self.passiv]), 1e-30)
            move = 0.1 if beta < 8 else 0.05
            lo_x, hi_x = np.maximum(0, x - move), np.minimum(1, x + move)
            l1, l2 = 1e-12, 1e12
            while l2 / l1 > 1 + 1e-5:
                lm = np.sqrt(l1 * l2)
                xn = np.clip(x * np.sqrt(ratio / lm), lo_x, hi_x)
                xn[self.passiv] = 1
                if (physisch(xn, beta) * self.vol).sum() > V_ziel:
                    l1 = lm
                else:
                    l2 = lm
            if spiegel is not None:
                xn = 0.5 * (xn + xn[spiegel])
            aenderung = np.abs(xn - x).max()
            x = xn
            xP = physisch(x, beta)
            grau = (4 * xP * (1 - xP) * self.vol).sum() / self.vol.sum()
            verlauf.append(dict(it=it, beta=beta, c=c, vol=(xP * self.vol).sum() / self.vol.sum(),
                                aenderung=aenderung, grau=grau))
            self.log(f"  {it:3d}  beta {beta:4.0f}  Ziel {c:8.4f}  Vol {verlauf[-1]['vol']:.3f}  "
                     f"Änderung {aenderung:.3f}  grau {grau:.3f}  {time.time() - t0:.1f} s")
            it_beta += 1
            if (aenderung < 0.01 and it_beta > 5) or it_beta >= stufe:
                if beta >= beta_max:
                    break
                beta, it_beta = min(2 * beta, beta_max), 0
        self.log(f"Optimierung fertig: {it} Schritte, {(time.time() - t_start) / 60:.1f} min")
        return dict(xP=physisch(x, beta), verlauf=verlauf)

    # --- Geometrie --------------------------------------------------------------
    def ergebnis_mesh(self, xP, pitch=0.35, schwelle=0.5, rundung=1.0, glatt=20, insel_anteil=0.02):
        """Dichtefeld -> glatte Fläche, geschnitten mit dem Original, vereinigt
        mit den Schutzbereichen. Kleine abgetrennte Inseln fallen weg.
        rundung [mm]: Weichzeichnen des Dichtefelds vor dem Flächenbau; rundet
        Fensterecken und Strebenansätze zu Hohlkehlen (etwa 1-2 x rundung) und
        nimmt die Rastertreppen weg. Streben unter etwa 2 x rundung werden
        dabei dünner, daher danach immer nachrechnen."""
        feld = np.zeros(self.innen.shape)
        feld[tuple(self.ijk.T)] = xP
        idx = ndi.distance_transform_edt(~self.innen, return_distances=False, return_indices=True)
        feld = np.pad(feld[tuple(idx)], 1, mode="edge")      # außen: nächster Innenwert
        mitte = [np.r_[2 * a[0] - m[0], m, 2 * a[-1] - m[-1]]
                 for a, m in ((a, 0.5 * (a[1:] + a[:-1])) for a in self.achsen)]
        ip = RegularGridInterpolator(mitte, feld)
        lo, hi = self.mesh.bounds
        g = [np.linspace(lo[a], hi[a], int(np.ceil((hi[a] - lo[a]) / pitch)) + 1) for a in range(3)]
        Y, Z = np.meshgrid(g[1], g[2], indexing="ij")
        F = np.empty((len(g[0]),) + Y.shape, np.float32)
        for i, xv in enumerate(g[0]):
            F[i] = ip(np.column_stack([np.full(Y.size, xv), Y.ravel(), Z.ravel()])).reshape(Y.shape)
        schritt = np.array([gi[1] - gi[0] for gi in g])
        if rundung > 0:
            F = ndi.gaussian_filter(F, rundung / schritt, mode="nearest")
        F = np.pad(F, 1, constant_values=0)
        v, f, _, _ = marching_cubes(F, schwelle, spacing=tuple(schritt))
        # ohne process: das Aufräumen entfernt entartete Dreiecke und reißt Löcher
        dichte = trimesh.Trimesh(v + lo - schritt, f, process=False)
        trimesh.smoothing.filter_taubin(dichte, iterations=glatt)
        if dichte.volume < 0:
            dichte.invert()
        teil = trimesh.boolean.intersection([self.mesh, dichte], engine="manifold")
        if self.schutz:
            schutz = trimesh.boolean.union(self.schutz, engine="manifold")
            schutz = trimesh.boolean.intersection([self.mesh, schutz], engine="manifold")
            teil = trimesh.boolean.union([teil, schutz], engine="manifold")
        stuecke = teil.split(only_watertight=False)
        groesste = max(s.volume for s in stuecke)
        weg = [s.volume for s in stuecke if s.volume <= insel_anteil * groesste]
        if weg:
            self.log(f"  {len(weg)} Inseln entfernt ({sum(weg):.1f} mm³)")
        return vereinfachen(trimesh.util.concatenate(
            [s for s in stuecke if s.volume > insel_anteil * groesste]))

    def aussenflaechen(self):
        """Sichtbare Elementflächen als Vierecke (n, 4, 3), dazu Element-Index
        und Normale, für Spannungsbilder."""
        innen = np.pad(self.innen, 1)
        ecken = self.knoten[self.enode]
        seiten = [((-1, 0, 0), [0, 3, 7, 4]), ((1, 0, 0), [1, 2, 6, 5]),
                  ((0, -1, 0), [0, 1, 5, 4]), ((0, 1, 0), [3, 2, 6, 7]),
                  ((0, 0, -1), [0, 1, 2, 3]), ((0, 0, 1), [4, 5, 6, 7])]
        quads, el, normalen = [], [], []
        for n, idx in seiten:
            nb = self.ijk + 1 + np.array(n)
            offen = np.nonzero(~innen[tuple(nb.T)])[0]
            quads.append(ecken[offen][:, idx])
            el.append(offen)
            normalen.append(np.tile(n, (len(offen), 1)))
        return np.concatenate(quads), np.concatenate(el), np.concatenate(normalen).astype(float)

    # --- Nachrechnen ----------------------------------------------------------
    def pruefen(self, kontakt_iter=10, tol=1e-7):
        """Voll massives Teil: Verschiebungen und Spannungen je Lastfall, mit
        Wandkontakt (iteriert, bis keine Feder mehr zieht oder eindringt)."""
        if not self._bereit:
            self._vorbereiten()
        E = np.full(len(self.vol), self.ws["E"])
        aktiv = {f: np.ones(len(self.k_dof), bool) for f in self.lastfaelle}
        U = None
        t0 = time.time()
        for k in range(1, kontakt_iter + 1):
            U = self.loesen(E, aktiv, U, tol)
            neu = {f: self.kontakt_update(U[f], aktiv[f]) for f in U}
            wechsel = sum(int((neu[f] != aktiv[f]).sum()) for f in U)
            if wechsel == 0:
                break
            aktiv = neu
        self._aktiv = aktiv
        self.log(f"Nachrechnung: {k} Kontakt-Durchgänge, {time.time() - t0:.0f} s")
        aus = {}
        frei_e = ~self.passiv if self.passiv.any() else np.ones(len(self.vol), bool)
        for fall, u in U.items():
            vm, s1 = self.spannungen(u)
            f = self.lastfaelle[fall][0]
            aus[fall] = dict(
                u=u, vm=vm, s1=s1,
                weg_max=np.linalg.norm(u.reshape(-1, 3), axis=1).max(),
                # Weg am Lastangriff in Lastrichtung (arbeitsgleich gemittelt)
                weg_last=(f @ u) / np.linalg.norm(f.reshape(-1, 3).sum(0)),
                vm_max=vm[frei_e].max(), vm_p99=np.percentile(vm[frei_e], 99),
                s1_max=s1[frei_e].max(), s1_p99=np.percentile(s1[frei_e], 99),
                s1_max_alle=s1.max(),
                kontakt_aktiv=int(aktiv[fall].sum()), kontakt_gesamt=len(aktiv[fall]))
        return aus


# ----------------------------------------------------------------------------
# Selbsttest
# ----------------------------------------------------------------------------
def selbsttest(parallel=False):
    """Kragbalken 100 x 10 x 3 mm, eingespannt bei x = 0, 10 N am Ende nach -z.
    Handrechnung: w = F L³ / (3 E I), I = b t³ / 12."""
    L, b, t, F = 100.0, 10.0, 3.0, 10.0
    ws = dict(ABS_LIKE_3, nu=0.0)
    m = Modell(quader([0, 0, 0], [L, b, t]), 1.0, ws, parallel=parallel)
    m.fest(m.knoten_wo(lambda X, Y, Z: X < 1e-9))
    m.last("Spitze", m.knoten_wo(lambda X, Y, Z: X > L - 1e-9), (0, 0, -F))
    m.last("Spitze halb", m.knoten_wo(lambda X, Y, Z: X > L - 1e-9), (0, 0, -F / 2))
    alle = m.pruefen()
    m.beenden()
    r = alle["Spitze"]
    print(f"Lastfall halbe Kraft: {alle['Spitze halb']['weg_last'] / r['weg_last']:.4f} (soll 0,5)")
    I = b * t ** 3 / 12
    soll = F * L ** 3 / (3 * ws["E"] * I)
    ist = -r["u"][2::3][m.knoten_wo(lambda X, Y, Z: X > L - 1e-9)].mean()
    print(f"Durchbiegung: FEM {ist:.2f} mm, Handrechnung {soll:.2f} mm "
          f"({100 * (ist / soll - 1):+.1f} %)")
    # Biegespannung an der Elementmitte nahe Einspannung: x = 0,5, z = ±1
    soll_s = F * (L - 0.5) * 1.0 / I
    print(f"Spannung Elementmitte x=0,5 z=±1: FEM {r['vm_max']:.1f} MPa, Handrechnung {soll_s:.1f} MPa")


if __name__ == "__main__":
    selbsttest()
    print("parallel:")
    selbsttest(parallel=True)
