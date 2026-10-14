"""Spherical harmonics (MRtrix3 convention) and FOD glyph mesh generation.

MRtrix3 stores even-order real SH coefficients, index(l, m) = l(l+1)/2 + m,
with an orthonormal basis:
    m < 0 : sqrt(2) * N_l|m| P_l|m|(cos theta) sin(|m| phi)
    m = 0 :           N_l0   P_l0(cos theta)
    m > 0 : sqrt(2) * N_lm   P_lm(cos theta)  cos(m phi)
where P includes the Condon-Shortley phase (-1)^m, and directions are expressed
in scanner (RAS) space.
"""

import numpy as np


def n_coefs(lmax):
    return (lmax + 1) * (lmax + 2) // 2


def sh_index(l, m):
    return l * (l + 1) // 2 + m


def _legendre_normalized(lmax, m, x):
    """Orthonormalised associated Legendre N_lm P_lm(x) for l = 0..lmax (no CS phase)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((lmax + 1,) + x.shape)
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    pmm = np.full(x.shape, 0.5 / np.sqrt(np.pi))
    for k in range(1, m + 1):
        pmm = pmm * np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * s
    if m > lmax:
        return out
    out[m] = pmm
    if m + 1 <= lmax:
        out[m + 1] = x * np.sqrt(2.0 * m + 3.0) * pmm
    for l in range(m + 2, lmax + 1):
        a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
        a_prev = np.sqrt((4.0 * (l - 1) ** 2 - 1.0) / ((l - 1) ** 2 - m * m))
        out[l] = a * (x * out[l - 1] - out[l - 2] / a_prev)
    return out


def sh_basis(dirs, lmax, condon_shortley=True):
    """Matrix B (n_dirs, n_coefs) so that amplitudes = B @ coefs."""
    dirs = np.asarray(dirs, dtype=np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    cos_el = dirs[:, 2]
    az = np.arctan2(dirs[:, 1], dirs[:, 0])
    B = np.zeros((dirs.shape[0], n_coefs(lmax)))
    for m in range(0, lmax + 1):
        P = _legendre_normalized(lmax, m, cos_el)
        if condon_shortley and (m & 1):
            P = -P
        for l in range(m + (m & 1), lmax + 1, 2):
            if m == 0:
                B[:, sh_index(l, 0)] = P[l]
            else:
                B[:, sh_index(l, m)] = np.sqrt(2.0) * P[l] * np.cos(m * az)
                B[:, sh_index(l, -m)] = np.sqrt(2.0) * P[l] * np.sin(m * az)
    return B


# ---------------------------------------------------------------------------
# Sphere tessellation
# ---------------------------------------------------------------------------

def icosphere(level=2):
    """Unit icosphere. level 0: 12 vertices, 1: 42, 2: 162, 3: 642, 4: 2562."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    for _ in range(level):
        cache = {}
        new_faces = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                v = verts[a] + verts[b]
                verts.append(v / np.linalg.norm(v))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts), np.array(faces, dtype=np.int64)


class GlyphTemplate(object):
    """Precomputed sphere + SH basis for fast glyph generation."""

    def __init__(self, lmax, level=2):
        self.lmax = lmax
        self.level = level
        self.dirs, self.faces = icosphere(level)
        self.dirs32 = self.dirs.astype(np.float32)
        self.B = sh_basis(self.dirs, lmax).astype(np.float32)  # (V, C)
        self.dir_rgb = (np.abs(self.dirs) * 255.0).clip(0, 255).astype(np.uint8)
        self._tangent_stencil()
        self._tri_cache = np.zeros((0, 3), dtype=np.int64)

    def _tangent_stencil(self):
        """For each vertex, two pairs of ring neighbours spanning its tangent plane.

        Normal ~ cross(p[a] - p[b], p[c] - p[d]): four gathers instead of a
        per-face computation, which keeps glyph generation interactive.
        """
        V = self.dirs.shape[0]
        ring = [set() for _ in range(V)]
        for a, b, c in self.faces:
            ring[a].update((b, c))
            ring[b].update((a, c))
            ring[c].update((a, b))
        st = np.zeros((V, 4), dtype=np.int64)
        for v in range(V):
            nb = np.array(sorted(ring[v]))
            d = self.dirs[v]
            # orthonormal tangent frame
            t1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(d, t1)
            proj1 = self.dirs[nb] @ t1
            proj2 = self.dirs[nb] @ t2
            a, b = nb[np.argmax(proj1)], nb[np.argmin(proj1)]
            c, e = nb[np.argmax(proj2)], nb[np.argmin(proj2)]
            st[v] = (a, b, c, e)
        # orient outwards
        n = np.cross(self.dirs[st[:, 0]] - self.dirs[st[:, 1]], self.dirs[st[:, 2]] - self.dirs[st[:, 3]])
        flip = np.einsum("ij,ij->i", n, self.dirs) < 0
        st[flip, 0], st[flip, 1] = st[flip, 1].copy(), st[flip, 0].copy()
        self.stencil = st

    def triangles(self, n_glyphs):
        """Triangle indices for n glyphs (cached; a prefix of a larger array is valid)."""
        F = self.faces.shape[0]
        if self._tri_cache.shape[0] < n_glyphs * F:
            n = max(n_glyphs, 2 * self._tri_cache.shape[0] // max(F, 1))
            V = self.dirs.shape[0]
            self._tri_cache = (self.faces[None] + (np.arange(n, dtype=np.int64) * V)[:, None, None]).reshape(-1, 3)
        return self._tri_cache[: n_glyphs * F]

    @property
    def n_verts(self):
        return self.dirs.shape[0]


def amplitudes(coefs, template):
    """(G, C) coefficients -> (G, V) amplitudes on the template sphere."""
    return coefs @ template.B.T


def build_glyph_mesh(centers, amps, template, scale, view_normal=None, negative="clip"):
    """Return arrays for a VTK triangle mesh of FOD glyphs.

    centers  (G,3) RAS mm; amps (G,V) already-normalised radii (unitless)
    scale    mm per unit amplitude
    Returns dict(points (G*V,3) f4, normals (G*V,3) f4, triangles (G*F,3) i8,
                 rgb (G*V,3) u1 direction colour, shade (G*V,) f4 or None)
    """
    G = centers.shape[0]
    V = template.n_verts
    r = np.abs(amps) if negative == "abs" else np.maximum(amps, 0.0)
    r = r.astype(np.float32, copy=False) * np.float32(scale)
    offs = r[..., None] * template.dirs32[None]
    pts = offs + centers.astype(np.float32, copy=False)[:, None, :]

    st = template.stencil
    vn = np.cross(offs[:, st[:, 0]] - offs[:, st[:, 1]], offs[:, st[:, 2]] - offs[:, st[:, 3]])
    norm = np.sqrt(np.einsum("gvi,gvi->gv", vn, vn))
    bad = norm < 1e-12
    norm[bad] = 1.0
    vn /= norm[..., None]
    if bad.any():
        vn[bad] = np.broadcast_to(template.dirs32[None], vn.shape)[bad]

    out = {
        "points": pts.reshape(-1, 3),
        "normals": vn.reshape(-1, 3),
        "triangles": template.triangles(G),
        "rgb": np.broadcast_to(template.dir_rgb[None], (G, V, 3)).reshape(-1, 3),
        "shade": None,
    }
    if view_normal is not None:
        n = np.asarray(view_normal, dtype=np.float32)
        n = n / (np.linalg.norm(n) + 1e-12)
        out["shade"] = np.float32(0.25) + np.float32(0.75) * np.abs(out["normals"] @ n)
    return out
