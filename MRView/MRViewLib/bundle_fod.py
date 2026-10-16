"""FODs seen through a fibre bundle (pure numpy).

Given the streamlines of a bundle and the voxel grid of an FOD image, this
computes which FOD voxels the bundle goes through (how many streamlines visit
each voxel) and, for each visited voxel, which fibre orientations the bundle
uses there. The FOD layer then shows only the glyphs of the bundle's voxels,
optionally keeping only the lobes aligned with the bundle's fibres.
"""

import numpy as np

from . import sh

# Fibre orientations are quantised on this icosphere (level 2: 162 directions,
# about 9 degrees apart), folded to one hemisphere because FODs are antipodal.
_QUANT_LEVEL = 2
_quant_dirs = None


def quant_dirs():
    global _quant_dirs
    if _quant_dirs is None:
        d, _ = sh.icosphere(_QUANT_LEVEL)
        _quant_dirs = d.astype(np.float32)
    return _quant_dirs


def densify(points, offsets, max_step):
    """Resample streamlines so that consecutive samples are at most max_step apart.

    Returns (samples (S,3) f4, streamline id (S,) i8, unit tangents (S,3) f4).
    Original vertices are kept; segments longer than max_step get extra points.
    """
    points = np.asarray(points, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    n_pts = points.shape[0]
    n_sl = offsets.size - 1
    if n_pts == 0 or n_sl <= 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    counts = np.diff(offsets)
    sl_of_point = np.repeat(np.arange(n_sl, dtype=np.int64), counts)
    is_last = np.zeros(n_pts, dtype=bool)
    is_last[offsets[1:][counts > 0] - 1] = True

    seg = np.zeros_like(points)
    seg[:-1] = points[1:] - points[:-1]
    seg[is_last] = 0.0
    length = np.sqrt(np.einsum("ij,ij->i", seg, seg))
    tan = np.zeros_like(points)
    ok = length > 0
    tan[ok] = seg[ok] / length[ok, None]
    # the last point of a streamline takes the tangent of its last segment
    prev = np.flatnonzero(is_last)
    prev = prev[(prev > 0) & ~is_last[np.maximum(prev - 1, 0)]]
    tan[prev] = tan[prev - 1]

    k = np.ones(n_pts, dtype=np.int64)
    k[ok] = np.maximum(1, np.ceil(length[ok] / float(max_step))).astype(np.int64)
    k[is_last] = 1
    total = int(k.sum())
    src = np.repeat(np.arange(n_pts, dtype=np.int64), k)
    first = np.cumsum(k) - k
    j = np.arange(total, dtype=np.int64) - np.repeat(first, k)
    frac = (j / np.repeat(k, k)).astype(np.float32)
    samples = points[src] + frac[:, None] * seg[src]
    return samples, sl_of_point[src], tan[src]


def nearest_dir(tangents, dirs, chunk=200000):
    """Index of the closest (antipodally symmetric) direction for each tangent."""
    out = np.empty(tangents.shape[0], dtype=np.int64)
    for s in range(0, tangents.shape[0], chunk):
        out[s:s + chunk] = np.argmax(np.abs(tangents[s:s + chunk] @ dirs.T), axis=1)
    return out


class BundleFODMap(object):
    """Voxel occupancy and fibre orientations of a bundle on an FOD grid.

    index_map: int array over the image grid, index of the stored FOD voxel or -1.
    ras_to_ijk: 4x4 world -> voxel indices.
    """

    def __init__(self, points, offsets, index_map, ras_to_ijk, spacing, n_voxels, max_samples=1500000, seed=0):
        offsets = np.asarray(offsets, dtype=np.int64)
        self.n_streamlines = offsets.size - 1
        self.subsampled = False
        if points.shape[0] > max_samples and self.n_streamlines > 1:
            # keep a random subset of whole streamlines (bounded memory and time)
            rng = np.random.RandomState(seed)
            keep = np.sort(rng.choice(self.n_streamlines,
                                      max(1, int(self.n_streamlines * max_samples / float(points.shape[0]))),
                                      replace=False))
            counts = np.diff(offsets)[keep]
            starts = offsets[:-1][keep]
            ends = np.cumsum(counts)
            idx = np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(starts - (ends - counts), counts)
            points = points[idx]
            offsets = np.concatenate(([0], ends))
            self.subsampled = True
        step = 0.5 * float(np.min(spacing))
        samples, sl, tan = densify(points, offsets, step)
        A = np.asarray(ras_to_ijk, dtype=np.float64)
        ijk = np.rint(samples @ A[:3, :3].T.astype(np.float32) + A[:3, 3].astype(np.float32)).astype(np.int64)
        shape = np.array(index_map.shape[:3])
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        vox = np.full(samples.shape[0], -1, dtype=np.int64)
        if inside.any():
            sub = ijk[inside]
            vox[inside] = index_map[sub[:, 0], sub[:, 1], sub[:, 2]]
        keep = vox >= 0
        vox, sl, tan = vox[keep], sl[keep], tan[keep]

        # streamline count per voxel: samples are ordered along each streamline, so a
        # streamline entering a voxel is a change of (streamline, voxel) between samples
        # (a streamline coming back into a voxel later is counted again, which is rare)
        if vox.size:
            new = np.ones(vox.size, dtype=bool)
            new[1:] = (vox[1:] != vox[:-1]) | (sl[1:] != sl[:-1])
            self.count = np.bincount(vox[new], minlength=n_voxels).astype(np.int32)
            # orientations: one tangent per sample except where a streamline lingers in a voxel
            # (keep the first and last sample of each stay, enough to follow a curve)
            last = np.ones(vox.size, dtype=bool)
            last[:-1] = new[1:]
            keep = new | last
            self._vox, self._tan, self._sl = vox[keep], tan[keep], sl[keep]
        else:
            self.count = np.zeros(n_voxels, dtype=np.int32)
            self._vox, self._tan, self._sl = vox, tan, sl
        self._orient = None  # (sorted voxel ids, direction ids, voxel starts)

    def mask(self, min_streamlines=1):
        return self.count >= max(1, int(min_streamlines))

    @property
    def n_voxels_hit(self):
        return int(np.count_nonzero(self.count))

    def _orientations(self):
        if self._orient is None:
            q = nearest_dir(self._tan, quant_dirs()) if self._vox.size else np.zeros(0, np.int64)
            nq = quant_dirs().shape[0]
            code = self._vox * nq + q
            if code.size:
                code = code[np.append(True, code[1:] != code[:-1])]
            pairs = np.unique(code)
            vox = pairs // nq
            uv, starts = np.unique(vox, return_index=True)
            self._orient = (uv, pairs % nq, starts)
            self._tan = self._sl = None  # no longer needed
        return self._orient

    def lobe_weights(self, voxel_ids, template_dirs, angle_deg, soft_deg=10.0):
        """(len(voxel_ids), V) weights in [0, 1]: 1 where the template direction is
        within angle_deg of a fibre orientation used by the bundle in that voxel,
        fading to 0 over soft_deg more degrees."""
        voxel_ids = np.asarray(voxel_ids, dtype=np.int64)
        V = template_dirs.shape[0]
        w = np.zeros((voxel_ids.size, V), dtype=np.float32)
        uv, qdir, starts = self._orientations()
        if voxel_ids.size == 0 or uv.size == 0:
            return w
        pos = np.searchsorted(uv, voxel_ids)
        pos_c = np.minimum(pos, uv.size - 1)
        found = uv[pos_c] == voxel_ids
        if not found.any():
            return w
        ends = np.append(starts[1:], qdir.size)
        rows = np.flatnonzero(found)
        a, b = starts[pos_c[rows]], ends[pos_c[rows]]
        n = b - a
        flat = np.repeat(a - np.cumsum(n) + n, n) + np.arange(int(n.sum()), dtype=np.int64)
        cos = np.abs(quant_dirs()[qdir[flat]] @ np.asarray(template_dirs, dtype=np.float32).T)
        row_starts = np.cumsum(n) - n
        best = np.maximum.reduceat(cos, row_starts, axis=0)
        c_in = np.cos(np.radians(angle_deg))
        c_out = np.cos(np.radians(min(angle_deg + soft_deg, 90.0)))
        t = np.clip((best - c_out) / max(c_in - c_out, 1e-6), 0.0, 1.0)
        w[rows] = t * t * (3.0 - 2.0 * t)
        return w
