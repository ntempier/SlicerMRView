"""Vectorised streamline operations (pure numpy).

A tractogram is stored compactly: points (N,3) float32 without delimiters and
offsets (M+1,) int64 so that streamline i is points[offsets[i]:offsets[i+1]].
"""

import numpy as np


def concat_ranges(starts, lengths):
    """Concatenate [s, s+l) ranges without a Python loop."""
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.cumsum(lengths)
    shift = np.repeat(starts - (ends - lengths), lengths)
    return np.arange(total, dtype=np.int64) + shift


class Tractogram(object):
    def __init__(self, points, offsets, header=None, name=""):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.header = header or {}
        self.name = name
        self._lengths_mm = None
        self._is_first = None
        self._is_last = None

    # -- basic properties --------------------------------------------------
    @property
    def n_streamlines(self):
        return self.offsets.size - 1

    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def starts(self):
        return self.offsets[:-1]

    @property
    def counts(self):
        return np.diff(self.offsets)

    @property
    def is_first(self):
        if self._is_first is None:
            m = np.zeros(self.n_points, dtype=bool)
            m[self.offsets[:-1]] = True
            self._is_first = m
        return self._is_first

    @property
    def is_last(self):
        if self._is_last is None:
            m = np.zeros(self.n_points, dtype=bool)
            m[self.offsets[1:] - 1] = True
            self._is_last = m
        return self._is_last

    def lengths_mm(self):
        if self._lengths_mm is None:
            if self.n_points < 2:
                self._lengths_mm = np.zeros(self.n_streamlines, dtype=np.float32)
            else:
                seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
                seg[self.is_last[:-1]] = 0.0  # no segment across streamlines
                seg = np.append(seg, 0.0)
                self._lengths_mm = np.add.reduceat(seg, self.starts).astype(np.float32)
        return self._lengths_mm

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        counts = self.counts[indices]
        idx = concat_ranges(self.starts[indices], counts)
        offsets = np.zeros(indices.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return Tractogram(self.points[idx], offsets, dict(self.header), self.name)

    def point_indices(self, streamline_mask_or_indices):
        sel = np.asarray(streamline_mask_or_indices)
        if sel.dtype == bool:
            sel = np.flatnonzero(sel)
        return concat_ranges(self.starts[sel], self.counts[sel])

    def per_point(self, per_streamline):
        return np.repeat(per_streamline, self.counts, axis=0)

    # -- colours -------------------------------------------------------------
    def tangents(self):
        p = self.points
        t = np.zeros_like(p)
        if p.shape[0] < 2:
            return t
        fwd = np.zeros_like(p)
        fwd[:-1] = p[1:] - p[:-1]
        fwd[self.is_last] = 0.0
        bwd = np.zeros_like(p)
        bwd[1:] = p[1:] - p[:-1]
        bwd[self.is_first] = 0.0
        t = fwd + bwd
        n = np.linalg.norm(t, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return t / n

    def direction_rgb(self):
        return (np.abs(self.tangents()) * 255.0).clip(0, 255).astype(np.uint8)

    def endpoint_rgb(self):
        d = self.points[self.offsets[1:] - 1] - self.points[self.offsets[:-1]]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n == 0] = 1.0
        rgb = (np.abs(d / n) * 255.0).clip(0, 255).astype(np.uint8)
        return self.per_point(rgb)

    def random_rgb(self, seed=0):
        rng = np.random.RandomState(seed)
        h = rng.rand(self.n_streamlines)
        s = 0.55 + 0.45 * rng.rand(self.n_streamlines)
        v = np.full(self.n_streamlines, 0.95)
        return self.per_point((hsv_to_rgb(h, s, v) * 255).astype(np.uint8))

    # -- slab cropping (2D views) -------------------------------------------
    def slab_segments(self, origin, normal, half_thickness, point_mask=None):
        """Indices i of segments (i, i+1) whose both ends lie within the slab."""
        if self.n_points < 2:
            return np.zeros(0, dtype=np.int64)
        n = np.asarray(normal, dtype=np.float32)
        n = n / (np.linalg.norm(n) + 1e-12)
        d = self.points @ n - np.float32(np.dot(np.asarray(origin, dtype=np.float32), n))
        inside = np.abs(d) <= half_thickness
        if point_mask is not None:
            inside &= point_mask
        valid = inside[:-1] & inside[1:] & ~self.is_last[:-1]
        return np.flatnonzero(valid)


def hsv_to_rgb(h, s, v):
    h = np.asarray(h) * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# ROI tests
# ---------------------------------------------------------------------------

def points_in_box(points, world_to_object, half_size):
    """world_to_object: 4x4; ROI is |x_obj| <= half_size (centred at object origin)."""
    M = np.asarray(world_to_object, dtype=np.float32)
    h = np.asarray(half_size, dtype=np.float32)
    # cheap axis-aligned prefilter using the box's world bounding box
    obj_to_world = np.linalg.inv(M.astype(np.float64))
    corners = np.array([[sx, sy, sz, 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.append(h, 1.0)
    wc = corners @ obj_to_world.T
    lo, hi = wc[:, :3].min(0), wc[:, :3].max(0)
    cand = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
    out = np.zeros(points.shape[0], dtype=bool)
    if cand.size:
        local = points[cand] @ M[:3, :3].T + M[:3, 3]
        out[cand] = np.all(np.abs(local) <= h, axis=1)
    return out


def points_in_ellipsoid(points, world_to_object, radii):
    M = np.asarray(world_to_object, dtype=np.float32)
    r = np.maximum(np.asarray(radii, dtype=np.float32), 1e-6)
    obj_to_world = np.linalg.inv(M.astype(np.float64))
    corners = np.array([[sx, sy, sz, 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.append(r, 1.0)
    wc = corners @ obj_to_world.T
    lo, hi = wc[:, :3].min(0), wc[:, :3].max(0)
    cand = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
    out = np.zeros(points.shape[0], dtype=bool)
    if cand.size:
        local = (points[cand] @ M[:3, :3].T + M[:3, 3]) / r
        out[cand] = np.einsum("ij,ij->i", local, local) <= 1.0
    return out


def points_in_mask(points, ras_to_ijk, mask_ijk):
    """mask_ijk: bool array indexed [i, j, k]."""
    A = np.asarray(ras_to_ijk, dtype=np.float32)
    ijk = np.rint(points @ A[:3, :3].T + A[:3, 3]).astype(np.int64)
    shape = np.array(mask_ijk.shape[:3])
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    out = np.zeros(points.shape[0], dtype=bool)
    if ok.any():
        sub = ijk[ok]
        out[ok] = mask_ijk[sub[:, 0], sub[:, 1], sub[:, 2]]
    return out


def streamline_hits(tractogram, point_test, endpoints_only=False):
    """Per-streamline bool: any point (or either endpoint) passes point_test."""
    if tractogram.n_streamlines == 0:
        return np.zeros(0, dtype=bool)
    if endpoints_only:
        first = tractogram.points[tractogram.offsets[:-1]]
        last = tractogram.points[tractogram.offsets[1:] - 1]
        return point_test(first) | point_test(last)
    inside = point_test(tractogram.points)
    return np.logical_or.reduceat(inside, tractogram.starts)


def combine_selection(n, includes, excludes, include_mode="all"):
    """includes/excludes: lists of per-streamline bool arrays."""
    sel = np.ones(n, dtype=bool)
    if includes:
        if include_mode == "any":
            acc = np.zeros(n, dtype=bool)
            for h in includes:
                acc |= h
            sel &= acc
        else:
            for h in includes:
                sel &= h
    for h in excludes:
        sel &= ~h
    return sel
