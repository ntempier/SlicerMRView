"""Tests for the Slicer-independent core (run: python -m pytest Testing/ or python Testing/test_core.py).

Optional: set MRVIEW_TEST_DATA to the Test_datas folder to also test real files.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from MRViewLib import mrtrix_io as io  # noqa: E402
from MRViewLib import sh, tracts  # noqa: E402


def _random_tractogram(n=50, seed=1):
    rng = np.random.RandomState(seed)
    lengths = rng.randint(2, 30, size=n)
    pts = np.cumsum(rng.randn(lengths.sum(), 3).astype(np.float32), axis=0)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    return pts, offsets


def test_tck_roundtrip():
    pts, offsets = _random_tractogram()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.tck")
        io.write_tck(p, pts, offsets, {"step_size": "0.5"})
        t = io.read_tck(p)
        assert np.array_equal(t["offsets"], offsets)
        assert np.allclose(t["points"], pts)
        assert t["header"]["step_size"] == "0.5"


def test_mif_roundtrip_and_layout():
    rng = np.random.RandomState(0)
    data = rng.rand(5, 6, 7, 15).astype(np.float32)
    M = np.array([[0, -2, 0, 10], [1.5, 0, 0, -3], [0, 0, 2.5, 7], [0, 0, 0, 1]], dtype=float)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.mif")
        io.write_mif(p, data, M)
        im = io.read_mif(p)
        assert np.allclose(im.data, data)
        assert np.allclose(im.ijk_to_ras, M)
        # Hand-written header with flipped / permuted layout
        p2 = os.path.join(d, "b.mif")
        hdr = ("mrtrix image\ndim: 5,6,7\nvox: 1,1,1\nlayout: -1,+0,+2\ndatatype: Float32LE\n"
               "transform: 1,0,0,0\ntransform: 0,1,0,0\ntransform: 0,0,1,0\n")
        vol = data[..., 0]
        # storage: axis1 fastest, then axis0, then axis2; axis0 reversed
        stored = np.flip(vol, axis=0).transpose(2, 0, 1).ravel(order="C")
        off = 256
        head = (hdr + "file: . %d\nEND\n" % off).encode().ljust(off, b"\0")
        with open(p2, "wb") as f:
            f.write(head)
            stored.astype("<f4").tofile(f)
        im2 = io.read_mif(p2)
        assert np.allclose(im2.data, vol)


def test_sh_orthonormal_and_mrtrix_convention():
    d, _ = sh.icosphere(5)
    B = sh.sh_basis(d, 8)
    G = B.T @ B * 4 * np.pi / d.shape[0]
    assert np.abs(G - np.eye(45)).max() < 0.06
    # Y00 = 1 / sqrt(4 pi)
    assert np.allclose(B[:, 0], 0.5 / np.sqrt(np.pi))
    # An FOD made from a delta along x peaks along +-x
    target = np.array([[1.0, 0.0, 0.0]])
    coefs = sh.sh_basis(target, 8)[0]
    amp = B @ coefs
    assert abs(d[np.argmax(amp)][0]) > 0.98


def test_glyph_mesh_shapes():
    tpl = sh.GlyphTemplate(8, level=2)
    coefs = np.zeros((4, 45), dtype=np.float32)
    coefs[:, 0] = 1.0
    amps = sh.amplitudes(coefs, tpl)
    mesh = sh.build_glyph_mesh(np.zeros((4, 3), np.float32), amps, tpl, 2.0, view_normal=(0, 0, 1))
    assert mesh["points"].shape == (4 * 162, 3)
    assert mesh["triangles"].shape == (4 * 320, 3)
    r = np.linalg.norm(mesh["points"], axis=1)
    assert np.allclose(r, 2.0 * 0.5 / np.sqrt(np.pi), atol=1e-5)
    # normals of a sphere point outwards
    assert np.all(np.einsum("ij,ij->i", mesh["normals"], mesh["points"]) > 0)


def test_tract_ops_and_roi():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],      # 0: along x through origin area
                    [0, 5, 0], [0, 6, 0],                  # 1: far in y
                    [10, 0, 0], [11, 0, 0], [12, 0, 0]], np.float32)  # 2: far in x
    offsets = np.array([0, 3, 5, 8])
    t = tracts.Tractogram(pts, offsets)
    assert np.allclose(t.lengths_mm(), [2, 1, 2])
    rgb = t.direction_rgb()
    assert rgb[0, 0] == 255 and rgb[3, 1] == 255
    sub = t.subset([2, 0])
    assert sub.n_streamlines == 2 and np.allclose(sub.points[0], [10, 0, 0])

    eye = np.eye(4)
    box = lambda p: tracts.points_in_box(p, eye, [1.5, 1.5, 1.5])
    hits = tracts.streamline_hits(t, box)
    assert hits.tolist() == [True, False, False]
    sphere_world_to_obj = np.eye(4)
    sphere_world_to_obj[:3, 3] = [-11, 0, 0]
    sph = lambda p: tracts.points_in_ellipsoid(p, sphere_world_to_obj, [0.5, 0.5, 0.5])
    assert tracts.streamline_hits(t, sph).tolist() == [False, False, True]
    # endpoint test: centre point of streamline 2 is inside but it is not an endpoint
    assert tracts.streamline_hits(t, sph, endpoints_only=True).tolist() == [False, False, False]

    mask = np.zeros((20, 20, 20), bool)
    mask[0, 5, 0] = True
    assert tracts.streamline_hits(t, lambda p: tracts.points_in_mask(p, eye, mask)).tolist() == [False, True, False]

    sel = tracts.combine_selection(3, [hits], [tracts.streamline_hits(t, sph)])
    assert sel.tolist() == [True, False, False]
    segs = t.slab_segments([0, 0, 0], [0, 1, 0], 0.5)
    assert segs.tolist() == [0, 1, 5, 6]


def test_rotated_box():
    # 45 degree rotated box around z: point (1.2, 1.2, 0) lies inside a 1x1 (half 0.5 x 2) box along the diagonal
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    obj_to_world = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    w2o = np.linalg.inv(obj_to_world)
    p = np.array([[1.2, 1.2, 0], [1.2, -1.2, 0]], np.float32)
    assert tracts.points_in_box(p, w2o, [2.0, 0.5, 0.5]).tolist() == [True, False]


def test_real_data():
    root = os.environ.get("MRVIEW_TEST_DATA")
    if not root or not os.path.isdir(root):
        return
    tck = [f for f in os.listdir(root) if f.endswith(".tck")]
    if tck:
        t = io.read_tck(os.path.join(root, tck[0]))
        n = int(t["header"].get("count", t["offsets"].size - 1))
        assert t["offsets"].size - 1 == n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
