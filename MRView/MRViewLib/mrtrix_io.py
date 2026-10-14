"""Pure-numpy readers/writers for MRtrix3 and NIfTI files.

No dependency on Slicer, VTK, nibabel or scipy so it can be unit-tested
anywhere. All coordinates are returned in scanner RAS (mm), which is both the
MRtrix3 "real space" and Slicer's world coordinate system.
"""

import gzip
import os
import re
import struct

import numpy as np


# ---------------------------------------------------------------------------
# Tractograms (.tck) and track scalar files (.tsf)
# ---------------------------------------------------------------------------

_DTYPES = {
    "float32le": "<f4", "float32be": ">f4", "float64le": "<f8", "float64be": ">f8",
    "float32": "<f4", "float64": "<f8",
    "int8": "i1", "uint8": "u1", "bit": "u1",
    "int16le": "<i2", "int16be": ">i2", "uint16le": "<u2", "uint16be": ">u2",
    "int32le": "<i4", "int32be": ">i4", "uint32le": "<u4", "uint32be": ">u4",
    "int64le": "<i8", "int64be": ">i8", "uint64le": "<u8", "uint64be": ">u8",
    "int16": "<i2", "uint16": "<u2", "int32": "<i4", "uint32": "<u4",
}


def _np_dtype(name):
    key = name.strip().lower()
    if key not in _DTYPES:
        raise ValueError("Unsupported MRtrix datatype: %s" % name)
    return np.dtype(_DTYPES[key])


def _read_text_header(fileobj, magic_prefix):
    """Read a key: value MRtrix header up to the END line."""
    first = fileobj.readline().decode("latin-1").strip()
    if not first.startswith(magic_prefix):
        raise ValueError("Not a valid MRtrix file (expected '%s', got '%s')" % (magic_prefix, first))
    header = {}
    while True:
        line = fileobj.readline()
        if not line:
            raise ValueError("Unexpected end of file while reading MRtrix header")
        line = line.decode("latin-1").rstrip("\r\n")
        if line.strip() == "END":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in header:
            if isinstance(header[key], list):
                header[key].append(value)
            else:
                header[key] = [header[key], value]
        else:
            header[key] = value
    return first, header


def _header_value(header, key, default=None):
    v = header.get(key, default)
    if isinstance(v, list):
        return v[-1]
    return v


def _parse_file_offset(header):
    file_entry = _header_value(header, "file")
    if file_entry is None:
        raise ValueError("MRtrix header has no 'file' entry")
    parts = file_entry.split()
    name = parts[0]
    offset = int(parts[1]) if len(parts) > 1 else 0
    return name, offset


def _split_tracks(raw):
    """Split an (N,k) array delimited by NaN rows and terminated by an Inf row.

    Returns (data_without_delimiters, offsets) where offsets has length
    n_tracks + 1 and track i is data[offsets[i]:offsets[i+1]].
    """
    col0 = raw[:, 0]
    inf_rows = np.flatnonzero(np.isinf(col0))
    if inf_rows.size:
        raw = raw[: inf_rows[0]]
        col0 = raw[:, 0]
    nan_rows = np.flatnonzero(np.isnan(col0))
    # Tracks lie between consecutive delimiters; data may or may not end with one
    ends = nan_rows
    if raw.shape[0] and (nan_rows.size == 0 or nan_rows[-1] != raw.shape[0] - 1):
        ends = np.append(nan_rows, raw.shape[0])
    starts = np.concatenate(([0], nan_rows + 1))[: ends.size]
    lengths = ends - starts
    keep = lengths > 0
    starts, lengths = starts[keep], lengths[keep]
    valid = np.ones(raw.shape[0], dtype=bool)
    valid[nan_rows] = False
    data = raw[valid]
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return data, offsets


def read_tck(path):
    """Read an MRtrix3 .tck file.

    Returns dict(points=(N,3) float32 RAS mm, offsets=(M+1,) int64, header=dict).
    """
    with open(path, "rb") as f:
        _, header = _read_text_header(f, "mrtrix tracks")
    dtype = _np_dtype(_header_value(header, "datatype", "Float32LE"))
    name, offset = _parse_file_offset(header)
    data_path = path if name == "." else os.path.join(os.path.dirname(path), name)
    raw = np.fromfile(data_path, dtype=dtype, offset=offset if name == "." else offset)
    raw = raw[: (raw.size // 3) * 3].reshape(-1, 3)
    points, offsets = _split_tracks(raw)
    return {"points": np.ascontiguousarray(points, dtype=np.float32), "offsets": offsets, "header": header}


def read_tsf(path):
    """Read an MRtrix3 track scalar file. Returns dict(values, offsets, header)."""
    with open(path, "rb") as f:
        _, header = _read_text_header(f, "mrtrix track scalars")
    dtype = _np_dtype(_header_value(header, "datatype", "Float32LE"))
    name, offset = _parse_file_offset(header)
    data_path = path if name == "." else os.path.join(os.path.dirname(path), name)
    raw = np.fromfile(data_path, dtype=dtype, offset=offset).reshape(-1, 1)
    values, offsets = _split_tracks(raw)
    return {"values": values[:, 0].astype(np.float32), "offsets": offsets, "header": header}


def write_tck(path, points, offsets, extra_header=None):
    """Write streamlines to an MRtrix3 .tck file (Float32LE)."""
    points = np.asarray(points, dtype="<f4")
    offsets = np.asarray(offsets, dtype=np.int64)
    n_tracks = offsets.size - 1
    lines = ["mrtrix tracks"]
    for k, v in (extra_header or {}).items():
        if k in ("count", "datatype", "file", "total_count"):
            continue
        for item in (v if isinstance(v, list) else [v]):
            lines.append("%s: %s" % (k, item))
    lines.append("count: %d" % n_tracks)
    lines.append("datatype: Float32LE")
    body = "\n".join(lines) + "\n"
    # The offset must account for its own digits
    offset = len(body) + len("file: . \nEND\n") + 1
    while True:
        candidate = body + "file: . %d\nEND\n" % offset
        if len(candidate) <= offset:
            break
        offset += 1
    hdr = candidate.encode("latin-1").ljust(offset, b"\0")

    total = points.shape[0] + n_tracks + 1
    out = np.empty((total, 3), dtype="<f4")
    lengths = np.diff(offsets)
    # destination index of each point: original index + number of delimiters before it
    track_id = np.repeat(np.arange(n_tracks), lengths)
    dest = np.arange(points.shape[0]) + track_id
    out[:] = np.nan
    out[dest] = points
    out[-1] = np.inf
    with open(path, "wb") as f:
        f.write(hdr)
        out.tofile(f)


# ---------------------------------------------------------------------------
# Images: MRtrix .mif / .mif.gz / .mih and NIfTI .nii / .nii.gz
# ---------------------------------------------------------------------------

class Image(object):
    """Minimal image container.

    data      numpy array indexed [i, j, k(, volume)] (Fortran / MRtrix axis order)
    ijk_to_ras 4x4 float64 matrix mapping voxel indices to scanner RAS (mm)
    """

    def __init__(self, data, ijk_to_ras, header=None, path=None):
        self.data = data
        self.ijk_to_ras = np.asarray(ijk_to_ras, dtype=np.float64)
        self.header = header or {}
        self.path = path

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_volumes(self):
        return 1 if self.data.ndim < 4 else int(np.prod(self.data.shape[3:]))

    @property
    def spacing(self):
        return np.linalg.norm(self.ijk_to_ras[:3, :3], axis=0)

    def volume(self, index=0):
        if self.data.ndim < 4:
            return self.data
        return self.data.reshape(self.data.shape[:3] + (-1,))[..., index]


def _open_maybe_gz(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return open(path, "rb")


def read_mif(path):
    """Read MRtrix .mif / .mif.gz / .mih images."""
    with _open_maybe_gz(path) as f:
        _, header = _read_text_header(f, "mrtrix image")
        dims = [int(x) for x in _header_value(header, "dim").split(",")]
        vox = [float(x) if x.strip().lower() != "nan" else 1.0 for x in _header_value(header, "vox").split(",")]
        layout = _header_value(header, "layout").split(",")
        dtype_name = _header_value(header, "datatype")
        if dtype_name.strip().lower() == "bit":
            raise ValueError("Bit-packed .mif images are not supported")
        dtype = _np_dtype(dtype_name)
        name, offset = _parse_file_offset(header)
        n = int(np.prod(dims))
        if name == ".":
            f.seek(0)
            buf = f.read()
            raw = np.frombuffer(buf, dtype=dtype, count=n, offset=offset)
        else:
            data_path = os.path.join(os.path.dirname(path), name)
            raw = np.fromfile(data_path, dtype=dtype, count=n, offset=offset)

    # layout entries like "-0", "+1", "+2": rank of stride (0 = fastest) and direction
    signs = [(-1 if s.strip().startswith("-") else 1) for s in layout]
    ranks = [int(s.strip().lstrip("+-")) for s in layout]
    order_fast_to_slow = np.argsort(ranks)
    c_shape = [dims[a] for a in order_fast_to_slow[::-1]]
    arr = raw.reshape(c_shape)
    # arr axes are (slowest ... fastest) -> map back to canonical axes
    canonical_axes = list(order_fast_to_slow[::-1])
    perm = [canonical_axes.index(a) for a in range(len(dims))]
    arr = np.transpose(arr, perm)
    for a, s in enumerate(signs):
        if s < 0:
            arr = np.flip(arr, axis=a)

    transform = np.eye(4)
    tr = header.get("transform")
    if tr is not None:
        rows = tr if isinstance(tr, list) else [tr]
        for r, row in enumerate(rows[:3]):
            transform[r, :] = [float(x) for x in row.split(",")]
    ijk_to_ras = transform.copy()
    ijk_to_ras[:3, :3] = transform[:3, :3] * np.asarray(vox[:3])[None, :]

    scale = float(_header_value(header, "scaling", "0,1").split(",")[1]) if "scaling" in header else 1.0
    inter = float(_header_value(header, "scaling", "0,1").split(",")[0]) if "scaling" in header else 0.0
    if scale != 1.0 or inter != 0.0:
        arr = arr.astype(np.float32) * scale + inter
    return Image(arr, ijk_to_ras, header, path)


_NIFTI_DTYPES = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}


def _quatern_to_mat(b, c, d, qx, qy, qz, dx, dy, dz, qfac):
    a = 1.0 - (b * b + c * c + d * d)
    if a < 1e-7:
        a = 1.0 / np.sqrt(b * b + c * c + d * d)
        b, c, d = b * a, c * a, d * a
        a = 0.0
    else:
        a = np.sqrt(a)
    R = np.array([
        [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
        [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
        [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b],
    ])
    if qfac < 0:
        dz = -dz
    M = np.eye(4)
    M[:3, :3] = R * np.array([dx, dy, dz])[None, :]
    M[:3, 3] = [qx, qy, qz]
    return M


def read_nifti(path):
    """Read a NIfTI-1 (.nii / .nii.gz) image without nibabel."""
    with _open_maybe_gz(path) as f:
        buf = f.read()
    endian = "<"
    if struct.unpack("<i", buf[:4])[0] != 348:
        endian = ">"
        if struct.unpack(">i", buf[:4])[0] != 348:
            if struct.unpack("<i", buf[:4])[0] == 540:
                raise ValueError("NIfTI-2 files are not supported yet")
            raise ValueError("Not a NIfTI-1 file: %s" % path)
    u = lambda fmt, off: struct.unpack(endian + fmt, buf[off: off + struct.calcsize(fmt)])
    dim = u("8h", 40)
    datatype = u("h", 70)[0]
    pixdim = u("8f", 76)
    vox_offset = int(u("f", 108)[0])
    scl_slope, scl_inter = u("2f", 112)
    qform_code, sform_code = u("2h", 252)
    quat = u("6f", 256)
    srow = np.array(u("12f", 280)).reshape(3, 4)

    ndim = dim[0]
    shape = tuple(int(x) for x in dim[1: ndim + 1])
    if datatype not in _NIFTI_DTYPES:
        raise ValueError("Unsupported NIfTI datatype %d" % datatype)
    dtype = np.dtype(endian + _NIFTI_DTYPES[datatype]) if _NIFTI_DTYPES[datatype][-1] != "1" else np.dtype(_NIFTI_DTYPES[datatype])
    n = int(np.prod(shape))
    raw = np.frombuffer(buf, dtype=dtype, count=n, offset=vox_offset)
    arr = raw.reshape(shape, order="F")
    if scl_slope not in (0.0, 1.0) or (scl_slope != 0.0 and scl_inter != 0.0):
        arr = arr.astype(np.float32) * scl_slope + scl_inter

    if sform_code > 0:
        M = np.eye(4)
        M[:3, :] = srow
    elif qform_code > 0:
        M = _quatern_to_mat(quat[0], quat[1], quat[2], quat[3], quat[4], quat[5],
                            pixdim[1], pixdim[2], pixdim[3], pixdim[0])
    else:
        M = np.diag([pixdim[1] or 1.0, pixdim[2] or 1.0, pixdim[3] or 1.0, 1.0])
    header = {"dim": dim, "pixdim": pixdim, "datatype": datatype, "sform_code": sform_code, "qform_code": qform_code}
    return Image(arr, M, header, path)


def read_image(path):
    """Read .mif/.mif.gz/.mih/.nii/.nii.gz based on the extension."""
    low = path.lower()
    if low.endswith((".mif", ".mif.gz", ".mih")):
        return read_mif(path)
    if low.endswith((".nii", ".nii.gz")):
        return read_nifti(path)
    raise ValueError("Unsupported image format: %s" % path)


def write_mif(path, data, ijk_to_ras, extra_header=None):
    """Write a float32 .mif image (used by tests and for exporting)."""
    data = np.asarray(data, dtype="<f4")
    vox = np.linalg.norm(ijk_to_ras[:3, :3], axis=0)
    rot = ijk_to_ras[:3, :3] / vox[None, :]
    dims = list(data.shape)
    vox_all = list(vox) + [1.0] * (len(dims) - 3)
    lines = ["mrtrix image",
             "dim: " + ",".join(str(d) for d in dims),
             "vox: " + ",".join("%g" % v for v in vox_all),
             "layout: " + ",".join("+%d" % i for i in range(len(dims))),
             "datatype: Float32LE"]
    for r in range(3):
        lines.append("transform: %s" % ",".join("%.10g" % x for x in list(rot[r]) + [ijk_to_ras[r, 3]]))
    for k, v in (extra_header or {}).items():
        lines.append("%s: %s" % (k, v))
    body = "\n".join(lines) + "\n"
    offset = len(body) + 32
    offset += (-offset) % 16
    hdr = (body + "file: . %d\nEND\n" % offset).encode("latin-1").ljust(offset, b"\0")
    with open(path, "wb") as f:
        f.write(hdr)
        data.ravel(order="F").tofile(f)


def is_sh_image(image):
    """True if the 4th dimension is a valid even-order SH coefficient count."""
    if image.data.ndim != 4:
        return False
    n = image.data.shape[3]
    return lmax_for_ncoefs(n) is not None


def lmax_for_ncoefs(n):
    for lmax in range(0, 32, 2):
        c = (lmax + 1) * (lmax + 2) // 2
        if c == n:
            return lmax
        if c > n:
            return None
    return None


def peek_image_dims(path):
    """Return the image dimensions from the header only (no voxel data read)."""
    low = path.lower()
    if low.endswith((".mif", ".mif.gz", ".mih")):
        with _open_maybe_gz(path) as f:
            _, header = _read_text_header(f, "mrtrix image")
        return [int(x) for x in _header_value(header, "dim").split(",")]
    if low.endswith((".nii", ".nii.gz")):
        with _open_maybe_gz(path) as f:
            buf = f.read(348)
        endian = "<" if struct.unpack("<i", buf[:4])[0] == 348 else ">"
        dim = struct.unpack(endian + "8h", buf[40:56])
        return [int(x) for x in dim[1: dim[0] + 1]]
    raise ValueError("Unsupported image format: %s" % path)


def looks_like_fod(path):
    """Heuristic used by the 'open' button: 4D with a typical SH coefficient count."""
    try:
        dims = peek_image_dims(path)
    except Exception:
        return False
    return len(dims) == 4 and dims[3] in (15, 28, 45, 66, 91)
