"""Slicer scene objects for tractograms, FOD glyphs and ROIs.

Each layer owns its MRML nodes. Geometry is computed with numpy (see tracts.py
and sh.py) and pushed to VTK in one go so updates stay interactive.
"""

import itertools

import numpy as np
import slicer
import vtk

from . import mrtrix_io, sh, tracts, vtk_utils

POINT_DATA = vtk.vtkAssignAttribute.POINT_DATA

COLOR_MODES = [
    ("direction", "Direction (locale)"),
    ("endpoints", "Direction (extrémités)"),
    ("solid", "Couleur unie"),
    ("random", "Aléatoire par fibre"),
    ("length", "Longueur de fibre"),
    ("tsf", "Scalaires (.tsf)"),
]
RGB_ARRAYS = {"direction": "Direction", "endpoints": "Endpoints", "random": "Random"}
SCALAR_ARRAYS = {"length": "Length", "tsf": "TSF"}

ROI_MODE_COLORS = {
    "include": (0.20, 0.80, 0.45),
    "exclude": (0.92, 0.30, 0.25),
    "endpoints": (0.98, 0.75, 0.10),
}

_uid = itertools.count(1)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def vtk_matrix(np_mat):
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(np_mat[r, c]))
    return m


def np_matrix(vtk_mat):
    return np.array([[vtk_mat.GetElement(r, c) for c in range(4)] for r in range(4)])


def slice_nodes():
    """Slice nodes that are shown in the current layout (Red, Green, Yellow, ...)."""
    nodes = []
    lm = slicer.app.layoutManager()
    if lm is None:
        return nodes
    for name in lm.sliceViewNames():
        w = lm.sliceWidget(name)
        if w is not None and w.mrmlSliceNode() is not None:
            nodes.append(w.mrmlSliceNode())
    return nodes


def three_d_view_ids():
    return [n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLViewNode")]


def slice_plane(slice_node):
    m = np_matrix(slice_node.GetSliceToRAS())
    e1 = m[:3, 0] / (np.linalg.norm(m[:3, 0]) + 1e-12)
    e2 = m[:3, 1] / (np.linalg.norm(m[:3, 1]) + 1e-12)
    n = m[:3, 2] / (np.linalg.norm(m[:3, 2]) + 1e-12)
    return m[:3, 3].copy(), n, e1, e2


def slice_signature(slice_node):
    m = np_matrix(slice_node.GetSliceToRAS())
    fov = slice_node.GetFieldOfView()
    return tuple(np.round(m[:3, :].ravel(), 3)) + tuple(np.round(fov, 2))


def slice_view_bounds(slice_node):
    """In-plane (u, v) bounds of the visible area of a slice view."""
    origin, n, e1, e2 = slice_plane(slice_node)
    xy_to_ras = np_matrix(slice_node.GetXYToRAS())
    dims = slice_node.GetDimensions()
    corners = np.array([[0, 0, 0, 1], [dims[0], 0, 0, 1], [0, dims[1], 0, 1], [dims[0], dims[1], 0, 1]], float)
    ras = corners @ xy_to_ras.T
    rel = ras[:, :3] - origin
    u, v = rel @ e1, rel @ e2
    return u.min(), u.max(), v.min(), v.max()


def slice_intersection_point():
    """RAS point where the Red/Green/Yellow planes meet (default ROI position)."""
    planes = []
    for sn in slice_nodes():
        o, n, _, _ = slice_plane(sn)
        planes.append((n, o))
    if len(planes) >= 3:
        A = np.array([p[0] for p in planes[:3]])
        b = np.array([np.dot(p[0], p[1]) for p in planes[:3]])
        if abs(np.linalg.det(A)) > 1e-6:
            return np.linalg.solve(A, b)
    return None


def _hide(node):
    node.SetHideFromEditors(True)
    node.SetSelectable(False)
    return node


def _new_model(name, polydata, hidden=False):
    model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    model.SetAndObservePolyData(polydata)
    model.CreateDefaultDisplayNodes()
    if hidden:
        _hide(model)
    return model


def _restrict_views(display_node, view_ids):
    display_node.RemoveAllViewNodeIDs()
    for vid in view_ids:
        display_node.AddViewNodeID(vid)


def _remove_model(model):
    if model is None or model.GetScene() is None:
        return
    for i in reversed(range(model.GetNumberOfDisplayNodes())):
        dn = model.GetNthDisplayNode(i)
        if dn is not None:
            slicer.mrmlScene.RemoveNode(dn)
    slicer.mrmlScene.RemoveNode(model)


def mrview_folder():
    sh_node = slicer.mrmlScene.GetSubjectHierarchyNode()
    scene_item = sh_node.GetSceneItemID()
    item = sh_node.GetItemChildWithName(scene_item, "MRView")
    if not item:
        item = sh_node.CreateFolderItem(scene_item, "MRView")
    return item


def put_in_folder(node):
    try:
        sh_node = slicer.mrmlScene.GetSubjectHierarchyNode()
        sh_node.SetItemParent(sh_node.GetItemByDataNode(node), mrview_folder())
    except Exception:
        pass


def add_volume_from_image(image, name, index=0, show=True):
    arr = image.volume(index)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    slicer.util.updateVolumeFromArray(node, np.ascontiguousarray(np.transpose(arr, (2, 1, 0))))
    node.SetIJKToRASMatrix(vtk_matrix(image.ijk_to_ras))
    node.CreateDefaultDisplayNodes()
    if show:
        slicer.util.setSliceViewerLayers(background=node, fit=True)
    return node


def vector_reference(vec):
    """Brightness reference of a vector map: 99.5th percentile of the largest component."""
    peak = np.abs(vec).max(axis=-1)
    vals = peak[peak > 0]
    if vals.size == 0:
        return 1.0
    ref = float(np.percentile(vals, 99.5))
    return ref if ref > 0 else float(vals.max())


def vector_to_rgb(vec, ref, gain=1.0):
    """[i,j,k,3] vectors -> uint8 RGB array in Slicer order [k,j,i,3].

    Same mapping as mrview's RGB colour map: |x| -> red, |y| -> green, |z| -> blue,
    scaled so that `ref / gain` is full brightness.
    """
    scale = 255.0 * gain / max(ref, 1e-12)
    out = np.empty(vec.shape, dtype=np.float32)
    np.abs(vec, out=out)
    out *= scale
    np.clip(out, 0, 255, out=out)
    return np.ascontiguousarray(np.transpose(out.astype(np.uint8), (2, 1, 0, 3)))


class RGBLayer(object):
    """Colour-coded vector map (e.g. FA-weighted principal eigenvector) shown as an RGB volume."""

    def __init__(self, image, name, path=None):
        self.uid = next(_uid)
        self.name = name
        self.path = path
        vec = np.array(image.data, dtype=np.float32)  # own writable copy (file data may be read-only)
        vec = vec.reshape(vec.shape[:3] + (3,))
        self.vec = np.nan_to_num(vec, copy=False)
        self.ref = vector_reference(self.vec)
        self.gain = 1.0
        self.spacing = image.spacing
        self.node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", name.split(".")[0])
        self.node.SetIJKToRASMatrix(vtk_matrix(image.ijk_to_ras))
        slicer.util.updateVolumeFromArray(self.node, vector_to_rgb(self.vec, self.ref, self.gain))
        self.node.CreateDefaultDisplayNodes()
        dn = self.node.GetDisplayNode()
        if dn is not None:
            dn.SetAutoWindowLevel(False)
            dn.SetWindowLevel(255.0, 127.5)
        put_in_folder(self.node)

    def info_text(self):
        i, j, k = self.vec.shape[:3]
        sp = self.spacing
        return "%d × %d × %d voxels · %.2g × %.2g × %.2g mm" % (i, j, k, sp[0], sp[1], sp[2])

    def set_gain(self, gain):
        self.gain = float(gain)
        if self.node is not None and self.node.GetScene() is not None:
            slicer.util.updateVolumeFromArray(self.node, vector_to_rgb(self.vec, self.ref, self.gain))

    def remove(self):
        if self.node is not None and self.node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(self.node)
        self.node = None


def colormap_lut(color_node_id, n=256):
    node = slicer.mrmlScene.GetNodeByID(color_node_id) if color_node_id else None
    lut = node.GetLookupTable() if node is not None else None
    if lut is None:
        x = np.linspace(0, 1, n)
        return np.stack([x, x, x], 1)
    rng = lut.GetRange()
    out = np.zeros((n, 3))
    for i, t in enumerate(np.linspace(rng[0], rng[1], n)):
        out[i] = lut.GetColor(t)
    return out


# ---------------------------------------------------------------------------
# Tractogram layer
# ---------------------------------------------------------------------------

class TractLayer(object):
    def __init__(self, tractogram, name, path=None, max_display=150000):
        self.uid = next(_uid)
        self.name = name
        self.path = path
        self.full = tractogram
        self.max_display = max_display
        # display properties
        self.visible = True
        self.show3d = True
        self.show2d = True
        self.color_mode = "direction"
        self.solid_color = (1.0, 0.85, 0.3)
        self.colormap_id = None
        self.scalar_range = None
        self.opacity = 1.0
        self.line_width = 2.0
        self.tubes = False
        self.slab_mm = 5.0
        self.full_tsf = None
        self.tsf_name = None

        self.model3d = None
        self.slab_models = {}  # slice node id -> model
        self.slab_signatures = {}
        self.roi_cache = {}
        self._build_working_set()
        self._create_3d_model()

    # -- data ----------------------------------------------------------------
    def _build_working_set(self):
        n = self.full.n_streamlines
        if self.max_display and n > self.max_display:
            rng = np.random.RandomState(42)
            self.ws_index = np.sort(rng.choice(n, self.max_display, replace=False))
            self.ws = self.full.subset(self.ws_index)
        else:
            self.ws_index = None
            self.ws = self.full
        self.selection = np.ones(self.ws.n_streamlines, dtype=bool)
        self._sel_points = None
        self._all_cells = None
        self.roi_cache = {}
        self.vtk_points = vtk_utils.vtk_points(self.ws.points)
        self.point_arrays = {}

    @property
    def n_selected(self):
        return int(self.selection.sum())

    @property
    def n_displayed_total(self):
        return self.ws.n_streamlines

    @property
    def is_subsampled(self):
        return self.ws_index is not None

    def set_max_display(self, value):
        if value == self.max_display:
            return
        self.max_display = value
        old_slabs = list(self.slab_models.values())
        self._build_working_set()
        # Swap in fresh meshes in one call: mutating a rendered mesh piece by piece
        # lets VTK render cells/arrays that no longer match the points (segfault).
        self.model3d.SetAndObservePolyData(self._new_polydata())
        for m in old_slabs:
            m.SetAndObservePolyData(self._new_polydata())
        self.slab_signatures = {}
        self.apply_display()
        self.update_3d()

    def ensure_array(self, mode):
        if mode in ("solid",):
            return None
        name = RGB_ARRAYS.get(mode) or SCALAR_ARRAYS.get(mode)
        if name in self.point_arrays:
            return name
        if mode == "direction":
            values = self.ws.direction_rgb()
        elif mode == "endpoints":
            values = self.ws.endpoint_rgb()
        elif mode == "random":
            values = self.ws.random_rgb(seed=self.uid)
        elif mode == "length":
            values = self.ws.per_point(self.ws.lengths_mm())
        elif mode == "tsf":
            if self.full_tsf is None:
                return None
            if self.ws_index is not None:
                values = self.full_tsf[self.full.point_indices(self.ws_index)]
            else:
                values = self.full_tsf
        else:
            return None
        arr = vtk_utils.data_array(values, name)
        self.point_arrays[name] = (arr, values)
        for pd in self._all_polydata():
            pd.GetPointData().AddArray(arr)
        return name

    def scalar_values(self, mode):
        name = self.ensure_array(mode)
        if name is None or name not in self.point_arrays:
            return None
        return self.point_arrays[name][1]

    def set_tsf(self, values, name):
        values = np.asarray(values, dtype=np.float32)
        if values.size != self.full.n_points:
            raise ValueError("Le fichier .tsf a %d valeurs mais le tractogramme a %d points."
                             % (values.size, self.full.n_points))
        self.full_tsf = values
        self.tsf_name = name
        if "TSF" in self.point_arrays:
            for pd in self._all_polydata():
                pd.GetPointData().RemoveArray("TSF")
            del self.point_arrays["TSF"]
        self.scalar_range = None

    def default_scalar_range(self, mode):
        v = self.scalar_values(mode)
        if v is None or v.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    # -- nodes ---------------------------------------------------------------
    def _all_polydata(self):
        pds = []
        if self.model3d is not None:
            pds.append(self.model3d.GetPolyData())
        pds += [m.GetPolyData() for m in self.slab_models.values()]
        return pds

    def _new_polydata(self):
        pd = vtk.vtkPolyData()
        pd.SetPoints(self.vtk_points)
        pd.SetLines(vtk_utils.empty_cells())
        for arr, _ in self.point_arrays.values():
            pd.GetPointData().AddArray(arr)
        return pd

    def _create_3d_model(self):
        self.model3d = _new_model(self.name, self._new_polydata())
        self.model3d.SetAttribute("MRView.Layer", "tract")
        put_in_folder(self.model3d)
        dn = self.model3d.GetDisplayNode()
        dn.SetVisibility2D(False)
        dn.SetVisibility3D(True)
        dn.SetBackfaceCulling(False)
        self.apply_display()
        self.update_3d()

    def _slab_model(self, slice_node):
        sid = slice_node.GetID()
        model = self.slab_models.get(sid)
        if model is not None and model.GetScene() is not None:
            return model
        model = _new_model("%s [2D %s]" % (self.name, slice_node.GetLayoutName()), self._new_polydata(), hidden=True)
        model.SetAttribute("MRView.Layer", "tract-slab")
        dn = model.GetDisplayNode()
        _restrict_views(dn, [sid])
        dn.SetVisibility3D(False)
        dn.SetVisibility2D(True)
        dn.SetSliceDisplayModeToProjection()
        self.slab_models[sid] = model
        self.slab_signatures.pop(sid, None)
        self._apply_display_to(dn, two_d=True)
        return model

    def display_nodes(self):
        out = []
        if self.model3d is not None and self.model3d.GetDisplayNode() is not None:
            out.append((self.model3d.GetDisplayNode(), False))
        for m in self.slab_models.values():
            if m.GetScene() is not None and m.GetDisplayNode() is not None:
                out.append((m.GetDisplayNode(), True))
        return out

    def _apply_display_to(self, dn, two_d):
        wasModifying = dn.StartModify()
        mode = self.color_mode
        name = self.ensure_array(mode)
        if mode == "solid" or name is None:
            dn.SetScalarVisibility(False)
            dn.SetColor(*self.solid_color)
        elif mode in RGB_ARRAYS:
            dn.SetActiveScalar(name, POINT_DATA)
            dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
            dn.SetScalarVisibility(True)
        else:
            dn.SetActiveScalar(name, POINT_DATA)
            if self.colormap_id:
                dn.SetAndObserveColorNodeID(self.colormap_id)
            if self.scalar_range is None:
                self.scalar_range = self.default_scalar_range(mode)
            dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            dn.SetScalarRange(*self.scalar_range)
            dn.SetScalarVisibility(True)
        if two_d:
            dn.SetVisibility(self.visible and self.show2d)
            dn.SetSliceIntersectionOpacity(self.opacity)
            dn.SetSliceIntersectionThickness(max(1, int(round(self.line_width))))
        else:
            dn.SetVisibility(self.visible and self.show3d)
            dn.SetOpacity(self.opacity)
            dn.SetLineWidth(self.line_width * (2.0 if self.tubes else 1.0))
        dn.EndModify(wasModifying)

    def apply_display(self):
        for dn, two_d in self.display_nodes():
            self._apply_display_to(dn, two_d)
        self.apply_tubes()

    def apply_tubes(self):
        """GPU impostor tubes (VTK RenderLinesAsTubes) on the 3D actor."""
        if self.model3d is None:
            return
        dn = self.model3d.GetDisplayNode()
        lm = slicer.app.layoutManager()
        if lm is None or dn is None:
            return
        for i in range(lm.threeDViewCount):
            try:
                view = lm.threeDWidget(i).threeDView()
                dm = view.displayableManagerByClassName("vtkMRMLModelDisplayableManager")
                actor = dm.GetActorByID(dn.GetID()) if dm is not None else None
                if actor is not None:
                    actor.GetProperty().SetRenderLinesAsTubes(bool(self.tubes))
                    view.scheduleRender()
            except Exception:
                pass

    # -- geometry ------------------------------------------------------------
    def set_selection(self, selection):
        if selection is None:
            selection = np.ones(self.ws.n_streamlines, dtype=bool)
        if np.array_equal(selection, self.selection):
            return False
        self.selection = selection
        self._sel_points = None
        self.slab_signatures = {}
        self.update_3d()
        return True

    @property
    def selected_points_mask(self):
        if self.selection.all():
            return None
        if self._sel_points is None:
            self._sel_points = self.ws.per_point(self.selection)
        return self._sel_points

    def update_3d(self):
        if self.model3d is None or self.model3d.GetScene() is None:
            return
        pd = self.model3d.GetPolyData()
        if self.selection.all():
            if self._all_cells is None:
                self._all_cells = vtk_utils.cell_array(self.ws.offsets, np.arange(self.ws.n_points, dtype=np.int64))
            cells = self._all_cells
        else:
            sel = np.flatnonzero(self.selection)
            counts = self.ws.counts[sel]
            offsets = np.zeros(sel.size + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            cells = vtk_utils.cell_array(offsets, tracts.concat_ranges(self.ws.starts[sel], counts))
        pd.SetLines(cells)
        pd.Modified()

    def update_slab(self, slice_node, force=False):
        if not (self.visible and self.show2d):
            model = self.slab_models.get(slice_node.GetID())
            if model is not None:
                model.GetDisplayNode().SetVisibility(False)
            return
        model = self._slab_model(slice_node)
        model.GetDisplayNode().SetVisibility(True)
        sig = slice_signature(slice_node)[:12] + (self.slab_mm,)
        if not force and self.slab_signatures.get(slice_node.GetID()) == sig:
            return
        self.slab_signatures[slice_node.GetID()] = sig
        origin, n, _, _ = slice_plane(slice_node)
        segs = self.ws.slab_segments(origin, n, 0.5 * self.slab_mm, self.selected_points_mask)
        pd = model.GetPolyData()
        pd.SetLines(vtk_utils.segment_cells(segs))
        pd.Modified()

    def invalidate_slabs(self):
        self.slab_signatures = {}

    def remove(self):
        _remove_model(self.model3d)
        for m in self.slab_models.values():
            _remove_model(m)
        self.model3d = None
        self.slab_models = {}

    def stats_text(self):
        L = self.full.lengths_mm()
        txt = "%s fibres · %s points" % (_fmt_int(self.full.n_streamlines), _fmt_int(self.full.n_points))
        if L.size:
            txt += " · longueur moy. %.1f mm" % float(L.mean())
        if self.is_subsampled:
            txt += "<br><span style='color:#d68910'>Affichage d'un sous-échantillon de %s fibres</span>" % _fmt_int(self.ws.n_streamlines)
        return txt

    def selection_for_full(self, roi_entries, include_mode):
        """Apply the ROI filters to the full tractogram (used for export)."""
        incl, excl = [], []
        for e in roi_entries:
            hits = tracts.streamline_hits(self.full, e.point_test(), endpoints_only=(e.mode == "endpoints"))
            (excl if e.mode == "exclude" else incl).append(hits)
        return tracts.combine_selection(self.full.n_streamlines, incl, excl, include_mode)


def _fmt_int(n):
    return "{:,}".format(int(n)).replace(",", " ")


# ---------------------------------------------------------------------------
# FOD layer
# ---------------------------------------------------------------------------

DETAIL_LEVELS = [(1, "Bas (42 dir.)"), (2, "Moyen (162 dir.)"), (3, "Haut (642 dir.)")]
MAX_GLYPH_VERTICES = 2500000


class FODLayer(object):
    def __init__(self, image, name, path=None, show_background=True):
        self.uid = next(_uid)
        self.name = name
        self.path = path
        self.lmax = mrtrix_io.lmax_for_ncoefs(image.data.shape[3])
        if self.lmax is None:
            raise ValueError("La 4e dimension (%d) ne correspond pas à des coefficients SH pairs." % image.data.shape[3])
        self.shape = image.data.shape
        self.ijk_to_ras = image.ijk_to_ras
        self.spacing = np.linalg.norm(image.ijk_to_ras[:3, :3], axis=0)
        data = image.data
        c0 = np.asarray(data[..., 0])
        mask = np.isfinite(c0) & (c0 > 0)
        ijk = np.argwhere(mask)
        self.coefs = np.ascontiguousarray(data[mask], dtype=np.float32)
        self.centers = (ijk @ self.ijk_to_ras[:3, :3].T + self.ijk_to_ras[:3, 3]).astype(np.float32)
        self.templates = {}
        # Reference amplitude: 95th percentile of the peak amplitude over a random subset
        tpl = self.template(2)
        rng = np.random.RandomState(0)
        sample = self.coefs[rng.choice(self.coefs.shape[0], min(20000, self.coefs.shape[0]), replace=False)] if self.coefs.shape[0] else self.coefs
        peaks = sh.amplitudes(sample, tpl).max(1) if sample.size else np.ones(1)
        self.ref = float(np.percentile(peaks, 95)) if peaks.size else 1.0
        if self.ref <= 0:
            self.ref = 1.0

        # display properties
        self.visible = True
        self.show2d = True
        self.show3d = False
        self.scale = 1.0
        self.threshold = 0.1
        self.level = 2
        self.normalize = False
        self.color_mode = "direction"
        self.solid_color = (0.85, 0.85, 0.85)
        self.opacity = 1.0
        self.models = {}  # slice node id -> model
        self.signatures = {}
        self.glyph_counts = {}

        self.l0_node = add_volume_from_image(_Wrap(c0, image.ijk_to_ras), "%s (l=0)" % name, show=False)
        put_in_folder(self.l0_node)
        if show_background:
            lm = slicer.app.layoutManager()
            has_bg = False
            if lm is not None:
                for sn in lm.sliceViewNames():
                    if lm.sliceWidget(sn).sliceLogic().GetSliceCompositeNode().GetBackgroundVolumeID():
                        has_bg = True
            if not has_bg:
                slicer.util.setSliceViewerLayers(background=self.l0_node, fit=True)

    def template(self, level):
        if level not in self.templates:
            self.templates[level] = sh.GlyphTemplate(self.lmax, level)
        return self.templates[level]

    def _model(self, slice_node):
        sid = slice_node.GetID()
        model = self.models.get(sid)
        if model is not None and model.GetScene() is not None:
            return model
        pd = vtk.vtkPolyData()
        model = _new_model("%s [glyphes %s]" % (self.name, slice_node.GetLayoutName()), pd, hidden=True)
        model.SetAttribute("MRView.Layer", "fod")
        dn2 = model.GetDisplayNode()
        _restrict_views(dn2, [sid])
        dn2.SetVisibility3D(False)
        dn2.SetVisibility2D(True)
        dn2.SetSliceDisplayModeToProjection()
        dn3 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        _hide(dn3)
        model.AddAndObserveDisplayNodeID(dn3.GetID())
        _restrict_views(dn3, three_d_view_ids())
        dn3.SetVisibility2D(False)
        dn3.SetVisibility3D(True)
        for dn, arr in ((dn2, "Shaded"), (dn3, "RGB")):
            dn.SetActiveScalar(arr, POINT_DATA)
            dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
            dn.SetScalarVisibility(True)
            dn.SetBackfaceCulling(False)
        self.models[sid] = model
        self.signatures.pop(sid, None)
        self._apply_display(model)
        return model

    def _apply_display(self, model):
        dn2 = model.GetNthDisplayNode(0)
        dn3 = model.GetNthDisplayNode(1)
        if dn2 is not None:
            dn2.SetVisibility(self.visible and self.show2d)
            dn2.SetSliceIntersectionOpacity(self.opacity)
            dn2.SetOpacity(self.opacity)
        if dn3 is not None:
            dn3.SetVisibility(self.visible and self.show3d)
            dn3.SetOpacity(self.opacity)

    def apply_display(self):
        for m in self.models.values():
            if m.GetScene() is not None:
                self._apply_display(m)

    def invalidate(self):
        self.signatures = {}

    def update_view(self, slice_node, force=False):
        if not (self.visible and (self.show2d or self.show3d)):
            for m in self.models.values():
                if m.GetScene() is not None:
                    self._apply_display(m)
            return
        model = self._model(slice_node)
        sid = slice_node.GetID()
        sig = slice_signature(slice_node)
        if not force and self.signatures.get(sid) == sig:
            return
        self.signatures[sid] = sig
        origin, n, e1, e2 = slice_plane(slice_node)

        axes = self.ijk_to_ras[:3, :3]
        half = 0.5 * float(np.max(np.abs(axes.T @ n)))
        rel = self.centers - origin.astype(np.float32)
        d = rel @ n.astype(np.float32)
        sel = (d >= -half) & (d < half)
        # cull glyphs outside the visible part of the slice view
        try:
            u0, u1, v0, v1 = slice_view_bounds(slice_node)
            margin = float(self.spacing.max())
            u = rel @ e1.astype(np.float32)
            v = rel @ e2.astype(np.float32)
            sel &= (u >= u0 - margin) & (u <= u1 + margin) & (v >= v0 - margin) & (v <= v1 + margin)
        except Exception:
            pass
        idx = np.flatnonzero(sel)

        level = self.level
        while level > 0 and idx.size * self.template(level).n_verts > MAX_GLYPH_VERTICES:
            level -= 1
        tpl = self.template(level)
        if idx.size and tpl.n_verts * idx.size > MAX_GLYPH_VERTICES:
            idx = idx[:: int(np.ceil(idx.size * tpl.n_verts / MAX_GLYPH_VERTICES))]

        if idx.size == 0:
            self._set_empty(model)
            self.glyph_counts[sid] = 0
            return
        amps = sh.amplitudes(self.coefs[idx], tpl)
        peak = amps.max(1)
        keep = peak > self.threshold * self.ref
        idx, amps, peak = idx[keep], amps[keep], peak[keep]
        self.glyph_counts[sid] = int(idx.size)
        if idx.size == 0:
            self._set_empty(model)
            return
        if self.normalize:
            r = amps / peak[:, None]
        else:
            r = amps / self.ref
        scale_mm = 0.5 * self.scale * float(self.spacing.min())
        mesh = sh.build_glyph_mesh(self.centers[idx], r, tpl, scale_mm, view_normal=n)
        if self.color_mode == "solid":
            rgb = np.empty((mesh["points"].shape[0], 3), dtype=np.uint8)
            rgb[:] = (np.asarray(self.solid_color) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb = mesh["rgb"]
        shaded = (rgb * mesh["shade"][:, None]).astype(np.uint8)
        # Build a complete new mesh and swap it in with a single call. Editing the
        # displayed mesh in place can trigger a render between SetPoints and
        # SetNormals, where VTK reads arrays of the wrong length and crashes.
        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_utils.vtk_points(mesh["points"]))
        pd.SetPolys(vtk_utils.triangle_cells(mesh["triangles"]))
        pd.GetPointData().SetNormals(vtk_utils.data_array(mesh["normals"], "Normals"))
        pd.GetPointData().AddArray(vtk_utils.data_array(rgb, "RGB"))
        pd.GetPointData().AddArray(vtk_utils.data_array(shaded, "Shaded"))
        model.SetAndObservePolyData(pd)

    def _set_empty(self, model):
        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk.vtkPoints())
        pd.SetPolys(vtk_utils.empty_cells())
        model.SetAndObservePolyData(pd)

    def info_text(self):
        s = self.shape
        return "lmax %d · %d×%d×%d · %s mm · %s voxels" % (
            self.lmax, s[0], s[1], s[2], "×".join("%.2g" % x for x in self.spacing), _fmt_int(self.coefs.shape[0]))

    def remove(self, remove_volume=True):
        for m in self.models.values():
            _remove_model(m)
        self.models = {}
        if remove_volume and self.l0_node is not None and self.l0_node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(self.l0_node)
        self.l0_node = None


class _Wrap(object):
    def __init__(self, data, ijk_to_ras):
        self.data = data
        self.ijk_to_ras = ijk_to_ras

    def volume(self, index=0):
        return self.data


# ---------------------------------------------------------------------------
# ROIs
# ---------------------------------------------------------------------------

class ROIEntry(object):
    def __init__(self, node, shape, mode="include", segment_id=None):
        self.uid = next(_uid)
        self.node = node
        self.shape = shape  # 'box' | 'sphere' | 'mask'
        self.mode = mode
        self.enabled = True
        self.segment_id = segment_id
        self.sphere_model = None
        self.version = 0  # bumped when a mask changes
        self._mask = None
        self._ras_to_ijk = None
        self.observer_tags = []

    @property
    def name(self):
        if self.shape == "mask" and self.segment_id and self.node is not None:
            seg = self.node.GetSegmentation().GetSegment(self.segment_id)
            if seg is not None:
                return seg.GetName()
        return self.node.GetName() if self.node is not None else "?"

    @property
    def shape_label(self):
        return {"box": "Boîte", "sphere": "Sphère", "mask": "Masque"}[self.shape]

    # -- geometry ------------------------------------------------------------
    def _box_geometry(self):
        o2w = np_matrix(self.node.GetObjectToWorldMatrix())
        w2o = np.linalg.inv(o2w)
        c = [0.0, 0.0, 0.0]
        self.node.GetCenterWorld(c)
        c_obj = w2o[:3, :3] @ np.asarray(c) + w2o[:3, 3]
        w2o[:3, 3] -= c_obj
        half = 0.5 * np.asarray(self.node.GetSize(), dtype=float)
        return w2o, half

    def signature(self):
        if self.shape == "mask":
            return ("mask", self.version)
        w2o, half = self._box_geometry()
        return (self.shape,) + tuple(np.round(w2o[:3, :].ravel(), 4)) + tuple(np.round(half, 4))

    def load_mask(self):
        node = self.node
        if node.IsA("vtkMRMLSegmentationNode"):
            label = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__mrview_tmp")
            try:
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(self.segment_id)
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(node, ids, label)
                arr = slicer.util.arrayFromVolume(label)
                m = vtk.vtkMatrix4x4()
                label.GetRASToIJKMatrix(m)
            finally:
                slicer.mrmlScene.RemoveNode(label)
        else:
            arr = slicer.util.arrayFromVolume(node)
            m = vtk.vtkMatrix4x4()
            node.GetRASToIJKMatrix(m)
        self._mask = np.ascontiguousarray(np.transpose(arr > 0, (2, 1, 0)))
        self._ras_to_ijk = np_matrix(m)
        self.version += 1

    def point_test(self):
        if self.shape == "mask":
            if self._mask is None:
                self.load_mask()
            mask, A = self._mask, self._ras_to_ijk
            return lambda p: tracts.points_in_mask(p, A, mask)
        w2o, half = self._box_geometry()
        if self.shape == "sphere":
            return lambda p: tracts.points_in_ellipsoid(p, w2o, half)
        return lambda p: tracts.points_in_box(p, w2o, half)

    # -- appearance ----------------------------------------------------------
    @property
    def color(self):
        return ROI_MODE_COLORS[self.mode]

    def apply_color(self):
        c = self.color
        if self.shape in ("box", "sphere"):
            dn = self.node.GetDisplayNode()
            if dn is not None:
                dn.SetSelectedColor(*c)
                dn.SetColor(*c)
                try:
                    if self.shape == "sphere":
                        dn.SetFillVisibility(False)
                        dn.SetOutlineOpacity(0.25)
                    else:
                        dn.SetFillOpacity(0.15)
                except Exception:
                    pass
        if self.sphere_model is not None:
            self.sphere_model.GetDisplayNode().SetColor(*c)
        elif self.shape == "mask" and self.node.IsA("vtkMRMLSegmentationNode") and self.segment_id:
            seg = self.node.GetSegmentation().GetSegment(self.segment_id)
            if seg is not None:
                seg.SetColor(*c)

    def update_sphere_model(self):
        if self.shape != "sphere":
            return
        if self.sphere_model is None or self.sphere_model.GetScene() is None:
            self.sphere_model = _new_model(self.node.GetName() + " [sphère]", vtk.vtkPolyData(), hidden=True)
            dn = self.sphere_model.GetDisplayNode()
            dn.SetOpacity(0.35)
            dn.SetVisibility2D(True)
            dn.SetSliceIntersectionThickness(2)
            dn.SetColor(*self.color)
        w2o, half = self._box_geometry()
        o2w = np.linalg.inv(w2o) @ np.diag(list(half) + [1.0])
        src = vtk.vtkSphereSource()
        src.SetThetaResolution(36)
        src.SetPhiResolution(18)
        src.SetRadius(1.0)
        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix(o2w))
        f = vtk.vtkTransformPolyDataFilter()
        f.SetInputConnection(src.GetOutputPort())
        f.SetTransform(tf)
        f.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(f.GetOutput())
        self.sphere_model.SetAndObservePolyData(out)

    def set_visible(self, visible):
        dn = self.node.GetDisplayNode()
        if dn is not None and self.shape != "mask":
            dn.SetVisibility(visible)
        if self.sphere_model is not None and self.sphere_model.GetDisplayNode() is not None:
            self.sphere_model.GetDisplayNode().SetVisibility(visible)

    def remove(self, remove_node=True):
        _remove_model(self.sphere_model)
        self.sphere_model = None
        if remove_node and self.shape != "mask" and self.node is not None and self.node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(self.node)
