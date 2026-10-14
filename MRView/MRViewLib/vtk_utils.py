"""numpy <-> VTK helpers (VTK >= 9)."""

import numpy as np
import vtk
from vtk.util import numpy_support


def vtk_points(points):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=1))
    return pts


def id_array(values):
    return numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(values, dtype=np.int64), deep=1)


def cell_array(offsets, connectivity):
    """vtkCellArray from VTK9-style offsets (n_cells+1) and connectivity."""
    ca = vtk.vtkCellArray()
    ca.SetData(id_array(offsets), id_array(connectivity))
    return ca


def polyline_cells(offsets, point_ids):
    return cell_array(offsets, point_ids)


def segment_cells(segment_start_ids):
    """Two-point line cells (i, i+1) for each given i."""
    k = segment_start_ids.size
    conn = np.empty(2 * k, dtype=np.int64)
    conn[0::2] = segment_start_ids
    conn[1::2] = segment_start_ids + 1
    return cell_array(np.arange(0, 2 * k + 1, 2, dtype=np.int64), conn)


def triangle_cells(triangles):
    k = triangles.shape[0]
    return cell_array(np.arange(0, 3 * k + 1, 3, dtype=np.int64), triangles.reshape(-1))


def data_array(values, name):
    arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(values), deep=1)
    arr.SetName(name)
    return arr


def empty_cells():
    return cell_array(np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64))
