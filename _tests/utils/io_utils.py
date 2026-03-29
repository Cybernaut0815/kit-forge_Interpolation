"""USD geometry I/O utilities for interpolation test scripts."""

import numpy as np
from pxr import Usd, UsdGeom, Vt


def load_usd_file(stage_path):
    """
    Load a USD file.

    Args:
        stage_path (str): Path to the USD file

    Returns:
        Usd.Stage: USD stage
    """
    stage = Usd.Stage.Open(stage_path)
    if not stage:
        raise ValueError(f"Failed to open USD stage at: {stage_path}")
    return stage


def get_usd_mesh_vertices(usd_stage, mesh_path):
    """
    Get the vertices of a USD mesh.

    Args:
        usd_stage (Usd.Stage): USD stage
        mesh_path (str): Path to the USD mesh (e.g., '/World/Mesh')

    Returns:
        np.array: Vertex positions (N x 3)
    """
    prim = usd_stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        raise ValueError(f"No valid prim found at path: {mesh_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at {mesh_path} is not a mesh")

    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    vertices = np.array(points, dtype=np.float32)
    return vertices


def update_usd_mesh_vertices(usd_stage, mesh_path, new_vertices):
    """
    Update the vertices of a USD mesh.

    Args:
        usd_stage (Usd.Stage): USD stage
        mesh_path (str): Path to the USD mesh (e.g., '/World/Mesh')
        new_vertices (np.array): New vertex positions (N x 3)

    Returns:
        Usd.Stage: USD stage with updated mesh
    """
    prim = usd_stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        raise ValueError(f"No valid prim found at path: {mesh_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at {mesh_path} is not a mesh")

    if new_vertices.dtype != np.float32:
        new_vertices = new_vertices.astype(np.float32)

    points_array = Vt.Vec3fArray.FromBuffer(new_vertices.flatten())

    points_attr = mesh.GetPointsAttr()
    points_attr.Set(points_array)

    return usd_stage


def add_lines_to_usd_stage(usd_stage, lines, line_path='/World/Lines', color=(1.0, 0.0, 0.0), line_width=2.0):
    """
    Add lines to a USD stage as BasisCurves.

    Args:
        usd_stage (Usd.Stage): USD stage
        lines (np.array): Array of line segments shape (N, 2, 3) where each line has 2 points
        line_path (str): Path where to add the lines in the stage
        color (tuple): RGB color for the lines (values 0-1)
        line_width (float): Width of the lines (default 2.0)

    Returns:
        Usd.Stage: USD stage with added lines
    """
    from pxr import Gf

    usd_stage.DefinePrim(line_path, 'Scope')

    all_points = []
    curve_vertex_counts = []

    for line in lines:
        all_points.extend([tuple(line[0]), tuple(line[1])])
        curve_vertex_counts.append(2)

    curves_path = f"{line_path}/Curves"
    curves_prim = usd_stage.DefinePrim(curves_path, 'BasisCurves')
    curves = UsdGeom.BasisCurves(curves_prim)

    points_array = Vt.Vec3fArray([Gf.Vec3f(*p) for p in all_points])
    curves.GetPointsAttr().Set(points_array)

    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray(curve_vertex_counts))

    curves.GetTypeAttr().Set('linear')
    curves.GetBasisAttr().Set('bezier')

    color_attr = curves.GetDisplayColorPrimvar()
    if not color_attr:
        color_attr = curves.CreateDisplayColorPrimvar()
    color_attr.Set([Gf.Vec3f(*color)])

    width_attr = curves.GetWidthsAttr()
    widths = [line_width] * len(all_points)
    width_attr.Set(Vt.FloatArray(widths))

    return usd_stage
