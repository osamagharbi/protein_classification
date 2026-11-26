#!/usr/bin/env python3
"""
VTK mesh/dataset -> SSL-ready graph converter (parallel)

Core behavior (default):
  - vertices -> nodes
  - mesh topology edges -> edges (NOT kNN)
  - no node reduction
  - feature adjustment for SSL:
      * xyz   -> centered + isotropically normalized (per-graph) for geometry reconstruction
      * normals -> unit vectors (for cosine/angular loss)
      * scalars -> raw + optional z-score normalization (train stats)

Optional:
  - reduce mesh BEFORE graph creation (trimesh quadric decimation), either by:
      * --target_vertices N   (best effort / approximate)
      * --reduce_ratio r      (keep ratio, e.g. 0.25 keeps ~25% face complexity)
    If reduced, scalar values are transferred to reduced vertices via nearest original vertex (KDTree).

Outputs one .pt file per input .vtk/.vtp containing a dict:
  x               [N, 8]  = [Potential_norm, NormalPotential_norm, nx, ny, nz, x_ssl, y_ssl, z_ssl]
  edge_index      [2, 2E] bidirectional edges (PyG-friendly COO)
  edge_index_mesh [2, E]  unique undirected mesh edges (exact topology edges of final mesh)
  pos             [N, 3]  xyz_ssl
  xyz_raw         [N, 3]  raw coordinates of final mesh vertices
  normals         [N, 3]  unit normals of final mesh vertices
  scalars_raw     [N, 2]  [Potential, NormalPotential]
  scalars_norm    [N, 2]  z-scored (if stats provided) else raw copy
  meta            dict

Notes:
  - If input is not vtkPolyData, boundary surface is extracted first.
  - If scalars exist only on cells, they are mapped to points.
  - No sampling, no kNN, no node dropping (except explicit failure on non-finite values).
"""

import os
import json
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import vtk
from vtk.util.numpy_support import vtk_to_numpy


# ---------------------------
# VTK I/O + preprocessing
# ---------------------------

def read_vtk_any(path: str):
    """Read legacy .vtk or XML .vtp."""
    ext = Path(path).suffix.lower()
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        reader = vtk.vtkGenericDataObjectReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
    else:
        raise ValueError(f"Unsupported extension: {ext} (expected .vtk or .vtp)")
    reader.SetFileName(path)
    reader.Update()
    out = reader.GetOutput()
    if out is None:
        raise RuntimeError(f"VTK reader returned no output for {path}")
    return out


def to_surface_polydata(data_obj):
    """Extract boundary surface polydata from any VTK dataset."""
    if isinstance(data_obj, vtk.vtkPolyData):
        poly = data_obj
    else:
        surf = vtk.vtkDataSetSurfaceFilter()
        surf.SetInputData(data_obj)
        surf.Update()
        poly = surf.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError("Surface extraction failed or produced empty polydata")
    return poly


def ensure_point_data_arrays(poly, required_names):
    """
    If required arrays are missing on points but exist on cells, convert cell data to point data.
    """
    pd = poly.GetPointData()
    cd = poly.GetCellData()

    need_c2p = False
    for name in required_names:
        if pd.GetArray(name) is None and cd is not None and cd.GetArray(name) is not None:
            need_c2p = True
            break

    if need_c2p:
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(poly)
        c2p.PassCellDataOn()
        c2p.Update()
        poly = c2p.GetOutput()

    return poly


def triangulate_polydata(poly):
    """Ensure polygons are triangles."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    out = tri.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        raise RuntimeError("Triangulation failed or produced empty polydata")
    return out


def compute_point_normals(poly):
    """
    Compute point normals on polydata.
    Splitting is off to avoid vertex duplication / topology changes.
    """
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(poly)
    nrm.ComputePointNormalsOn()
    nrm.ComputeCellNormalsOff()
    nrm.SplittingOff()
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOff()
    nrm.Update()
    out = nrm.GetOutput()
    if out is None:
        raise RuntimeError("vtkPolyDataNormals failed")
    return out


def vtk_scalar_array_to_numpy(arr, name, expected_len):
    if arr is None:
        raise KeyError(f"Missing required VTK array: {name}")
    x = np.asarray(vtk_to_numpy(arr))
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"{name} is not scalar-like: shape={x.shape}")
    if x.shape[0] != expected_len:
        raise ValueError(f"{name} length mismatch: {x.shape[0]} vs {expected_len}")
    return x.astype(np.float32, copy=False)


def vtk_vec3_array_to_numpy(arr, name, expected_len):
    if arr is None:
        raise KeyError(f"Missing required VTK array: {name}")
    x = np.asarray(vtk_to_numpy(arr))
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} is not vec3-like: shape={x.shape}")
    if x.shape[0] != expected_len:
        raise ValueError(f"{name} length mismatch: {x.shape[0]} vs {expected_len}")
    return x.astype(np.float32, copy=False)


def extract_points_features_from_vtk(filepath, potential_name="Potential", normalpot_name="NormalPotential"):
    """
    Returns:
      poly        vtkPolyData (surface, with point normals computed)
      xyz_raw     [N,3]
      normals_raw [N,3]
      scalars_raw [N,2] = [Potential, NormalPotential]
    """
    obj = read_vtk_any(filepath)
    poly = to_surface_polydata(obj)
    poly = ensure_point_data_arrays(poly, [potential_name, normalpot_name])
    poly = triangulate_polydata(poly)
    poly = compute_point_normals(poly)

    n = poly.GetNumberOfPoints()
    if n < 2:
        raise RuntimeError(f"Too few points: {n}")

    pts = poly.GetPoints()
    if pts is None or pts.GetData() is None:
        raise RuntimeError("PolyData has no point coordinates")
    xyz_raw = np.asarray(vtk_to_numpy(pts.GetData()), dtype=np.float32)
    if xyz_raw.ndim != 2 or xyz_raw.shape[1] != 3:
        raise RuntimeError(f"Unexpected points shape: {xyz_raw.shape}")

    pd = poly.GetPointData()

    normals_arr = pd.GetNormals()
    if normals_arr is None:
        normals_arr = pd.GetArray("Normals")
    normals_raw = vtk_vec3_array_to_numpy(normals_arr, "Normals", n)

    pot = vtk_scalar_array_to_numpy(pd.GetArray(potential_name), potential_name, n)
    npot = vtk_scalar_array_to_numpy(pd.GetArray(normalpot_name), normalpot_name, n)
    scalars_raw = np.stack([pot, npot], axis=1).astype(np.float32, copy=False)

    return poly, xyz_raw, normals_raw, scalars_raw


# ---------------------------
# Mesh topology edges (exact VTK mesh edges)
# ---------------------------

def extract_mesh_edges_exact(poly):
    """
    Extract exact topology edges from vtkPolyData using vtkExtractEdges.
    Returns unique undirected edges [E,2] with original point indices.
    """
    ex = vtk.vtkExtractEdges()
    ex.SetInputData(poly)

    # Preserve original point numbering if supported
    if hasattr(ex, "UseAllPointsOn"):
        ex.UseAllPointsOn()
    elif hasattr(ex, "SetUseAllPoints"):
        ex.SetUseAllPoints(True)

    ex.Update()
    edge_poly = ex.GetOutput()
    if edge_poly is None:
        raise RuntimeError("vtkExtractEdges failed")

    n_in = poly.GetNumberOfPoints()
    n_edge_pts = edge_poly.GetNumberOfPoints()

    # Fail loudly if point numbering may have changed
    if n_edge_pts != n_in:
        raise RuntimeError(
            f"vtkExtractEdges output point count ({n_edge_pts}) != input point count ({n_in}). "
            "Cannot guarantee exact vertex->node indexing. "
            "Try a VTK version with UseAllPoints support."
        )

    lines = edge_poly.GetLines()
    if lines is None:
        return np.zeros((0, 2), dtype=np.int64)

    id_list = vtk.vtkIdList()
    edges = set()

    lines.InitTraversal()
    while lines.GetNextCell(id_list):
        m = id_list.GetNumberOfIds()
        if m < 2:
            continue
        # Usually line segments, but handle polyline safely
        for i in range(m - 1):
            a = int(id_list.GetId(i))
            b = int(id_list.GetId(i + 1))
            if a == b:
                continue
            if a < b:
                edges.add((a, b))
            else:
                edges.add((b, a))

    if not edges:
        return np.zeros((0, 2), dtype=np.int64)

    return np.array(sorted(edges), dtype=np.int64)


# ---------------------------
# Trimesh reduction helpers (optional)
# ---------------------------

def vtk_polydata_to_trimesh(poly):
    """
    Convert triangulated vtkPolyData -> trimesh.Trimesh (geometry only).
    """
    poly = triangulate_polydata(poly)

    verts = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)

    polys = poly.GetPolys()
    if polys is None:
        raise RuntimeError("PolyData has no polygons for trimesh conversion")

    conn = np.asarray(vtk_to_numpy(polys.GetData()), dtype=np.int64)
    if conn.size == 0:
        raise RuntimeError("Empty polygon connectivity")

    faces = []
    i = 0
    while i < len(conn):
        n = int(conn[i]); i += 1
        if n < 3:
            i += n
            continue
        ids = conn[i:i+n]
        i += n
        if n == 3:
            faces.append(ids)
        else:
            # Triangle fan fallback (TriangleFilter should already triangulate)
            for j in range(1, n - 1):
                faces.append([ids[0], ids[j], ids[j + 1]])

    if not faces:
        raise RuntimeError("No triangular faces found")

    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise RuntimeError("Trimesh conversion produced empty mesh")
    return mesh


def trimesh_edges_from_faces(faces):
    """Build unique undirected edges [E,2] from triangular faces [F,3]."""
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0).astype(np.int64, copy=False)

    edges = np.sort(edges, axis=1)   # undirected canonical form
    edges = np.unique(edges, axis=0)
    return edges


def fallback_simplify_by_vertex_subsampling(mesh, target_ratio):
    """
    Fallback mesh simplification: subsample vertices uniformly by distance.
    Keeps approximately target_ratio fraction of vertices.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    n_target = max(4, int(len(vertices) * target_ratio))
    
    # Simple: subsample every nth vertex
    step = max(1, len(vertices) // n_target)
    keep_indices = np.arange(0, len(vertices), step)[:n_target]
    keep_indices = np.sort(keep_indices)
    
    # Create mapping: old_idx -> new_idx (or -1 if not kept)
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[keep_indices] = np.arange(len(keep_indices))
    
    # Filter faces to only those with all vertices kept
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mask = (old_to_new[faces[:, 0]] >= 0) & \
           (old_to_new[faces[:, 1]] >= 0) & \
           (old_to_new[faces[:, 2]] >= 0)
    
    new_faces = old_to_new[faces[mask]]
    
    if len(new_faces) == 0:
        return mesh.copy()  # No valid faces, return original
    
    # Create new mesh
    new_verts = vertices[keep_indices]
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    
    return new_mesh


def maybe_reduce_mesh_with_trimesh(
    poly,
    xyz_raw,
    scalars_raw,
    target_vertices=None,
    reduce_ratio=None,
    reduce_aggression=None,
):
    """
    Optional reduction BEFORE graph creation.

    Returns:
      None  -> if no reduction requested (default behavior)
      (xyz_out, normals_out, scalars_out, edges_undirected, reduction_meta)
    """
    if target_vertices is None and reduce_ratio is None:
        return None  # default = no reduction

    if target_vertices is not None and reduce_ratio is not None:
        raise ValueError("Use only one of --target_vertices or --reduce_ratio")

    mesh = vtk_polydata_to_trimesh(poly)

    n0_v = int(mesh.vertices.shape[0])
    n0_f = int(mesh.faces.shape[0])
    if n0_v < 4 or n0_f < 4:
        raise RuntimeError(f"Mesh too small to decimate safely (V={n0_v}, F={n0_f})")

    # trimesh decimation targets faces/percent; target_vertices is approximate via face estimate
    reduced = None
    use_fallback = False
    
    if target_vertices is not None:
        target_vertices = int(target_vertices)
        if target_vertices < 4:
            raise ValueError("--target_vertices must be >= 4")

        if target_vertices >= n0_v:
            reduced = mesh.copy()
            target_faces_est = n0_f
        else:
            target_ratio = target_vertices / max(n0_v, 1)
            face_per_vertex = n0_f / max(n0_v, 1)
            target_faces_est = max(4, int(round(target_vertices * face_per_vertex)))
            
            # Try quadric decimation first
            try:
                kwargs = {"face_count": target_faces_est}
                if reduce_aggression is not None:
                    kwargs["aggression"] = int(reduce_aggression)
                reduced = mesh.simplify_quadric_decimation(**kwargs)
            except Exception as e:
                print(f"  ⚠ Quadric decimation failed ({type(e).__name__}), using fallback vertex subsampling")
                use_fallback = True
                reduced = fallback_simplify_by_vertex_subsampling(mesh, target_ratio)

    else:
        reduce_ratio = float(reduce_ratio)
        if not (0.0 < reduce_ratio <= 1.0):
            raise ValueError("--reduce_ratio must be in (0, 1]")

        if reduce_ratio == 1.0:
            reduced = mesh.copy()
            target_faces_est = n0_f
        else:
            target_faces_est = max(4, int(round(n0_f * reduce_ratio)))
            
            # Try quadric decimation first
            try:
                kwargs = {"face_count": target_faces_est}
                if reduce_aggression is not None:
                    kwargs["aggression"] = int(reduce_aggression)
                reduced = mesh.simplify_quadric_decimation(**kwargs)
            except Exception as e:
                print(f"  ⚠ Quadric decimation failed ({type(e).__name__}), using fallback vertex subsampling")
                use_fallback = True
                reduced = fallback_simplify_by_vertex_subsampling(mesh, reduce_ratio)

    if reduced is None or reduced.vertices.shape[0] == 0 or reduced.faces.shape[0] == 0:
        raise RuntimeError("Mesh reduction produced empty mesh")

    xyz_out = np.asarray(reduced.vertices, dtype=np.float32)
    faces_out = np.asarray(reduced.faces, dtype=np.int64)
    edges_undirected = trimesh_edges_from_faces(faces_out)

    normals_out = np.asarray(reduced.vertex_normals, dtype=np.float32)
    if normals_out.shape != xyz_out.shape:
        raise RuntimeError(f"Unexpected reduced vertex_normals shape: {normals_out.shape} vs {xyz_out.shape}")

    # Transfer scalars from original vertices -> reduced vertices (nearest original vertex)
    tree = cKDTree(np.asarray(xyz_raw, dtype=np.float64))
    _, nn_idx = tree.query(np.asarray(xyz_out, dtype=np.float64), k=1)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    scalars_out = scalars_raw[nn_idx].astype(np.float32, copy=False)

    meta = {
        "applied": True,
        "method": "vertex_subsampling_fallback" if use_fallback else "trimesh_quadric_decimation",
        "requested_target_vertices": None if target_vertices is None else int(target_vertices),
        "requested_reduce_ratio": None if reduce_ratio is None else float(reduce_ratio),
        "target_faces_estimate": int(target_faces_est),
        "aggression": None if reduce_aggression is None else int(reduce_aggression),
        "orig_vertices": n0_v,
        "orig_faces": n0_f,
        "reduced_vertices": int(xyz_out.shape[0]),
        "reduced_faces": int(faces_out.shape[0]),
        "scalar_transfer": "nearest_original_vertex",
    }
    return xyz_out, normals_out, scalars_out, edges_undirected, meta


# ---------------------------
# Feature adjustment for SSL
# ---------------------------

def normalize_xyz_ssl(xyz_raw, eps=1e-8):
    """
    Per-graph centered + isotropic scaling to unit ball (max radius).
    """
    center = xyz_raw.mean(axis=0, keepdims=True).astype(np.float32)
    xyz_c = xyz_raw - center
    radii = np.linalg.norm(xyz_c, axis=1)
    scale = float(max(float(np.max(radii)), eps))
    xyz_ssl = (xyz_c / scale).astype(np.float32, copy=False)
    return xyz_ssl, center.squeeze(0), scale


def normalize_normals_unit(normals_raw, eps=1e-8):
    norms = np.linalg.norm(normals_raw, axis=1, keepdims=True)
    normals = normals_raw / np.maximum(norms, eps)
    return normals.astype(np.float32, copy=False)


def normalize_scalars(scalars_raw, scalar_stats=None, clip_z=None):
    """
    scalar_stats:
      {"means": [m_potential, m_normalpot], "stds": [s_potential, s_normalpot]}
    If None, returns raw scalars (copy).
    """
    scalars_raw = scalars_raw.astype(np.float32, copy=False)
    if scalar_stats is None:
        z = scalars_raw.copy()
    else:
        means = np.asarray(scalar_stats["means"], dtype=np.float32).reshape(1, 2)
        stds = np.asarray(scalar_stats["stds"], dtype=np.float32).reshape(1, 2)
        stds = np.maximum(stds, 1e-8)
        z = (scalars_raw - means) / stds

    if clip_z is not None:
        z = np.clip(z, -float(clip_z), float(clip_z))
    return z.astype(np.float32, copy=False)


# ---------------------------
# Graph assembly
# ---------------------------

def make_graph_from_vtk(
    filepath,
    potential_name="Potential",
    normalpot_name="NormalPotential",
    scalar_stats=None,
    scalar_clip_z=None,
    target_vertices=None,
    reduce_ratio=None,
    reduce_aggression=None,
):
    poly, xyz_raw_orig, normals_raw_orig, scalars_raw_orig = extract_points_features_from_vtk(
        filepath, potential_name=potential_name, normalpot_name=normalpot_name
    )

    # Optional reduction BEFORE graph creation (default = no reduction)
    reduction_result = maybe_reduce_mesh_with_trimesh(
        poly=poly,
        xyz_raw=xyz_raw_orig,
        scalars_raw=scalars_raw_orig,
        target_vertices=target_vertices,
        reduce_ratio=reduce_ratio,
        reduce_aggression=reduce_aggression,
    )

    if reduction_result is None:
        # No reduction: exact mesh topology edges from VTK mesh
        xyz_raw = xyz_raw_orig
        normals_raw = normals_raw_orig
        scalars_raw = scalars_raw_orig
        edges_undirected = extract_mesh_edges_exact(poly)
        reduction_meta = {
            "applied": False,
            "method": None,
            "requested_target_vertices": None,
            "requested_reduce_ratio": None,
            "aggression": None,
        }
    else:
        # Reduced mesh: topology edges come from reduced trimesh faces
        xyz_raw, normals_raw, scalars_raw, edges_undirected, reduction_meta = reduction_result

    # Keep exact topology indexing: fail instead of dropping invalid rows
    finite_mask = (
        np.isfinite(xyz_raw).all(axis=1)
        & np.isfinite(normals_raw).all(axis=1)
        & np.isfinite(scalars_raw).all(axis=1)
    )
    if not np.all(finite_mask):
        bad = int((~finite_mask).sum())
        raise RuntimeError(
            f"Found {bad} non-finite vertices/features after preprocessing. "
            "Refusing to drop rows because that would break vertex/edge indexing."
        )

    n = int(xyz_raw.shape[0])

    # Feature adjustment
    xyz_ssl, center, scale = normalize_xyz_ssl(xyz_raw)
    normals = normalize_normals_unit(normals_raw)
    scalars_norm = normalize_scalars(scalars_raw, scalar_stats=scalar_stats, clip_z=scalar_clip_z)

    # Node feature order for SSL:
    # [Potential_norm, NormalPotential_norm, nx, ny, nz, x_ssl, y_ssl, z_ssl]
    x = np.concatenate([scalars_norm, normals, xyz_ssl], axis=1).astype(np.float32, copy=False)

    # Exact mesh edges (unique undirected)
    if edges_undirected.size:
        edge_index_mesh = torch.from_numpy(edges_undirected.T.copy()).long()
        rev = edges_undirected[:, [1, 0]]
        bidir = np.concatenate([edges_undirected, rev], axis=0)  # [2E,2]
        edge_index = torch.from_numpy(bidir.T.copy()).long()      # [2,2E]
    else:
        edge_index_mesh = torch.empty((2, 0), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = {
        "x": torch.from_numpy(x).float(),
        "edge_index": edge_index,                    # bidirectional (PyG-friendly)
        "edge_index_mesh": edge_index_mesh,          # exact unique undirected mesh edges
        "pos": torch.from_numpy(xyz_ssl).float(),    # SSL geometry space
        "xyz_raw": torch.from_numpy(xyz_raw).float(),
        "normals": torch.from_numpy(normals).float(),        # unit normals
        "scalars_raw": torch.from_numpy(scalars_raw).float(),
        "scalars_norm": torch.from_numpy(scalars_norm).float(),
        "meta": {
            "source_file": str(filepath),
            "num_nodes": n,
            "num_mesh_edges_undirected": int(edges_undirected.shape[0]),
            "num_edges_bidir": int(edge_index.shape[1]),
            "feature_order_x": [
                "Potential_norm", "NormalPotential_norm", "nx", "ny", "nz", "x_ssl", "y_ssl", "z_ssl"
            ],
            "scalar_raw_order": ["Potential", "NormalPotential"],
            "xyz_ssl_norm": {
                "type": "center_then_unit_ball_max_radius",
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "scale": float(scale),
            },
            "scalars_norm": None if scalar_stats is None else {
                "type": "zscore",
                "means": [float(v) for v in scalar_stats["means"]],
                "stds": [float(v) for v in scalar_stats["stds"]],
                "clip_z": None if scalar_clip_z is None else float(scalar_clip_z),
            },
            "reduction": reduction_meta,
            "topology": "mesh_edges" if not reduction_meta["applied"] else "reduced_mesh_edges",
            "node_source": "mesh_vertices" if not reduction_meta["applied"] else "reduced_mesh_vertices",
            "no_subsampling": True,
            "no_knn": True,
        },
    }
    return graph


# ---------------------------
# Optional scalar stats (train split)
# ---------------------------

class RunningStats2D:
    def __init__(self, dim):
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.sum.shape[0]:
            raise ValueError(f"Expected [N,{self.sum.shape[0]}], got {x.shape}")
        self.n += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += (x * x).sum(axis=0)

    def finalize(self):
        if self.n == 0:
            raise RuntimeError("No samples seen")
        mean = self.sum / self.n
        var = np.maximum(self.sumsq / self.n - mean * mean, 0.0)
        std = np.sqrt(var)
        return {"means": mean.tolist(), "stds": std.tolist(), "count": int(self.n)}


def list_input_files(indir):
    p = Path(indir)
    return sorted([str(f) for f in p.iterdir() if f.suffix.lower() in (".vtk", ".vtp")])


def fit_scalar_stats(indir, potential_name="Potential", normalpot_name="NormalPotential", out_json=None):
    """
    Fit scalar z-score stats on ORIGINAL mesh vertices (recommended on TRAIN split only).
    Note: If you always reduce meshes before training, you may optionally fit stats on reduced graphs instead.
    """
    files = list_input_files(indir)
    if not files:
        raise RuntimeError(f"No .vtk/.vtp files found in {indir}")

    rs = RunningStats2D(dim=2)
    for fp in tqdm(files, total=len(files), desc="Fitting scalar stats"):
        _, _, _, scalars_raw = extract_points_features_from_vtk(
            fp, potential_name=potential_name, normalpot_name=normalpot_name
        )
        rs.update(scalars_raw)

    stats = rs.finalize()
    if out_json is not None:
        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved scalar stats -> {out_json}")
    print(json.dumps(stats, indent=2))
    return stats


# ---------------------------
# Parallel conversion
# ---------------------------

def _worker_convert(task):
    (
        filepath,
        outdir,
        potential_name,
        normalpot_name,
        scalar_stats,
        scalar_clip_z,
        target_vertices,
        reduce_ratio,
        reduce_aggression,
    ) = task
    try:
        graph = make_graph_from_vtk(
            filepath=filepath,
            potential_name=potential_name,
            normalpot_name=normalpot_name,
            scalar_stats=scalar_stats,
            scalar_clip_z=scalar_clip_z,
            target_vertices=target_vertices,
            reduce_ratio=reduce_ratio,
            reduce_aggression=reduce_aggression,
        )

        stem = Path(filepath).stem.replace(":", "_")
        out_path = Path(outdir) / f"{stem}_graph.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, str(out_path))

        return {"file": filepath, "ok": True, "error": None, "n": int(graph["meta"]["num_nodes"])}
    except Exception as e:
        return {
            "file": filepath,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }


def convert_dir_parallel(
    indir,
    outdir,
    potential_name="Potential",
    normalpot_name="NormalPotential",
    scalar_stats_json=None,
    scalar_clip_z=None,
    num_workers=None,
    maxtasksperchild=25,
    chunksize=1,
    target_vertices=None,
    reduce_ratio=None,
    reduce_aggression=None,
):
    files = list_input_files(indir)
    if not files:
        raise RuntimeError(f"No .vtk/.vtp files found in {indir}")

    if target_vertices is not None and reduce_ratio is not None:
        raise ValueError("Use only one of --target_vertices or --reduce_ratio")

    scalar_stats = None
    if scalar_stats_json:
        with open(scalar_stats_json, "r") as f:
            scalar_stats = json.load(f)
        if "means" not in scalar_stats or "stds" not in scalar_stats:
            raise ValueError("scalar_stats_json must contain keys: means, stds")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 1

    tasks = [
        (
            fp,
            str(outdir),
            potential_name,
            normalpot_name,
            scalar_stats,
            scalar_clip_z,
            target_vertices,
            reduce_ratio,
            reduce_aggression,
        )
        for fp in files
    ]

    print(f"Found {len(files)} files")
    print(f"Using {num_workers} worker(s)")
    if target_vertices is None and reduce_ratio is None:
        print("Mesh reduction: OFF (default)")
    elif target_vertices is not None:
        print(f"Mesh reduction: ON (target_vertices≈{target_vertices}, aggression={reduce_aggression})")
    else:
        print(f"Mesh reduction: ON (reduce_ratio={reduce_ratio}, aggression={reduce_aggression})")

    n_ok, n_fail = 0, 0
    failures = []

    if num_workers == 1:
        iterator = map(_worker_convert, tasks)
        for res in tqdm(iterator, total=len(tasks), desc="Converting"):
            if res["ok"]:
                n_ok += 1
            else:
                n_fail += 1
                failures.append(res)
    else:
        # Spawn is safer with native libs and cross-platform multiprocessing
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers, maxtasksperchild=maxtasksperchild) as pool:
            iterator = pool.imap_unordered(_worker_convert, tasks, chunksize=chunksize)
            for res in tqdm(iterator, total=len(tasks), desc="Converting"):
                if res["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append(res)

    print(f"Done. ok={n_ok}, fail={n_fail}, outdir={outdir}")

    if failures:
        print(f"\n{len(failures)} failures (showing up to 10):")
        for r in failures[:10]:
            print(f"  - {Path(r['file']).name}: {r['error']}")
            # Uncomment for short traces:
            # print(r.get('traceback', ''))


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert VTK meshes/datasets to SSL-ready graphs (vertices->nodes, mesh edges->edges)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("fit-stats", help="Fit scalar z-score stats on TRAIN split")
    p_stats.add_argument("--indir", required=True)
    p_stats.add_argument("--potential_name", default="Potential")
    p_stats.add_argument("--normalpot_name", default="NormalPotential")
    p_stats.add_argument("--out_json", required=True)

    p_conv = sub.add_parser("convert", help="Convert .vtk/.vtp files to graph .pt files")
    p_conv.add_argument("--indir", required=True)
    p_conv.add_argument("--outdir", required=True)
    p_conv.add_argument("--potential_name", default="Potential")
    p_conv.add_argument("--normalpot_name", default="NormalPotential")
    p_conv.add_argument("--scalar_stats_json", default=None, help="Optional train-split scalar stats JSON")
    p_conv.add_argument("--scalar_clip_z", type=float, default=None, help="Optional clip in z-space (e.g., 5)")
    p_conv.add_argument("-j", "--num_workers", type=int, default=None, help="Parallel workers (default: cpu_count)")
    p_conv.add_argument("--maxtasksperchild", type=int, default=25, help="Restart workers after N tasks")
    p_conv.add_argument("--chunksize", type=int, default=1, help="Pool imap_unordered chunksize")

    # Optional reduction BEFORE graph creation (default = no reduction)
    p_conv.add_argument("--target_vertices", type=int, default=None,
                        help="Optional best-effort target number of vertices BEFORE graph creation (default: no reduction)")
    p_conv.add_argument("--reduce_ratio", type=float, default=None,
                        help="Optional keep-ratio in (0,1] BEFORE graph creation (e.g. 0.25 keeps ~25%%; default: no reduction)")
    p_conv.add_argument("--reduce_aggression", type=int, default=None,
                        help="Optional trimesh decimation aggression (0..10, lower=better, slower)")

    args = parser.parse_args()

    if args.cmd == "fit-stats":
        fit_scalar_stats(
            indir=args.indir,
            potential_name=args.potential_name,
            normalpot_name=args.normalpot_name,
            out_json=args.out_json,
        )
    elif args.cmd == "convert":
        convert_dir_parallel(
            indir=args.indir,
            outdir=args.outdir,
            potential_name=args.potential_name,
            normalpot_name=args.normalpot_name,
            scalar_stats_json=args.scalar_stats_json,
            scalar_clip_z=args.scalar_clip_z,
            num_workers=args.num_workers,
            maxtasksperchild=args.maxtasksperchild,
            chunksize=args.chunksize,
            target_vertices=args.target_vertices,
            reduce_ratio=args.reduce_ratio,
            reduce_aggression=args.reduce_aggression,
        )
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()