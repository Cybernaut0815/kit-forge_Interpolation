#%%

# =========================================
# Import necessary libraries
# =========================================

import sys
from pathlib import Path

# Add _tests and submodule root to path so local utils and interpolation imports resolve
_tests_root = str(Path(__file__).resolve().parent)
_submodule_root = str(Path(__file__).resolve().parent.parent)
for _path in (_tests_root, _submodule_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

import numpy as np

import helper
import utils.io_utils as io
from utils.usd_viewer import open_usd_viewer
from pxr import Usd, UsdGeom

#%%

# =========================================
# Testing Bilinear Interpolation with Tangent Vectors
# =========================================

from planar.BilinearInterpolationQuad import reverse_bilinear_interpolation_quad_with_tangents

# Define a test quadrilateral in XZ plane (Y-up coordinate system)
test_quad = np.array([
    [-0.5, -0.5],  # bottom-left
    [0.5, -0.5],   # bottom-right
    [-0.7, 0.7],   # top-left (skewed)
    [0.5, 0.5]     # top-right
])

# Generate UV grid for testing
grid_size = 8
uv_grid = helper.generate_uv_grid(grid_size, uv_min=0.0, uv_max=1.0)

print("=== Bilinear Interpolation with Tangents ===")
print(f"Generated UV grid: {uv_grid.shape}")
print(f"UV range: [{np.min(uv_grid):.3f}, {np.max(uv_grid):.3f}]")

# Get Cartesian coordinates and tangent vectors
cartesian_coords, tangent_u, tangent_v = reverse_bilinear_interpolation_quad_with_tangents(test_quad, uv_grid)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Tangent U shape: {tangent_u.shape}")
print(f"Tangent V shape: {tangent_v.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Convert 2D XZ coordinates to 3D (Y=0 for ground plane)
# Format: [X, Y=0, Z]
points_3d = np.column_stack([cartesian_coords[:,0], np.zeros(len(cartesian_coords)), cartesian_coords[:,1]])
tangent_u_3d = np.column_stack([tangent_u[:,0], np.zeros(len(tangent_u)), tangent_u[:,1]])
tangent_v_3d = np.column_stack([tangent_v[:,0], np.zeros(len(tangent_v)), tangent_v[:,1]])
test_quad_3d = np.column_stack([test_quad[:,0], np.zeros(len(test_quad)), test_quad[:,1]])

# Define quad edges
quad_edges = [[0, 1], [1, 3], [3, 2], [2, 0]]  # v0->v1, v1->v3, v3->v2, v2->v0

# Create boundary lines
boundary_lines = np.array([
    [test_quad_3d[e[0]], test_quad_3d[e[1]]] for e in quad_edges
])

# Create arrow lines for tangents (scaled)
arrow_scale = 0.1
tangent_u_lines = np.array([
    [points_3d[i], points_3d[i] + tangent_u_3d[i] * arrow_scale] 
    for i in range(len(points_3d))
])
tangent_v_lines = np.array([
    [points_3d[i], points_3d[i] + tangent_v_3d[i] * arrow_scale] 
    for i in range(len(points_3d))
])

# Add all geometry under /geometry to inherit transform
stage = io.add_lines_to_usd_stage(stage, boundary_lines, '/geometry/QuadBoundary', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, tangent_u_lines, '/geometry/TangentU', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, tangent_v_lines, '/geometry/TangentV', color=(0.0, 0.0, 1.0))  # Blue

# Save the stage
output_path = str(output_dir / "Quad_TestVault_bilinear_tangents_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
print("Green: Quad boundary")
print("Red: Tangent U vectors")
print("Blue: Tangent V vectors")
open_usd_viewer(output_path)

#%%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Projective Interpolation with Tangent Vectors
# =========================================

from planar.ProjectiveInterpolationQuad import reverse_projective_interpolation_quad_with_tangents

print("\n=== Projective Interpolation with Tangents ===")

# Get Cartesian coordinates and tangent vectors
cartesian_coords, tangent_u, tangent_v = reverse_projective_interpolation_quad_with_tangents(test_quad, uv_grid)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Tangent U shape: {tangent_u.shape}")
print(f"Tangent V shape: {tangent_v.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Convert 2D to 3D coordinates
points_3d = np.column_stack([cartesian_coords[:,0], np.zeros(len(cartesian_coords)), cartesian_coords[:,1]])
tangent_u_3d = np.column_stack([tangent_u[:,0], np.zeros(len(tangent_u)), tangent_u[:,1]])
tangent_v_3d = np.column_stack([tangent_v[:,0], np.zeros(len(tangent_v)), tangent_v[:,1]])

# Create boundary lines (reuse test_quad_3d from bilinear section)
boundary_lines = np.array([
    [test_quad_3d[e[0]], test_quad_3d[e[1]]] for e in quad_edges
])

# Create arrow lines for tangents
tangent_u_lines = np.array([
    [points_3d[i], points_3d[i] + tangent_u_3d[i] * arrow_scale] 
    for i in range(len(points_3d))
])
tangent_v_lines = np.array([
    [points_3d[i], points_3d[i] + tangent_v_3d[i] * arrow_scale] 
    for i in range(len(points_3d))
])

# Add geometry
stage = io.add_lines_to_usd_stage(stage, boundary_lines, '/geometry/QuadBoundary', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, tangent_u_lines, '/geometry/TangentU', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, tangent_v_lines, '/geometry/TangentV', color=(0.0, 0.0, 1.0))  # Blue

# Save the stage
output_path = str(output_dir / "Quad_TestVault_projective_tangents_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
print("Green: Quad boundary")
print("Red: Tangent U vectors")
print("Blue: Tangent V vectors")
open_usd_viewer(output_path)

#%%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Barycentric Interpolation with Direction Vectors
# =========================================

from planar.BarycentricInterpolationTri import reverse_barycentric_interpolation_tri_with_tangents

# Define a test triangle in XZ plane
test_tri = np.array([
    [0, -0.57735],      # bottom vertex
    [-0.5, 0.288663],   # top-left vertex
    [0.5, 0.288663]     # top-right vertex
])

# Generate barycentric coordinates grid
grid_size_tri = 8
alpha = np.linspace(0.1, 0.8, grid_size_tri)
beta = np.linspace(0.1, 0.8, grid_size_tri)
alpha_grid, beta_grid = np.meshgrid(alpha, beta)

# Flatten and compute gamma
alpha_flat = alpha_grid.ravel()
beta_flat = beta_grid.ravel()
gamma_flat = 1.0 - alpha_flat - beta_flat

# Filter out invalid barycentric coordinates
valid_mask = gamma_flat > 0.0
alpha_valid = alpha_flat[valid_mask]
beta_valid = beta_flat[valid_mask]
gamma_valid = gamma_flat[valid_mask]

barycentric_coords = np.column_stack([alpha_valid, beta_valid, gamma_valid])

print("\n=== Barycentric Interpolation with Direction Vectors ===")
print(f"Generated barycentric coordinates: {barycentric_coords.shape}")

# Get Cartesian coordinates and direction vectors
cartesian_coords, dir_v0, dir_v1, dir_v2 = reverse_barycentric_interpolation_tri_with_tangents(test_tri, barycentric_coords)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Direction to V0 shape: {dir_v0.shape}")
print(f"Direction to V1 shape: {dir_v1.shape}")
print(f"Direction to V2 shape: {dir_v2.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Convert 2D to 3D coordinates
points_3d = np.column_stack([cartesian_coords[:,0], np.zeros(len(cartesian_coords)), cartesian_coords[:,1]])
dir_v0_3d = np.column_stack([dir_v0[:,0], np.zeros(len(dir_v0)), dir_v0[:,1]])
dir_v1_3d = np.column_stack([dir_v1[:,0], np.zeros(len(dir_v1)), dir_v1[:,1]])
dir_v2_3d = np.column_stack([dir_v2[:,0], np.zeros(len(dir_v2)), dir_v2[:,1]])
test_tri_3d = np.column_stack([test_tri[:,0], np.zeros(len(test_tri)), test_tri[:,1]])

# Define triangle edges
tri_edges = [[0, 1], [1, 2], [2, 0]]

# Create boundary lines
tri_boundary_lines = np.array([
    [test_tri_3d[e[0]], test_tri_3d[e[1]]] for e in tri_edges
])

# Create arrow lines for direction vectors
arrow_scale_tri = 0.1
dir_v0_lines = np.array([
    [points_3d[i], points_3d[i] + dir_v0_3d[i] * arrow_scale_tri] 
    for i in range(len(points_3d))
])
dir_v1_lines = np.array([
    [points_3d[i], points_3d[i] + dir_v1_3d[i] * arrow_scale_tri] 
    for i in range(len(points_3d))
])
dir_v2_lines = np.array([
    [points_3d[i], points_3d[i] + dir_v2_3d[i] * arrow_scale_tri] 
    for i in range(len(points_3d))
])

# Add geometry
stage = io.add_lines_to_usd_stage(stage, tri_boundary_lines, '/geometry/TriBoundary', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, dir_v0_lines, '/geometry/DirV0', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, dir_v1_lines, '/geometry/DirV1', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, dir_v2_lines, '/geometry/DirV2', color=(0.0, 0.0, 1.0))  # Blue

# Save the stage
output_path = str(output_dir / "Tri_TestVault_barycentric_directions_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nVisualizing barycentric interpolation direction vectors...")
print("Red: Triangle boundary & Direction to vertex 0")
print("Green: Direction to vertex 1")
print("Blue: Direction to vertex 2")
open_usd_viewer(output_path)

#%%
