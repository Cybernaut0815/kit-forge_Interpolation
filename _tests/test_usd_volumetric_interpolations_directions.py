#%%

# =========================================
# Import necessary libraries
# =========================================

import sys
from pathlib import Path

# Add repository root to path so imports like "src.*" resolve correctly
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

import numpy as np

import src.geometry.IO as io
from src.viz.usdview import open_usd_viewer
from pxr import Usd, UsdGeom

#%%

# =========================================
# Testing Trilinear Interpolation Hexahedron with Tangent Vectors
# =========================================

from src.interpolation.volumetric.TrilinearInterpolationHexahedron import reverse_trilinear_interpolation_hexahedron_with_tangents

# Define a test hexahedron (8 vertices) in Y-up coordinate system
# Order: [v000, v100, v010, v110, v001, v101, v011, v111]
test_hex = np.array([
    [0.0, 0.0, 0.0],   # v000 (u=0, v=0, w=0)
    [1.0, 0.0, 0.0],   # v100 (u=1, v=0, w=0)
    [0.0, 1.0, 0.0],   # v010 (u=0, v=1, w=0)
    [1.2, 1.0, 0.0],   # v110 (u=1, v=1, w=0) - slightly skewed
    [0.0, 0.0, 1.0],   # v001 (u=0, v=0, w=1)
    [1.0, 0.0, 1.0],   # v101 (u=1, v=0, w=1)
    [0.0, 1.2, 1.0],   # v011 (u=0, v=1, w=1) - slightly skewed
    [1.0, 1.0, 1.0],   # v111 (u=1, v=1, w=1)
])

# Generate UVW grid for testing
grid_size = 5
u = np.linspace(0.1, 0.9, grid_size)
v = np.linspace(0.1, 0.9, grid_size)
w = np.linspace(0.1, 0.9, grid_size)
u_grid, v_grid, w_grid = np.meshgrid(u, v, w)
uvw_grid = np.column_stack([u_grid.ravel(), v_grid.ravel(), w_grid.ravel()])

print("=== Trilinear Interpolation Hexahedron with Tangents ===")
print(f"Generated UVW grid: {uvw_grid.shape}")
print(f"UVW range: [{np.min(uvw_grid):.3f}, {np.max(uvw_grid):.3f}]")

# Get Cartesian coordinates and tangent vectors
cartesian_coords, tangent_u, tangent_v, tangent_w = reverse_trilinear_interpolation_hexahedron_with_tangents(test_hex, uvw_grid)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Tangent U shape: {tangent_u.shape}")
print(f"Tangent V shape: {tangent_v.shape}")
print(f"Tangent W shape: {tangent_w.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Create hexahedron boundary edges (12 edges of a cube/hex)
hex_edges = [
    [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face (w=0)
    [4, 5], [5, 7], [7, 6], [6, 4],  # Top face (w=1)
    [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
]

# Create boundary lines
boundary_lines = np.array([
    [test_hex[e[0]], test_hex[e[1]]] for e in hex_edges
])

# Create arrow lines for tangents (scaled)
arrow_scale = 0.15
tangent_u_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_u[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
tangent_v_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_v[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
tangent_w_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_w[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])

# Add geometry under /geometry to inherit transform
stage = io.add_lines_to_usd_stage(stage, boundary_lines, '/geometry/HexBoundary', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, tangent_u_lines, '/geometry/TangentU', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, tangent_v_lines, '/geometry/TangentV', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, tangent_w_lines, '/geometry/TangentW', color=(0.0, 0.0, 1.0))  # Blue

# Save the stage
output_path = str(output_dir / "Quad_TestVault_hexahedron_tangents_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nVisualizing trilinear interpolation tangent vectors...")
print("Red: Hexahedron boundary & Tangent U vectors")
print("Green: Tangent V vectors")
print("Blue: Tangent W vectors")
open_usd_viewer(output_path)

#%%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Barycentric Interpolation Tetrahedron with Direction Vectors
# =========================================

from src.interpolation.volumetric.BarycentricInterpolationTetrahedron import reverse_barycentric_interpolation_tetrahedron_with_tangents

# Define a test tetrahedron (4 vertices)
test_tet = np.array([
    [0.0, 0.0, 0.0],      # v0 - bottom vertex
    [1.0, 0.0, 0.0],      # v1 - base right
    [0.5, 1.0, 0.0],      # v2 - base back
    [0.5, 0.3, 1.0],      # v3 - top vertex
])

# Generate barycentric coordinates grid inside tetrahedron
grid_size_tet = 6
alpha = np.linspace(0.1, 0.7, grid_size_tet)
beta = np.linspace(0.1, 0.7, grid_size_tet)
gamma = np.linspace(0.1, 0.7, grid_size_tet)

alpha_grid, beta_grid, gamma_grid = np.meshgrid(alpha, beta, gamma)
alpha_flat = alpha_grid.ravel()
beta_flat = beta_grid.ravel()
gamma_flat = gamma_grid.ravel()
delta_flat = 1.0 - alpha_flat - beta_flat - gamma_flat

# Filter out invalid barycentric coordinates
valid_mask = (delta_flat > 0.05) & (delta_flat < 0.95)
alpha_valid = alpha_flat[valid_mask]
beta_valid = beta_flat[valid_mask]
gamma_valid = gamma_flat[valid_mask]
delta_valid = delta_flat[valid_mask]

barycentric_coords = np.column_stack([alpha_valid, beta_valid, gamma_valid, delta_valid])

print("\n=== Barycentric Interpolation Tetrahedron with Direction Vectors ===")
print(f"Generated barycentric coordinates: {barycentric_coords.shape}")

# Get Cartesian coordinates and direction vectors
cartesian_coords, dir_v0, dir_v1, dir_v2, dir_v3 = reverse_barycentric_interpolation_tetrahedron_with_tangents(test_tet, barycentric_coords)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Direction to V0 shape: {dir_v0.shape}")
print(f"Direction to V1 shape: {dir_v1.shape}")
print(f"Direction to V2 shape: {dir_v2.shape}")
print(f"Direction to V3 shape: {dir_v3.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Create tetrahedron boundary edges (6 edges)
tet_edges = [
    [0, 1], [0, 2], [0, 3],  # Edges from v0
    [1, 2], [1, 3], [2, 3],  # Other edges
]

# Create boundary lines
tet_boundary_lines = np.array([
    [test_tet[e[0]], test_tet[e[1]]] for e in tet_edges
])

# Create arrow lines for direction vectors
arrow_scale = 0.15
dir_v0_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v0[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
dir_v1_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v1[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
dir_v2_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v2[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
dir_v3_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v3[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])

# Add geometry
stage = io.add_lines_to_usd_stage(stage, tet_boundary_lines, '/geometry/TetBoundary', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, dir_v0_lines, '/geometry/DirV0', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, dir_v1_lines, '/geometry/DirV1', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, dir_v2_lines, '/geometry/DirV2', color=(0.0, 0.0, 1.0))  # Blue
stage = io.add_lines_to_usd_stage(stage, dir_v3_lines, '/geometry/DirV3', color=(0.0, 1.0, 1.0))  # Cyan

# Save the stage
output_path = str(output_dir / "Tetra_TestVault_barycentric_directions_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nVisualizing barycentric interpolation tetrahedron direction vectors...")
print("Red: Tetrahedron boundary & Direction to vertex 0")
print("Green: Direction to vertex 1")
print("Blue: Direction to vertex 2")
print("Cyan: Direction to vertex 3")
open_usd_viewer(output_path)

#%%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Barycentric Linear Interpolation Trigonal (Triangular Prism) with Direction Vectors
# =========================================

from src.interpolation.volumetric.BarycentricLinearInterpolationTrigonal import reverse_barycentric_linear_interpolation_trigonal_with_tangents

# Define a test triangular prism (6 vertices)
# Order: [v0_bottom, v1_bottom, v2_bottom, v0_top, v1_top, v2_top]
test_prism = np.array([
    # Bottom triangle
    [0.0, 0.0, 0.0],      # v0_bottom
    [1.0, 0.0, 0.0],      # v1_bottom
    [0.5, 0.0, 0.866],    # v2_bottom (equilateral triangle in XZ plane at y=0)
    # Top triangle
    [0.1, 1.0, 0.1],      # v0_top (slightly offset)
    [1.1, 1.0, 0.0],      # v1_top (slightly offset)
    [0.6, 1.2, 0.966],    # v2_top (slightly offset)
])

# Generate parametric coordinates grid
grid_size_prism = 5
alpha_p = np.linspace(0.2, 0.6, grid_size_prism)
beta_p = np.linspace(0.2, 0.6, grid_size_prism)
t_p = np.linspace(0.2, 0.8, grid_size_prism)

alpha_grid_p, beta_grid_p, t_grid_p = np.meshgrid(alpha_p, beta_p, t_p)
alpha_flat_p = alpha_grid_p.ravel()
beta_flat_p = beta_grid_p.ravel()
t_flat_p = t_grid_p.ravel()
gamma_flat_p = 1.0 - alpha_flat_p - beta_flat_p

# Filter out invalid barycentric coordinates
valid_mask_p = gamma_flat_p > 0.1
alpha_valid_p = alpha_flat_p[valid_mask_p]
beta_valid_p = beta_flat_p[valid_mask_p]
gamma_valid_p = gamma_flat_p[valid_mask_p]
t_valid_p = t_flat_p[valid_mask_p]

parametric_coords = np.column_stack([alpha_valid_p, beta_valid_p, gamma_valid_p, t_valid_p])

print("\n=== Barycentric Linear Interpolation Trigonal (Triangular Prism) with Direction Vectors ===")
print(f"Generated parametric coordinates: {parametric_coords.shape}")

# Get Cartesian coordinates and direction vectors
cartesian_coords, dir_v0, dir_v1, dir_v2, tangent_t = reverse_barycentric_linear_interpolation_trigonal_with_tangents(test_prism, parametric_coords)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Direction to V0 shape: {dir_v0.shape}")
print(f"Direction to V1 shape: {dir_v1.shape}")
print(f"Direction to V2 shape: {dir_v2.shape}")
print(f"Tangent T shape: {tangent_t.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Create triangular prism boundary edges (9 edges)
prism_edges = [
    [0, 1], [1, 2], [2, 0],  # Bottom triangle
    [3, 4], [4, 5], [5, 3],  # Top triangle
    [0, 3], [1, 4], [2, 5],  # Vertical edges
]

# Create boundary lines
prism_boundary_lines = np.array([
    [test_prism[e[0]], test_prism[e[1]]] for e in prism_edges
])

# Create arrow lines for direction vectors
arrow_scale = 0.15
dir_v0_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v0[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
dir_v1_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v1[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
dir_v2_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + dir_v2[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
tangent_t_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_t[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])

# Add geometry
stage = io.add_lines_to_usd_stage(stage, prism_boundary_lines, '/geometry/PrismBoundary', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, dir_v0_lines, '/geometry/DirV0', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, dir_v1_lines, '/geometry/DirV1', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, dir_v2_lines, '/geometry/DirV2', color=(0.0, 0.0, 1.0))  # Blue
stage = io.add_lines_to_usd_stage(stage, tangent_t_lines, '/geometry/TangentT', color=(0.0, 1.0, 1.0))  # Cyan

# Save the stage
output_path = str(output_dir / "Tri_TestVault_prism_directions_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nVisualizing barycentric linear interpolation trigonal direction vectors...")
print("Red: Triangular Prism boundary & Direction to vertex 0")
print("Green: Direction to vertex 1")
print("Blue: Direction to vertex 2")
print("Cyan: Tangent along prism axis (t-direction)")
open_usd_viewer(output_path)

#%%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Trilinear Interpolation Pentahedron (Pyramid) with Tangent Vectors
# =========================================

from src.interpolation.volumetric.TrilinearInterpolationPentahedron import reverse_trilinear_interpolation_pentahedron_with_tangents

# Define a test pentahedron (5 vertices: 4 base + 1 apex)
# Order: [v00, v10, v01, v11, apex]
test_pyramid = np.array([
    # Base quadrilateral
    [0.0, 0.0, 0.0],   # v00 (u=0, v=0)
    [1.0, 0.0, 0.0],   # v10 (u=1, v=0)
    [0.0, 0.0, 1.0],   # v01 (u=0, v=1)
    [1.2, -0.25, 1.2],   # v11 (u=1, v=1)
    # Apex
    [0.5, 1.5, 0.5],   # apex (center above base)
])

# Generate UVW grid for testing
grid_size_pyr = 5
u_pyr = np.linspace(0.1, 0.9, grid_size_pyr)
v_pyr = np.linspace(0.1, 0.9, grid_size_pyr)
w_pyr = np.linspace(0.2, 0.8, grid_size_pyr)

u_grid_pyr, v_grid_pyr, w_grid_pyr = np.meshgrid(u_pyr, v_pyr, w_pyr)
uvw_grid_pyr = np.column_stack([u_grid_pyr.ravel(), v_grid_pyr.ravel(), w_grid_pyr.ravel()])

print("\n=== Trilinear Interpolation Pentahedron (Pyramid) with Tangents ===")
print(f"Generated UVW grid: {uvw_grid_pyr.shape}")

# Get Cartesian coordinates and tangent vectors
cartesian_coords, tangent_u, tangent_v, tangent_w = reverse_trilinear_interpolation_pentahedron_with_tangents(test_pyramid, uvw_grid_pyr)

print(f"Cartesian coordinates shape: {cartesian_coords.shape}")
print(f"Tangent U shape: {tangent_u.shape}")
print(f"Tangent V shape: {tangent_v.shape}")
print(f"Tangent W shape: {tangent_w.shape}")

# Create empty USD stage with Y-up
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# Create root group for organization
UsdGeom.Xform.Define(stage, '/geometry')

# Create pentahedron boundary edges (8 edges: 4 base + 4 apex)
pyramid_edges = [
    [0, 1], [1, 3], [3, 2], [2, 0],  # Base quadrilateral
    [0, 4], [1, 4], [2, 4], [3, 4],  # Edges to apex
]

# Create boundary lines
pyramid_boundary_lines = np.array([
    [test_pyramid[e[0]], test_pyramid[e[1]]] for e in pyramid_edges
])

# Create arrow lines for tangents
arrow_scale = 0.15
tangent_u_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_u[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
tangent_v_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_v[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])
tangent_w_lines = np.array([
    [cartesian_coords[i], cartesian_coords[i] + tangent_w[i] * arrow_scale] 
    for i in range(len(cartesian_coords))
])

# Add geometry
stage = io.add_lines_to_usd_stage(stage, pyramid_boundary_lines, '/geometry/PyramidBoundary', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, tangent_u_lines, '/geometry/TangentU', color=(1.0, 0.0, 0.0))  # Red
stage = io.add_lines_to_usd_stage(stage, tangent_v_lines, '/geometry/TangentV', color=(0.0, 1.0, 0.0))  # Green
stage = io.add_lines_to_usd_stage(stage, tangent_w_lines, '/geometry/TangentW', color=(0.0, 0.0, 1.0))  # Blue

# Save the stage
output_path = str(output_dir / "Penta_TestVault_pyramid_tangents_output.usd")
stage.Export(output_path)
print(f"\nSaved stage to: {output_path}")

# Open in USD viewer
print("\nVisualizing trilinear interpolation pentahedron tangent vectors...")
print("Red: Pentahedron boundary & Tangent U vectors")
print("Green: Tangent V vectors")
print("Blue: Tangent W vectors")
open_usd_viewer(output_path)

# %%
