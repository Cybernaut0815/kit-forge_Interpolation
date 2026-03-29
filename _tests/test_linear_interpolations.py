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

import numpy as np
from importlib import reload

# %%

# =========================================
# Import linear interpolation functions
# =========================================

import linear.LinearInterpolationLine as linline
reload(linline)
from linear.LinearInterpolationLine import linear_interpolation_line, reverse_linear_interpolation_line, reverse_linear_interpolation_line_with_tangent

# Define a 3D line segment
line = np.array([
    [0.0, 0.0, 0.0],  # start point
    [1.0, 2.0, 3.0]   # end point
])

# =========================================
# Test 1: Forward interpolation (Cartesian to t)
# =========================================

# Test 1: Forward interpolation (Cartesian to t)
print("=== Test 1: Forward Interpolation (Cartesian to t) ===")
test_points = np.array([
    [0.0, 0.0, 0.0],    # Should give t=0
    [1.0, 2.0, 3.0],    # Should give t=1
    [0.5, 1.0, 1.5],    # Should give t=0.5
    [0.25, 0.5, 0.75],  # Should give t=0.25
])

t_values = linear_interpolation_line(line, test_points)
print(f"Test points:\n{test_points}")
print(f"Computed t values:\n{t_values}")

# =========================================
# Test 2: Reverse interpolation (t to Cartesian)
# =========================================

# Test 2: Reverse interpolation (t to Cartesian)
print("\n=== Test 2: Reverse Interpolation (t to Cartesian) ===")
test_t = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
reconstructed_points = reverse_linear_interpolation_line(line, test_t)
print(f"Input t values:\n{test_t}")
print(f"Reconstructed points:\n{reconstructed_points}")

# =========================================
# Test 3: Reverse interpolation with tangent
# =========================================

# Test 3: Reverse interpolation with tangent
print("\n=== Test 3: Reverse Interpolation with Tangent ===")
cartesian_coords, tangent_vectors = reverse_linear_interpolation_line_with_tangent(line, test_t)
print(f"Cartesian coordinates:\n{cartesian_coords}")
print(f"Tangent vectors (constant along line):\n{tangent_vectors}")
print(f"Tangent magnitude: {np.linalg.norm(tangent_vectors[0]):.6f}")

# =========================================
# Test 4: Round-trip test
# =========================================

print("\n=== Test 4: Round-trip Test ===")
original_points = np.array([
    [0.2, 0.4, 0.6],
    [0.7, 1.4, 2.1],
    [0.9, 1.8, 2.7]
])

# Forward: Cartesian -> t
t_computed = linear_interpolation_line(line, original_points)
# Reverse: t -> Cartesian
reconstructed = reverse_linear_interpolation_line(line, t_computed)

print(f"Original points:\n{original_points}")
print(f"Computed t:\n{t_computed}")
print(f"Reconstructed points:\n{reconstructed}")
print(f"Max reconstruction error: {np.max(np.abs(original_points - reconstructed)):.2e}")

# =========================================
# Test 5: Test with points outside [0,1] range
# =========================================

# Test 5: Test with points outside [0,1] range
print("\n=== Test 5: Points Outside [0,1] Range ===")
extended_t = np.array([[-0.5], [1.5], [2.0]])
extended_points = reverse_linear_interpolation_line(line, extended_t)
print(f"Extended t values:\n{extended_t}")
print(f"Points on line extension:\n{extended_points}")

# %%

