"""
Import path setup utility for interpolation test modules.
Use this to set up paths in Jupyter notebooks or interactive environments.
"""

import sys
from pathlib import Path


def setup_interpolation_paths():
    """
    Configure sys.path to find utils and interpolation modules.
    Call this at the start of your notebook or interactive session.
    
    Returns:
        tuple: (_tests_dir, submodule_root)
    """
    
    # Find _tests directory by searching filesystem
    cwd = Path.cwd()
    
    # Strategy 1: Check if _tests exists in current directory
    if (cwd / '_tests').exists():
        _tests_dir = cwd / '_tests'
        submodule_root = cwd
    # Strategy 2: Check if we're in _tests directory
    elif cwd.name == '_tests':
        _tests_dir = cwd
        submodule_root = cwd.parent
    # Strategy 3: Check parent
    elif (cwd.parent / '_tests').exists():
        _tests_dir = cwd.parent / '_tests'
        submodule_root = cwd.parent
    # Strategy 4: Search up the tree
    else:
        search_dir = cwd
        _tests_dir = None
        for _ in range(10):  # Search up to 10 levels
            if (search_dir / '_tests').exists():
                _tests_dir = search_dir / '_tests'
                submodule_root = search_dir
                break
            search_dir = search_dir.parent
        
        if _tests_dir is None:
            raise FileNotFoundError(
                f"Could not find '_tests' directory starting from {cwd}\n"
                f"Please ensure your working directory is set to the interpolation submodule root:\n"
                f"  import os; os.chdir('path/to/kit-forge_GeometrySets/src/interpolation')"
            )
    
    # Add paths in correct order
    tests_path = str(_tests_dir)
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    
    submodule_path = str(submodule_root)
    if submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)
    
    print(f"✓ Paths configured:")
    print(f"  - _tests:         {_tests_dir}")
    print(f"  - submodule root: {submodule_root}")
    print(f"  - sys.path[0]:    {sys.path[0]}")
    print(f"  - sys.path[1]:    {sys.path[1]}")
    
    return _tests_dir, submodule_root


# Auto-setup when this module is imported
_tests_root, _submodule_root = setup_interpolation_paths()
