"""USD viewer launcher for interpolation test scripts."""

import subprocess
import os
import json
from pathlib import Path
from typing import Optional


def load_config() -> dict:
    """Load configuration from .config.json in interpolation submodule root."""
    config_path = Path(__file__).resolve().parents[2] / ".config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_clean_windows_env() -> dict:
    """Create a minimal clean Windows environment without Python/venv/conda variables."""
    return {
        'SystemRoot': os.environ.get('SystemRoot', 'C:\\Windows'),
        'SystemDrive': os.environ.get('SystemDrive', 'C:'),
        'ComSpec': os.environ.get('ComSpec', 'C:\\Windows\\System32\\cmd.exe'),
        'TEMP': os.environ.get('TEMP', 'C:\\Windows\\Temp'),
        'TMP': os.environ.get('TMP', 'C:\\Windows\\Temp'),
        'windir': os.environ.get('windir', 'C:\\Windows'),
        'PATH': os.pathsep.join([
            'C:\\Windows\\System32',
            'C:\\Windows',
            'C:\\Windows\\System32\\Wbem',
            'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\'
        ])
    }


def open_usd_viewer(
    stage_path: str,
    viewer_path: Optional[str] = None,
    wait_for_completion: bool = False,
    capture_output: bool = False
) -> subprocess.Popen:
    """
    Open a USD stage in the USD viewer.

    Args:
        stage_path: Path to the USD stage file to open
        viewer_path: Path to usdview.bat. If None, tries to locate it relative to this file.
        wait_for_completion: If True, wait for viewer to close and return output.
        capture_output: If True, capture stdout/stderr.

    Returns:
        subprocess.Popen: The process object for the viewer
    """
    stage_path = os.path.abspath(stage_path)

    if viewer_path is None:
        # Priority: config file > environment variable > relative paths
        config = load_config()
        if 'usdview_path' in config:
            viewer_path = config['usdview_path']
        else:
            env_viewer = os.environ.get('USDVIEW_PATH')
            if env_viewer:
                viewer_path = env_viewer
            else:
                # Try relative path that works when embedded in parent repo
                candidates = [
                    Path(__file__).resolve().parents[3] / "usdview" / "scripts" / "usdview.bat",
                    Path(__file__).resolve().parents[2] / "usdview" / "scripts" / "usdview.bat",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        viewer_path = str(candidate)
                        break

    if viewer_path is None:
        raise FileNotFoundError(
            "USD viewer not found. Set USDVIEW_PATH environment variable or pass viewer_path."
        )

    viewer_path = os.path.abspath(str(viewer_path))

    if not os.path.exists(viewer_path):
        raise FileNotFoundError(f"USD viewer not found at: {viewer_path}")
    if not os.path.exists(stage_path):
        raise FileNotFoundError(f"USD stage not found at: {stage_path}")

    clean_env = get_clean_windows_env()

    process = subprocess.Popen(
        [clean_env['ComSpec'], '/c', viewer_path, stage_path],
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        env=clean_env,
        cwd=os.path.dirname(viewer_path),
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    if wait_for_completion:
        stdout, stderr = process.communicate()
        if capture_output:
            print("=== STDOUT ===")
            print(stdout if stdout else "(empty)")
            print("\n=== STDERR ===")
            print(stderr if stderr else "(empty)")
            print(f"\n=== Return Code: {process.returncode} ===")

    return process
