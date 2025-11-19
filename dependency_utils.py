"""
Utilities to ensure optional Python dependencies are available at runtime.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Optional


def ensure_module(module_name: str, package_spec: Optional[str] = None):
    """
    Import `module_name`, installing it with pip if it is missing.

    Parameters
    ----------
    module_name: str
        The name passed to `importlib.import_module`.
    package_spec: Optional[str]
        Optional package specifier (e.g., "joblib==1.4.2"). When omitted, the
        module_name is used.

    Returns
    -------
    ModuleType
        The imported module.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

        package = package_spec or module_name
        print(f"[dependency_utils] '{module_name}' no encontrado. Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(module_name)

