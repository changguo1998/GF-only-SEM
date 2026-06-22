"""Shared fixtures for workflow integration tests."""

import os
import sys

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if os.path.join(_project_root, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "tools"))
