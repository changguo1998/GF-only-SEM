"""Generate this case's regular 50×50×25 topology-only mesh."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "examples/halfspace-50x50x25/mesh_gen.py"), run_name="__main__")
