"""Make the separately packaged VsViewer modules available during repo tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
