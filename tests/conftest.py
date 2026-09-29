import sys
from pathlib import Path

# Make repo-root packages (evaluation/, src/) importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
