import sys
from pathlib import Path

# Make repo-root packages (evaluation/, src/) importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: real-stack test needing the models/libraries; skipped when absent",
    )
