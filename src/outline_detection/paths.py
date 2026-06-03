"""Install-safe output locations.

Reports are written relative to the current working directory (where the
command is run from), not the installed package location.
"""

from pathlib import Path


def reports_dir():
    return Path.cwd() / "reports"


def evaluations_dir():
    return reports_dir() / "evaluations"


def analysis_dir():
    return reports_dir() / "analysis"


def diagnostics_dir():
    return reports_dir() / "diagnostics"


def models_dir():
    return reports_dir() / "models"


def ensure_report_dir(path):
    """Create parent directories for a report/output path and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
