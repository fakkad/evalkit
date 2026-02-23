"""Report generation -- JSON output and HTML report via Jinja2."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evalkit import __version__
from evalkit.models import SuiteResult

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def save_json(result: SuiteResult, output_path: str | Path) -> Path:
    """Save SuiteResult as JSON with full provenance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2))
    return output_path


def load_json(path: str | Path) -> SuiteResult:
    """Load SuiteResult from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return SuiteResult(**data)


def generate_html(result: SuiteResult, output_path: str | Path) -> Path:
    """Generate an HTML report from a SuiteResult."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    html = template.render(
        result=result,
        version=__version__,
    )
    output_path.write_text(html)
    return output_path
