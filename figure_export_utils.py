"""Save matplotlib figures as PNG (submission default) plus high-DPI PDF for journals."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_figure_png_and_pdf(
    path_png: str | Path,
    *,
    dpi_png: int = 300,
    dpi_pdf: int = 400,
    **kwargs: Any,
) -> tuple[Path, Path]:
    """
    Save the current pyplot figure to PNG and to PDF.

    PNG uses dpi_png (default 300). PDF uses dpi_pdf (default 400) for any
    rasterized elements; vector content stays resolution-independent.
    """
    p = Path(path_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=dpi_png, **kwargs)
    pdf = p.with_suffix(".pdf")
    plt.savefig(pdf, dpi=dpi_pdf, format="pdf", **kwargs)
    return p, pdf
