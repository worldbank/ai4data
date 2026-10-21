"""Build deterministic data snapshot output names."""

from pathlib import Path


def build_snapshot_stem(source_pdf: str | Path, label: str, index: int) -> str:
    """Build a zero-based snapshot stem from its source document and label.

    Parameters
    ----------
    source_pdf : str | Path
        Source PDF path whose filename stem identifies the document.
    label : str
        Snapshot label, either ``"figure"`` or ``"table"``.
    index : int
        Zero-based per-label snapshot index for the source document.

    Returns
    -------
    str
        Stem in ``{source}_{figure|table}_{index:03d}`` form.

    Raises
    ------
    ValueError
        If the label is unsupported or the index is negative.
    """
    normalized_label = label.lower()
    if normalized_label not in {"figure", "table"}:
        raise ValueError(f"Unsupported snapshot label: {label}")
    if index < 0:
        raise ValueError("Snapshot index must be zero or greater.")
    return f"{Path(source_pdf).stem}_{normalized_label}_{index:03d}"
