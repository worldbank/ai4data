"""
Import a previously exported Label Studio project into a new project.

Reads a JSON backup file produced by ``export_project.py``, converts the
source PDFs to page images, creates a new Label Studio project, and
imports all tasks with their existing annotations so work can continue.

Usage::

    python -m scripts.import_project \\
        --project_name="My Restored Project" \\
        --input_path=backups/project_22_backup.json \\
        --input_pdf_dir=pdf_input/
"""

import argparse
from pathlib import Path

from dsa.ls_helpers import (
    BASE_HOST_PATH,
    convert_pdfs_to_images,
    create_project,
    import_tasks_to_project,
)
from dsa.utils import load_json


def _infer_dataset_name(tasks: list[dict]) -> str:
    """Infer the dataset name from exported task page paths.

    Parses the first available page path to extract the dataset
    directory name used in Label Studio local file storage.

    Parameters
    ----------
    tasks : list[dict]
        List of exported task dicts, each containing
        ``data.pages`` with Label Studio page-path strings.

    Returns
    -------
    str
        The inferred dataset name.

    Raises
    ------
    ValueError
        If no page paths are found or the format is unexpected.
    """
    prefix = BASE_HOST_PATH
    for task in tasks:
        pages = task.get("data", {}).get("pages", [])
        if pages:
            page = pages[0]
            if prefix in page:
                remainder = page.split(prefix, 1)[1]
                dataset_name = remainder.split("/", 1)[0]
                return dataset_name

    raise ValueError(
        "Could not infer dataset name from exported tasks. "
        "Please provide --dataset_name explicitly."
    )


def import_project(
    project_name: str,
    input_path: str | Path,
    input_pdf_dir: str | Path,
    dataset_name: str | None = None,
) -> None:
    """Restore a Label Studio project from an exported JSON backup.

    Converts source PDFs to page images, creates a new Label Studio
    project, and imports all tasks with their annotations intact.

    Parameters
    ----------
    project_name : str
        Display name for the new Label Studio project.
    input_path : str | Path
        Path to the exported JSON file (output of ``export_project.py``).
    input_pdf_dir : str | Path
        Directory containing the source PDF files. These are converted
        to page images for Label Studio to display.
    dataset_name : str | None
        Name of the dataset directory under ``labelstudio_data/``.
        If ``None``, it is inferred from the exported task data.
    """
    tasks = load_json(input_path)
    print(f"Loaded {len(tasks)} tasks from {input_path}")

    # Resolve dataset name
    if dataset_name is None:
        dataset_name = _infer_dataset_name(tasks)
        print(f"Inferred dataset name: {dataset_name}")
    else:
        print(f"Using provided dataset name: {dataset_name}")

    # Convert PDFs to page images
    convert_pdfs_to_images(input_pdf_dir, dataset_name)

    # Create new project and import tasks
    project_id = create_project(project_name)
    import_tasks_to_project(project_id, tasks, dataset_name)

    print(f"Project '{project_name}' restored successfully (ID: {project_id}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Import a Label Studio project backup into a new project "
            "with existing annotations."
        )
    )
    parser.add_argument(
        "--project_name",
        required=True,
        help="Display name for the new Label Studio project.",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the exported JSON backup file.",
    )
    parser.add_argument(
        "--input_pdf_dir",
        required=True,
        help="Directory containing the source PDF files.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help=(
            "Dataset directory name under labelstudio_data/. "
            "If omitted, inferred from the exported task data."
        ),
    )
    args = parser.parse_args()
    import_project(
        args.project_name, args.input_path, args.input_pdf_dir, args.dataset_name
    )
