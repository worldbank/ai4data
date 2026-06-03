"""
Create Label Studio annotation tasks from PDF files.

Converts each page of each PDF into a PNG image, creates a new
Label Studio project, and imports the tasks with local file storage
configured.

Usage::

    python create_tasks.py \\
        --project_name="My Project" \\
        --input_pdf_dir=pdf_input/ \\
        --dataset_name=dataset
"""

import argparse
import json
from pathlib import Path

from dsa.ls_helpers import (
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT,
    convert_pdfs_to_images,
    create_project,
    import_tasks_to_project,
)


def main(
    project_name: str,
    dataset_name: str,
    input_pdf_dir: str | Path,
) -> None:
    """Convert PDFs to page images and create a Label Studio project with tasks.

    Each PDF page is rendered at 300 DPI and saved as a PNG file.
    A ``tasks.json`` file is generated, a new Label Studio project is
    created, and the tasks are imported with local storage configured.

    Parameters
    ----------
    project_name : str
        Display name for the new Label Studio project.
    dataset_name : str
        Name for the output dataset directory under ``labelstudio_data/``.
    input_pdf_dir : str | Path
        Directory containing PDF files to process.
    """
    dataset_dir = Path(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) / dataset_name

    # Convert PDFs to page images
    pages_by_file = convert_pdfs_to_images(input_pdf_dir, dataset_name)

    # Build task JSON
    task_json: list[dict] = []
    for doc_id, image_list in pages_by_file.items():
        task_json.append(
            {
                "data": {"pages": image_list},
                "meta": {"file": doc_id},
            }
        )

    # Save task json file
    task_json_path = dataset_dir / "tasks.json"
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(task_json, f, indent=2)
    print(f"Wrote {len(task_json)} tasks to {task_json_path}")

    # Create project and import tasks
    project_id = create_project(project_name)
    import_tasks_to_project(project_id, task_json, dataset_name)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PDFs to page images and create Label Studio tasks."
    )
    parser.add_argument(
        "--project_name",
        required=True,
        help="Project name that will appear on Label Studio.",
    )
    parser.add_argument(
        "--dataset_name",
        default="dataset",
        help="Dataset name; defines the output directory (default: dataset).",
    )
    parser.add_argument(
        "--input_pdf_dir",
        default="pdf_input/",
        help="Path to the input directory containing PDFs (default: pdf_input/).",
    )
    args = parser.parse_args()
    main(args.project_name, args.dataset_name, args.input_pdf_dir)
