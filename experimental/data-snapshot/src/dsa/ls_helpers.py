"""
Shared Label Studio SDK helper functions.

Provides common client initialization, project management, and data
preparation utilities used across Label Studio management scripts.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from label_studio_sdk import LabelStudio
from tqdm.auto import tqdm

from dsa.constants import ROOT
from dsa.utils import convert_pdf_to_images

load_dotenv()

API_KEY = os.getenv("LABELSTUDIO_API_KEY")
LS_BASE_URL = "http://localhost:8080"
BASE_HOST_PATH = "/data/local-files/?d="
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "labelstudio_data/"


if not API_KEY:
    raise ValueError(
        "LABELSTUDIO_API_KEY environment variable is not set. "
        "Please set it in your .env file."
    )


def get_client() -> LabelStudio:
    """Create and return an authenticated Label Studio SDK client.

    Returns
    -------
    LabelStudio
        An authenticated client instance.
    """
    return LabelStudio(base_url=LS_BASE_URL, api_key=API_KEY)


def create_project(project_name: str) -> int:
    """Create a new Label Studio project with the standard labeling config.

    Parameters
    ----------
    project_name : str
        Display name for the new project.

    Returns
    -------
    int
        The ID of the newly created project.
    """
    client = get_client()

    with open(ROOT / "labeling_interface_template.xml", "r", encoding="utf-8") as f:
        label_config = f.read()

    project = client.projects.create(title=project_name, label_config=label_config)
    print(f"Project created with ID: {project.id}")

    return project.id


def import_tasks_to_project(
    project_id: int,
    tasks: list[dict],
    dataset_name: str,
) -> None:
    """Import tasks into a project and configure local file storage.

    Parameters
    ----------
    project_id : int
        The Label Studio project ID to import tasks into.
    tasks : list[dict]
        List of task dicts to import. Each must contain at minimum
        a ``data`` key.
    dataset_name : str
        Name of the dataset directory for local storage configuration.
    """
    client = get_client()

    client.projects.import_tasks(id=project_id, request=tasks)

    path = f"/label-studio/data/{dataset_name}"
    client.import_storage.local.create(
        path=path, project=project_id, title=dataset_name
    )

    print("Successfully imported tasks and set up local storage.")


def convert_pdfs_to_images(
    input_pdf_dir: str | Path,
    dataset_name: str,
    backend: Literal["pymupdf", "pdf2image"] = "pymupdf",
) -> dict[str, list[str]]:
    """Convert PDF files to per-page PNG images for Label Studio.

    Each page is rendered at 300 DPI and saved as a PNG file under
    ``labelstudio_data/<dataset_name>/``. Returns a mapping from PDF
    filename to the list of Label Studio page-path strings.

    Parameters
    ----------
    input_pdf_dir : str | Path
        Directory containing PDF files to process.
    dataset_name : str
        Name of the dataset directory under ``labelstudio_data/``.
    backend : {"pymupdf", "pdf2image"}
        Library to use for PDF-to-image rendering.

    Returns
    -------
    dict[str, list[str]]
        Mapping of PDF filename → list of Label Studio page-path strings
        (e.g. ``"/data/local-files/?d=dataset/file.pdf_p000.png"``).
    """
    dataset_dir = Path(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    pages_by_file: dict[str, list[str]] = {}
    files = list(Path(input_pdf_dir).rglob("*.pdf"))

    for f in tqdm(files, desc="Converting PDFs to images"):
        images = convert_pdf_to_images(f, dpi=300, backend=backend)
        image_list: list[str] = []

        for idx, image in enumerate(images):
            fname = f"{f.name}_p{idx:03d}.png"
            image.save(dataset_dir / fname, "PNG")
            page = f"{BASE_HOST_PATH}{dataset_name}/{fname}"
            image_list.append(page)

        pages_by_file[f.name] = image_list

    return pages_by_file
