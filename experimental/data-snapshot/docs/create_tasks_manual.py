"""
(Deprecated: Use with caution.) Create Label Studio annotation tasks from PDF files.

Converts each page of each PDF into a PNG image and generates a
``tasks.json`` file suitable for importing into Label Studio's
multi-page document annotation workflow.

Usage::

    python create_tasks_manual.py --input_dir=pdf_input/ --dataset_name=dataset
"""

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from data_snapshot.utils import convert_pdf_to_images

BASE_HOST_PATH = "/data/local-files/?d="
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "labelstudio_data/"


def main(input_dir: str | Path, dataset_name: str) -> None:
    """Convert PDFs to page images and create a Label Studio task file.

    Each PDF page is rendered at 300 DPI and saved as a PNG file.
    A ``tasks.json`` file is generated containing references to all
    pages, grouped by source PDF.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing PDF files to process.
    dataset_name : str
        Name for the output dataset directory under
        ``labelstudio_data/``.
    """
    output_dir = Path(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) / dataset_name
    task_json = []

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(Path(input_dir).rglob("*.pdf"))
    for f in tqdm(files):
        images = convert_pdf_to_images(f, dpi=300)

        image_list = []
        for idx, image in enumerate(images):
            # Save png file
            fname = f"{f.name}_p{idx:03d}.png"
            image.save(Path(output_dir) / fname, "PNG")

            # Compile to task json
            page = f"{BASE_HOST_PATH}{dataset_name}/{fname}"
            image_list.append(page)

        task_json.append(
            {
                "data": {"pages": image_list},
                "meta": {"file": f.name},
            }
        )

    with open(output_dir / "tasks.json", "w") as f:
        json.dump(task_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PDFs to page images and create Label Studio tasks."
    )
    parser.add_argument(
        "--input_dir",
        default="pdf_input/",
        help="Path to the input directory (default: pdf_input/)",
    )
    parser.add_argument(
        "--dataset_name",
        default="dataset",
        help="Dataset name; defines the output directory (default: dataset)",
    )
    args = parser.parse_args()
    main(args.input_dir, args.dataset_name)
