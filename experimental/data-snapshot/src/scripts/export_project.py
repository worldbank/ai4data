"""
Export Label Studio project annotations to a JSON backup file.

Downloads all tasks (including unannotated ones) from a Label Studio
project and saves them to a local JSON file for backup or migration.

Usage::

    python -m scripts.export_project \\
        --project_id=22 \\
        --output_path=data/backups/project_22_backup.json
"""

import argparse
import json
from pathlib import Path

from dsa.ls_helpers import get_client


def export_project(project_id: int, output_path: str | Path) -> None:
    """Export all tasks and annotations from a Label Studio project.

    Uses the Label Studio SDK ``download_sync`` endpoint to retrieve
    every task in the project (annotated or not) and writes the result
    as a formatted JSON file.

    Parameters
    ----------
    project_id : int
        The Label Studio project ID to export.
    output_path : str | Path
        File path where the exported JSON will be saved. Parent
        directories are created automatically if they do not exist.
    """
    client = get_client()

    print(f"Exporting project {project_id}...")
    response = client.projects.exports.download_sync(
        id=project_id,
        download_all_tasks=True,
        export_type="JSON",
    )
    tasks = json.loads(b"".join(response).decode())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"Exported {len(tasks)} tasks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Label Studio project annotations to a JSON file."
    )
    parser.add_argument(
        "--project_id",
        type=int,
        required=True,
        help="Label Studio project ID to export.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the exported JSON file.",
    )
    args = parser.parse_args()
    export_project(args.project_id, args.output_path)
