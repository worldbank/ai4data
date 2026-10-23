"""Provide command-line interfaces for each pipeline stage."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai4data.data_snapshot.inputs import discover_snapshot_images
from ai4data.data_snapshot.metadata_extraction import extract_metadata_files
from ai4data.data_snapshot.pipeline import run_extraction, run_pipeline


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end pipeline from command-line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        Arguments to parse, excluding the executable name. Uses ``sys.argv``
        when omitted.

    Returns
    -------
    int
        Zero when the pipeline completes successfully, otherwise one.

    Raises
    ------
    SystemExit
        If arguments or local inputs are invalid.
    """
    parser = _snapshot_parser(
        "python -m ai4data.data_snapshot",
        "Extract data snapshots and Schema v1.4 metadata from PDFs.",
    )
    _add_metadata_provider_argument(parser)
    return _run_snapshot_command(parser, argv, end_to_end=True)


def snapshot_main(argv: list[str] | None = None) -> int:
    """Run data snapshot extraction from command-line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        Arguments to parse, excluding the executable name. Uses ``sys.argv``
        when omitted.

    Returns
    -------
    int
        Zero when extraction completes successfully, otherwise one.

    Raises
    ------
    SystemExit
        If arguments or local inputs are invalid.
    """
    parser = _snapshot_parser(
        "python -m ai4data.data_snapshot.extract_data_snapshot",
        "Extract data snapshots from PDFs without metadata extraction.",
    )
    return _run_snapshot_command(parser, argv, end_to_end=False)


def metadata_main(argv: list[str] | None = None) -> int:
    """Run metadata extraction from command-line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        Arguments to parse, excluding the executable name. Uses ``sys.argv``
        when omitted.

    Returns
    -------
    int
        Zero when extraction completes successfully, otherwise one.

    Raises
    ------
    SystemExit
        If arguments or local inputs are invalid.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ai4data.data_snapshot.extract_metadata",
        description="Extract Schema v1.4 metadata from snapshot images.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A snapshot image or directory containing snapshot images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Metadata output directory. By default, infer the metadata sibling "
            "of an input snapshots directory."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Replace existing metadata without modifying snapshots.",
    )
    _add_metadata_provider_argument(parser)
    args = parser.parse_args(argv)

    try:
        snapshot_paths = discover_snapshot_images(args.input)
        output_dir = args.output_dir or _infer_metadata_output_dir(args.input)
        paths = extract_metadata_files(
            snapshot_paths,
            output_dir,
            force=args.force,
            metadata_provider=args.metadata_provider,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in paths:
        print(path)
    return 0


def _snapshot_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Build the shared PDF extraction argument parser."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--model",
        default="doclayoutyolo",
        help="One extractor: doclayoutyolo (default), tfid, yolo11, or yolo26.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A PDF file or directory containing PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output root (default: output).",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=300,
        help="PDF rendering resolution (default: 300).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Delete matching crops and metadata before extraction.",
    )
    return parser


def _add_metadata_provider_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared metadata provider option to a parser."""
    parser.add_argument(
        "--metadata-provider",
        choices=("openrouter", "openai"),
        default="openrouter",
        help="Metadata API provider (default: openrouter).",
    )


def _run_snapshot_command(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    *,
    end_to_end: bool,
) -> int:
    """Run one PDF command and print its output paths."""
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        if end_to_end:
            paths = run_pipeline(
                args.input,
                args.output_dir,
                args.model,
                dpi=args.dpi,
                force=args.force,
                metadata_provider=args.metadata_provider,
            )
        else:
            paths = run_extraction(
                args.input,
                args.output_dir,
                args.model,
                dpi=args.dpi,
                force=args.force,
            )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in paths:
        print(path)
    return 0


def _infer_metadata_output_dir(input_path: Path) -> Path:
    """Infer a sibling metadata directory from a snapshots input path."""
    snapshots_dir = input_path if input_path.is_dir() else input_path.parent
    if snapshots_dir.name != "snapshots":
        raise ValueError(
            "--output-dir is required unless the input is in a snapshots directory."
        )
    return snapshots_dir.parent / "metadata"
