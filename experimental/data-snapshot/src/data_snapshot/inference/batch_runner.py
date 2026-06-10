"""Run all layout detection adapters in batch over partitioned PDF directories.

Iterates over numbered batch directories (e.g. ``pdf_input/unhcr_batch1/``)
and runs each supported adapter, writing per-batch prediction JSON files
to ``data/batch_runs/``.

Usage::

    uv run python -m data_snapshot.inference.batch_runner \\
        --source unhcr --batch_start 1 --batch_end 5
"""

import argparse
import logging

from tqdm.auto import tqdm

from data_snapshot.constants import MODELS_DIR
from data_snapshot.inference.tfid import TFIDConfig, run_tfid_adapter_directory
from data_snapshot.inference.yolo11 import YOLO11Config, run_yolo11_adapter_directory
from data_snapshot.inference.yolo26 import YOLO26Config, run_yolo26_adapter_directory
from data_snapshot.inference.doclayoutyolo import (
    DocLayoutYOLOConfig,
    run_doclayout_yolo_adapter_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="batch_runner.log",
    filemode="a",
)

OUTPUT_DIR = "data/batch_runs/"


def tfid_batch_runner(source: str, batch_start: int, batch_end: int) -> None:
    """Run the TF-ID adapter across a range of batch directories.

    Parameters
    ----------
    source : str
        Source corpus name (e.g. ``"unhcr"``, ``"prwp"``, ``"refugee"``).
    batch_start : int
        First batch number (inclusive).
    batch_end : int
        Last batch number (inclusive).
    """
    logging.info("Start TFID runner.")
    for b in tqdm(range(batch_start, batch_end + 1), desc="Processing batches"):
        path_dir = f"pdf_input/{source}_batch{b}"
        out_path = OUTPUT_DIR + f"tfid-large_{source}_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = TFIDConfig(
            model_id="yifeihu/TF-ID-large",
            device="cpu",
            dpi=300,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_tfid_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


def yolo11_runner(source: str, batch_start: int, batch_end: int) -> None:
    """Run the YOLO11 adapter across a range of batch directories.

    Parameters
    ----------
    source : str
        Source corpus name (e.g. ``"unhcr"``, ``"prwp"``, ``"refugee"``).
    batch_start : int
        First batch number (inclusive).
    batch_end : int
        Last batch number (inclusive).
    """
    logging.info("Start yolo11 runner.")
    for b in tqdm(range(batch_start, batch_end + 1), desc="Processing batches"):
        path_dir = f"pdf_input/{source}_batch{b}"
        out_path = OUTPUT_DIR + f"yolo11_{source}_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = YOLO11Config(
            repo_id="Armaggheddon/yolo11-document-layout",
            filename="yolo11m_doc_layout.pt",
            device="cpu",
            dpi=300,
            conf=0.25,
            iou=0.7,
            imgsz=1280,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_yolo11_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


def yolo26_runner(source: str, batch_start: int, batch_end: int) -> None:
    """Run the YOLO26 adapter across a range of batch directories.

    Parameters
    ----------
    source : str
        Source corpus name (e.g. ``"unhcr"``, ``"prwp"``, ``"refugee"``).
    batch_start : int
        First batch number (inclusive).
    batch_end : int
        Last batch number (inclusive).
    """
    logging.info("Start yolo26 runner.")
    for b in tqdm(range(batch_start, batch_end + 1), desc="Processing batches"):
        path_dir = f"pdf_input/{source}_batch{b}"
        out_path = OUTPUT_DIR + f"yolo26_{source}_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = YOLO26Config(
            repo_id="Armaggheddon/yolo26-document-layout",
            filename="yolo26m_doc_layout.pt",
            device="cpu",
            dpi=300,
            conf=0.25,
            iou=0.7,
            imgsz=1280,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_yolo26_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


def doclayoutyolo_runner(source: str, batch_start: int, batch_end: int) -> None:
    """Run the DocLayout-YOLO adapter across a range of batch directories.

    Parameters
    ----------
    source : str
        Source corpus name (e.g. ``"unhcr"``, ``"prwp"``, ``"refugee"``).
    batch_start : int
        First batch number (inclusive).
    batch_end : int
        Last batch number (inclusive).
    """
    logging.info("Start doclayoutyolo runner.")
    for b in tqdm(range(batch_start, batch_end + 1), desc="Processing batches"):
        path_dir = f"pdf_input/{source}_batch{b}"
        out_path = OUTPUT_DIR + f"doclayoutyolo_{source}_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = DocLayoutYOLOConfig(
            model_path=str(MODELS_DIR / "doclayout_yolo_docstructbench_imgsz1024.pt"),
            device="cpu",
            dpi=300,
            conf=0.2,
            imgsz=1024,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_doclayout_yolo_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run all layout detection adapters in batch over "
            "partitioned PDF directories."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["unhcr", "prwp", "refugee"],
        required=True,
        help="Source name (unhcr, prwp, refugee)",
    )
    parser.add_argument(
        "--batch_start",
        type=int,
        required=True,
        help="First batch number (inclusive).",
    )
    parser.add_argument(
        "--batch_end",
        type=int,
        required=True,
        help="Last batch number (inclusive).",
    )
    args = parser.parse_args()

    doclayoutyolo_runner(args.source, args.batch_start, args.batch_end)
    yolo11_runner(args.source, args.batch_start, args.batch_end)
    yolo26_runner(args.source, args.batch_start, args.batch_end)
    tfid_batch_runner(args.source, args.batch_start, args.batch_end)
