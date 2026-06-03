import argparse
import logging
from tqdm.auto import tqdm

from dsa.constants import MODELS_DIR
from dsa.adapters.tfid import TFIDConfig, run_tfid_adapter_directory
from dsa.adapters.yolo11 import YOLO11Config, run_yolo11_adapter_directory
from dsa.adapters.yolo26 import YOLO26Config, run_yolo26_adapter_directory
from dsa.adapters.doclayoutyolo import (
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


def tfid_batch_runner(source, batch_start, batch_end):
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


def yolo11_runner(source, batch_start, batch_end):
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


def yolo26_runner(source, batch_start, batch_end):
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


def doclayoutyolo_runner(source, batch_start, batch_end):
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
    parser = argparse.ArgumentParser()
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
    )
    parser.add_argument(
        "--batch_end",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    doclayoutyolo_runner(args.source, args.batch_start, args.batch_end)
    yolo11_runner(args.source, args.batch_start, args.batch_end)
    yolo26_runner(args.source, args.batch_start, args.batch_end)
    tfid_batch_runner(args.source, args.batch_start, args.batch_end)
