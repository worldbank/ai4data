# data-snapshot

Data snapshot toolkit for PDF documents.

This package provides the source code for [Benchmarking Open-Source Layout Detection Models for Data Snapshot Extraction from Institutional Documents](https://arxiv.org/abs/2606.06242).

Dataset: [ai4data/data-snapshot](https://huggingface.co/datasets/ai4data/data-snapshot)

## Package structure

```
src/data_snapshot/
├── annotation/       # Label Studio annotation project management
├── inference/        # Layout detection model adapters
├── evaluation/       # Evaluation framework
├── metadata/         # Metadata tools
├── misc/             # Other tools
├── constants.py      # Shared constants
└── utils.py          # Shared utilities
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```shell
# Core package
uv sync

# With model-specific dependencies
uv sync --extra tfid             # TF-ID adapter
uv sync --extra doclayout_yolo   # DocLayout-YOLO adapter
uv sync --extra yolo11           # YOLO11 adapter
uv sync --extra yolo26           # YOLO26 adapter
uv sync --extra viz              # Visualization

# All dependencies (development)
uv sync --all-extras
```

### Environment variables

Copy `.env.template` to `.env` and fill in the required values:
```
LABELSTUDIO_API_KEY=    # Required for annotation workflows (See `Add API key` section)
```

---

# Annotation

Annotation workflows for creating ground truth datasets on Label Studio.

## First time setup

1. Clone the repository.
    ```shell
    git clone git@github.com:ajdajd/data-snapshot-annotation.git
    ```
2. Go inside the repo and start the service.
    ```shell
    cd data-snapshot-annotation
    docker compose up
    ```
3. If you encounter permission issues, run
    ```shell
    # Note: Make sure your cwd is the project directory!
    sudo chmod -R 777 .
    ```
    then retry.
    ```shell
    docker compose up
    ```
4. Open `http://localhost:8080/` on a web browser and create a login.

## Add API key

1. Open Label Studio.
2. At the top-left corner, click the hamburger menu (≡) and select `Organization`.
3. At the top-right corner, click `API Tokens Settings`.
4. Enable `Legacy Tokens` and click `Save Changes`.
5. At the top-right corner, click the portrait and select `Account & Settings`.
6. At the left menu, click `Legacy Token`.
7. Copy the token and add it to the `.env` file.

## Setting up a new PDF annotation project

1. Add PDF files to annotate in the `pdf_input` directory.
2. Start Label Studio.
    ```shell
    docker compose up
    ```
3. Run `create_tasks.py`.
    ```shell
    uv run python -m data_snapshot.annotation.create_tasks \
    --project_name="My annotation project" \
    --dataset_name=my_dataset \
    --input_pdf_dir=pdf_input/
    ```
4. Open Label Studio and refresh the web browser. The newly created project should appear.

## Setting up a new PDF annotation project with pre-labeling

1. Add PDF files to annotate in the `pdf_input` directory.
2. Generate prediction file(s). See [Inference](#inference) for the list of supported models and installation info.
    ```shell
    uv run python -m data_snapshot.inference.{adapter} \
    --input_pdf_dir=pdf_input/ \
    --output_json_path=data/evaluation_input/preds.json
    ```
3. (Optional) Combine prediction files by assigning a class to a source.
    ```shell
    uv run python -m data_snapshot.annotation.merge_class_preds \
    --figure_preds=data/evaluation_input/preds1.json \
    --table_preds=data/evaluation_input/preds2.json \
    --output_json_path=data/evaluation_input/combined_preds.json  
    ```
4. Start Label Studio.
    ```shell
    docker compose up
    ```
5. Create project and tasks.
    ```shell
    uv run python -m data_snapshot.annotation.create_tasks_with_prelabeling \
    --project_name="My project with prelabeling" \
    --dataset_name=my_dataset \
    --input_pdf_dir=pdf_input/ \
    --pred_json_path=data/evaluation_input/preds.json
    ```
5. Open Label Studio and refresh the web browser. The newly created project should appear.

## Backing up an annotation project

1. Make sure Label Studio is started.
    ```shell
    docker compose up
    ```
2. Before exporting, make sure all annotations are submitted and that there are no drafts. You check this by going to the project page and filter for tasks where drafts exist.
3. Run `export_project.py`. This will create the backup JSON file in the specified path.
    ```shell
    uv run python -m data_snapshot.annotation.export_project \
    --project_id=22 \
    --output_path=data/backups/project_22_backup.json
    ```

## Restoring an annotation project

1. Prepare the following files:
    1. Backup JSON file (e.g., `backups/project_22_backup.json`)
    2. PDF files (e.g., in `pdf_input/`)
2. Make sure Label Studio is started and an [API key is added to the `.env` file](#add-api-key).
    ```shell
    docker compose up
    ```
3. Run `import_project.py`.
    ```shell
    uv run python -m data_snapshot.annotation.import_project \
    --project_name="My restored project" \
    --input_path=backups/project_22_backup.json \
    --input_pdf_dir=pdf_input/
    ```
    Note: `dataset_name` is an optional parameter that must match the value in the backup file. If not specified, the script will infer it from the backup file.

    If value is not correct, the images will not load properly and the source storage must be manually configured. See [docs/manual_setup.md](docs/manual_setup.md) for instructions.

---

# Inference

Layout detection model adapters that perform the following:
1. Run Figure and Table detection on a directory of PDF files.
2. Convert the raw model outputs into the [Unified Evaluation Schema v1.3 format](docs/data-snapshot-eval-v1.3.schema.json).

For unsupported models, a small adapter module must be written for them for integration into this repository's framework.

## Supported models

| Model | Optional Dependency | Adapter Module |
|---|---|---|
| [yifeihu/TF-ID-large](https://huggingface.co/yifeihu/TF-ID-large) | `uv sync --extra tfid` | `data_snapshot.inference.tfid` |
| [juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) | `uv sync --extra doclayout_yolo` | `data_snapshot.inference.doclayoutyolo` |
| [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout) | `uv sync --extra yolo11` | `data_snapshot.inference.yolo11` |
| [Armaggheddon/yolo26-document-layout](https://huggingface.co/Armaggheddon/yolo26-document-layout) | `uv sync --extra yolo26` | `data_snapshot.inference.yolo26` |

## Generating prediction files

These prediction files will be used in evaluation.

```shell
# Example: DocLayout-YOLO
uv sync --extra doclayout_yolo
uv run python -m data_snapshot.inference.doclayoutyolo \
    --input_pdf_dir=pdf_input/ \
    --output_json_path=data/evaluation_input/doclayoutyolo.json
```

All adapters follow the same CLI pattern with `--input_pdf_dir` and `--output_json_path` arguments.

## Batch inference

Run all adapters across multiple batch directories:

```shell
uv run python -m data_snapshot.inference.batch_runner \
    --source unhcr --batch_start 1 --batch_end 5
```

---

# Evaluation

Evaluation framework for benchmarking layout detection models against human-annotated ground truth.

See [docs/evaluation_spec.md](docs/evaluation_spec.md) for the full evaluation specification.

## Pre-requisites

1. Generate ground truth file from human annotations.
    1. Export annotations by following the [backup procedure](#backing-up-an-annotation-project).
    2. Run the Label Studio adapter:
        ```shell
        uv run python -m data_snapshot.annotation.labelstudio_adapter \
        --input_json_path=path/to/exported_json \
        --output_json_path=data/evaluation_input/ground_truth.json
        ```

## Running an evaluation

```shell
uv run python -m data_snapshot.evaluation.evaluate_model \
    --gt_json_path=path/to/ground_truth.json \
    --pred_json_path=path/to/preds.json \
    --output_report_path=data/evaluation_output/report.json
```

Both ground truth and prediction files must conform to the [Unified Evaluation Schema v1.3](docs/data-snapshot-eval-v1.3.schema.json).

## Visualizing predictions

To render annotated page images comparing ground truth and predictions:

```shell
uv sync --extra viz
uv run python -m data_snapshot.evaluation.visualize_pages \
    --gt_json_path=path/to/gt.json \
    --pred_json_path=path/to/pred.json \
    --input_pdf_dir=pdf_input \
    --output_dir=data/visualize_pages/
```

## Merging batch predictions

Combine multiple per-batch prediction JSON files (e.g. from `batch_runner`) into a single file:

```shell
uv run python -m data_snapshot.evaluation.merge_batch_preds \
    --input_dir=data/batch_runs/ \
    --output_json=data/evaluation_input/combined.json
```

## Filtering small predictions

Remove predictions whose bounding-box area falls below a threshold:

```shell
uv run python -m data_snapshot.evaluation.filter_small_preds \
    --input_json=data/evaluation_input/preds.json \
    --output_json=data/evaluation_input/preds_filtered.json
```

---

# Metadata

Tools for converting and enforcing metadata schemas for HuggingFace dataset uploads.

## Converting UNHCR metadata

Convert UNHCR/ReliefWeb scraped metadata JSON files to the Document Metadata Schema:

```shell
uv run python -m data_snapshot.metadata.unhcr_to_schema \
    --input_dir=data/hf_metadata/unhcr/ \
    --output_dir=data/hf_metadata_converted/unhcr/
```

## Enforcing metadata schema

Unify metadata schemas across dataset subsets for HuggingFace compatibility:

```shell
uv run python -m data_snapshot.metadata.enforce_metadata_schema \
    --input_dir data/hf_metadata/unhcr/ \
    --input_dir data/hf_metadata/prwp/ \
    --output_dir data/hf_metadata_fixed/
```

---

# Misc

## Extracting snapshots

Crop annotated bounding boxes from PDF pages into individual PNG files:

```shell
uv run python -m data_snapshot.misc.extract_snapshots \
    --input_json=data/evaluation_input/ground_truth.json \
    --input_pdf_dir=pdf_input/ \
    --output_dir=data/snapshots/
```

## Splitting annotations for HuggingFace

Split a combined evaluation JSON file into per-document files for HuggingFace dataset upload:

```shell
uv run python -m data_snapshot.misc.annotations_to_hf_dataset \
    --input_json=data/evaluation_input/ground_truth.json \
    --output_dir=data/hf_annotations/
```

---

# Troubleshooting

- `PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'` when setting up Label Studio
  - Solution: Give writeable permission to the project directory.
    ```shell
    # Note: Make sure your cwd is the project directory!
    sudo chmod -R 777 .
    ```

---

# License

MIT License — see [LICENSE](LICENSE) for details.

---

# Citation

```bibtex
@misc{dy2026benchmarkingopensourcelayoutdetection,
      title={Benchmarking Open-Source Layout Detection Models for Data Snapshot Extraction from Institutional Documents}, 
      author={AJ Carl P. Dy and Aivin V. Solatorio},
      year={2026},
      eprint={2606.06242},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.06242}, 
}
```
