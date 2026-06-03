## Setting up an annotation project (manual)

### 1. Pre-requisites

1. Install the repository.
    ```shell
    pip install -e .
    ```
2. Install Poppler.
    ```shell
    sudo apt-get install poppler-utils
    ```

### 2. Converting PDFs to images and creating tasks for Label Studio
1. Add PDF files to the `pdf_input` directory.
2. Run `python create_tasks_manual.py --dataset_name={dataset}`. The `dataset_name` parameter may be set into any string.
3. This will generate the following files into the `labelstudio_data/{dataset}` directory:
    - Individual PNG files for each page of each PDF
    - A `tasks.json` file.

### 3. Creating an annotation project
1. Setup the project.
    1. Open Label Studio and click `Create Project`.
    2. Fill out Project Name page.
    3. In Data Import, click `Upload Files` and select the `tasks.json` generated in the previous section.
    4. In Labeling Setup, select `Multi-page document annotation`.
    5. Create label names. This can be edited later.
    6. Click `Save`.
2. Setup the dataset.
    1. Go to the project's settings.
    8. Select `Cloud Storage` > `Add Source Storage` > `Local Files` > `Next`.
    9. Add a Storage Title.
    10. In Absolute local path, replace `/label-studio/data/your-subdirectory` with `/label-studio/data/{dataset}`.
    11. Click `Test Connection` > `Next`.
    12. Import Method: `Tasks - Treat each JSON, JSONL, or Parquet...`
    13. Click `Next` > `Save`. (Important: Do NOT click `Save & Sync`.)
3. Go to the project tab. Each row (called a "task") should correspond to a PDF file to annotate.
