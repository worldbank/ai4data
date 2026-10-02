# Study data

The source images and parent-document metadata are published separately in
[`ai4data/data-snapshot`](https://huggingface.co/datasets/ai4data/data-snapshot).
They are not duplicated in this repository.

Remote downloads are pinned to dataset revision
`626f3c54ba6d8f6e0ba0795f9e544e24b2650faf`; the manifests select the exact
files used by the paper from that revision.

The two CSV files in `manifests/` are the authoritative study splits:

- `development_snapshots.csv`: 210 snapshots used for schema development.
- `heldout_snapshots.csv`: 202 snapshots used by all held-out validations.

Materialize both collections with:

```shell
uv run python scripts/fetch_data.py
```

To copy from an existing local clone instead of downloading:

```shell
uv run python scripts/fetch_data.py --source-root /path/to/data-snapshot
```
