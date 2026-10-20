# ai4data Lab

ai4data Lab is a fully client-side WebGPU playground that runs large language models, embeddings, and zero-shot entity extraction in the browser. Documents, prompts, and model state never leave the user's machine.

## What runs in the browser

- **Gemma 4 Mobile (WGSL)** for evidence-grounded document chat. Citation IDs travel from the parser through retrieval, prompt, and the evidence map.
- **Bonsai 1.7B ONNX** as an alternate chat model loaded through Transformers.js.
- **Xenova/all-MiniLM-L6-v2** dense embeddings with reciprocal rank fusion over lexical candidates.
- **Anonym-IA/gliner_large-v2.1 ONNX** for zero-shot span extraction on arbitrary text and labels.
- **LiquidAI/LFM2.5-2.6B-ONNX** research agent driven from a Web Worker, with tool permissions and a compiled artifact.

## Requirements

- Node.js 18 or newer (Vite 8 requires a recent Node).
- A Chromium-based browser with WebGPU enabled (Chrome 113+, Edge, Brave, Arc). Safari and Firefox need nightly or experimental builds.
- Roughly 3 GB of free disk for the cached model weights, plus enough GPU memory for the selected model.

## Model assets (download separately)

The repository excludes the large ONNX and WebAssembly assets that Vite serves directly. Download them into the matching paths before running the app:

- `ai4data_lab/public/onnx/gliner_large-v2.1_q4.onnx` (about 900 MB) from `https://huggingface.co/Anonym-IA/gliner_large-v2.1/resolve/main/onnx/model_q4.onnx`
- `ai4data_lab/public/onnx/gist-embedding-v0.onnx` and `ai4data_lab/public/onnx/splade-sparse.onnx` from the corresponding Hugging Face repos.
- `ai4data_lab/public/wllama/wllama.wasm` from the `wllama` package's release artifacts, for example `https://unpkg.com/@wllama/wllama@3.5.1/dist/wllama.wasm`.

The browser also streams chat and agent weights from Hugging Face at runtime, so an internet connection is still required on first run.

## Quick start

```bash
cd ai4data_lab
npm install
npm run dev -- --host 0.0.0.0
```

Vite prints the local URL (default `http://localhost:5173/`). Use `--host 0.0.0.0` when running on a server so the page is reachable from another device.

### Other commands

```bash
npm run build    # production bundle in dist/
npm run preview  # serve the production bundle locally
npm run lint     # oxlint over the source tree
```

## Project layout

```
ai4data_lab/
  index.html              # Vite entry, sets <title> and loads main.jsx
  vite.config.js          # React + Tailwind plugins, port 5174, COOP/COEP headers
  package.json            # Scripts and dependencies
  public/                 # Static assets served at /
  src/
    main.jsx              # React root
    App.jsx               # All three tabs: Chat, Extract, Agent
    App.css, index.css    # Tailwind and global styles
    agentConfig.js        # Tools and prompts for the research agent
    modelConfig.js        # Model metadata for the UI
    worker.js             # Shared worker entry helper
    hooks/
      useAgentWorker.js   # Web Worker bridge for the LFM agent
    workers/
      agent.worker.js     # LFM runtime and tool execution
    lib/
      gemma-4-e2b.js      # Generated Gemma 4 WGSL runtime
      bonsai27b.js        # Generated Bonsai runtime
    assets/               # Logos and static images
  WEBGPU_MASTERY_GUIDE.md # Companion notes on WebGPU usage
```

## How a document chat turn works

1. The user picks a PDF, JSON, text, or Markdown file in the sidebar. `App.jsx:handleFileUpload` parses the file into citation-bearing section blocks.
2. When the chat model is ready, MiniLM embeddings are computed for the parsed blocks. The UI shows a stem-search fallback when embeddings fail to load.
3. `App.jsx:hybridRetrieve` combines lexical and dense scores with reciprocal rank fusion, `App.jsx:rerankCandidates` keeps the top five sections, and the prompt builder constrains Gemma to the surviving evidence.
4. The Gemma runtime streams tokens back into `gemmaMessages`. `AssistantMessage` and `EvidenceMapModal` render the answer with numbered citations and an evidence map.
5. Clearing chat preserves the document, embeddings, and model instance.

See `docs/gemma4-document-chat-architecture.md` in the repository root for the full developer walkthrough.

## Configuration

Set defaults through environment variables before `npm run dev` when needed:

- `VITE_DEFAULT_MODEL`: which chat model is preselected in the dropdown.
- `VITE_GLINER_VARIANT`: which GLiNER ONNX variant is preselected.

Vite already loads `.env` files at the project root, so create one if you need overrides. Most users do not need any configuration.

## Privacy

- The browser downloads model weights directly from Hugging Face or from the local `/onnx/` static asset.
- Documents are parsed in memory; the server never receives their contents.
- The LFM research agent runs inside a Web Worker and only contacts external services the user has enabled (for example, Wikipedia search) through the agent's tool list.

## CLI bridge (experimental)

The page can stream prompts to a CLI over a local WebSocket relay. The browser is the only WebGPU host; the CLI is a thin client.

```bash
cd ai4data_lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/bridge_server.py
```

Open the page with the bridge flag and load a model:

```
http://localhost:5174/?bridge=1
```

The page reports `ready` to the relay when the model is loaded. The CLI tool in `app/cli/ai4data_lab.py` connects to `ws://127.0.0.1:8765` and streams prompts. See `app/cli/ai4data_lab.py` for the command-line interface.

### Bridge limitations

- One CLI client and one page at a time.
- The bridge is opt-in. Open the page with `?bridge=1`; otherwise the relay sees no `ready` frame.
- `stop` cancels the current request and any in-flight generation because the UI composer and the CLI share the same stop flag.

## Troubleshooting

- **`navigator.gpu` missing**: enable WebGPU in your browser flags. The chat model cannot start without it.
- **First run is slow**: weights are streamed on demand and cached in CacheStorage. Subsequent runs reuse the cached files.
- **GLiNER extraction fails on long text**: the implementation builds all spans up to width 12. Shorten the input or split it into smaller batches.

## License

This project is part of the AI-DQSS repository and follows the same license as the parent project.
