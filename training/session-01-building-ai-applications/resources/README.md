# Participant guide

Session 01 · Building AI applications for research and data

## The notebooks

Six notebooks build one application step by step. Open them in Google Colab from the repository, or clone the repository and run them locally.

| Notebook | Adds | Time to read |
|---|---|---|
| [01 · From a language model to an application](../notebooks/01-from-model-to-application.ipynb) | Instructions and context | 15 min |
| [02 · Preparing knowledge for an application](../notebooks/02-preparing-knowledge.ipynb) | Metadata, chunks, embeddings, an index | 15 min |
| [03 · Retrieval-augmented generation](../notebooks/03-retrieval-augmented-generation.ipynb) | Retrieval, citations, a retrieval failure and its fixes | 20 min |
| [04 · Connecting the application to data and tools](../notebooks/04-tools-and-structured-outputs.ipynb) | A data tool, tool calls, a typed output | 20 min |
| [05 · Workflows and agents](../notebooks/05-workflows-and-agents.ipynb) | Two tools, a fixed workflow and a bounded loop | 20 min |
| [06 · Making the application trustworthy](../notebooks/06-verification-and-evaluation.ipynb) | Claim verification, an evaluation suite | 20 min |

Every notebook is committed with its outputs, so it can be read without running anything. The notebooks are built on LangChain, LangGraph, FastMCP, Pydantic, and the program's `pcn-core`; `notebooks/requirements.txt` lists the versions.

## Running without a key

The notebooks run in a fresh Colab environment with no API key. Model outputs are then recorded ones, labelled `[recorded output: ...]`, replayed by a LangChain chat model that stands in for the live one. Everything else runs for real: the vector store, retrieval, the data tool, the MCP server, the schemas, the agent graph, the verifier, and the evaluation.

## Running with a key

Set three values, in Colab under Secrets or in the environment:

| Variable | Meaning | Default |
|---|---|---|
| `WORKSHOP_API_KEY` | Key for an OpenAI-compatible chat API | none; recorded outputs are used |
| `WORKSHOP_BASE_URL` | Base URL of the API, for a gateway or a self-hosted model | the OpenAI endpoint |
| `WORKSHOP_MODEL` | Model name | `gpt-4o-mini` |

With a key, the same cells call the model live. Outputs will differ from the recorded ones in wording; the behaviours the text describes should hold. Where they do not, that is a finding worth noting.

Never write a key into a notebook. The setup cell reads it from the environment.

## Running locally

```bash
git clone https://github.com/rafmacalaba/ai4data.git     # replace with the published repository when released
cd ai4data/training/session-01-building-ai-applications-v2/notebooks
pip install -r requirements.txt jupyter
jupyter lab
```

The embedding model (about 90 MB) downloads on first use.

## The shared code

`notebooks/workshop.py` holds what the libraries do not: the worked-example data and the indicator tool over it, a chat model that replays recorded outputs, the output schemas, the claim-tag parsing that hands numbers to `pcn-core`, and the evaluation helpers. `notebooks/indicator_server.py` is the same tool as a FastMCP server. Together about 450 lines.

## Exercises

Each notebook ends with one exercise. In recorded mode some exercises ask for a prediction, because a changed prompt has no recorded output. With a key, every exercise runs.

## Further reading

- [Glossary](glossary.md)
- [Evaluation examples](evaluation-examples.md): what the evaluation table reports and how to read it
- [Next steps](next-steps.md): where to go after the session, by role
