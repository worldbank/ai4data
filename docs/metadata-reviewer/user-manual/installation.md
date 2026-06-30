# Installation

## Requirements

- **Python 3.11 or newer.**
- `autogen-agentchat` (installed automatically with the package extras below).
- Valid credentials or a reachable endpoint for your chosen LLM provider.

## Installing with provider extras

Install the base package together with the extra for the provider you
intend to use. Each extra pulls in only the dependencies that provider
needs, so you do not have to install client libraries you will never
call.

```shell
pip install ai4data[metadata-reviewer,openai]      # OpenAI
pip install ai4data[metadata-reviewer,anthropic]   # Anthropic Claude
pip install ai4data[metadata-reviewer,azure]       # Azure OpenAI
pip install ai4data[metadata-reviewer,ollama]      # Ollama (local)
```

The provider packages are imported **lazily** — the code for a
provider is only imported when you call its factory method. Installing
only the `openai` extra will therefore not cause import errors as long as
you never call `from_anthropic`, and vice versa.

## Notes for corporate or proxied networks

The OpenAI and Azure factory methods construct their HTTP client with
TLS verification disabled (`httpx.AsyncClient(verify=False)`), which lets
them operate behind inspecting proxies that re-sign TLS traffic. If your
environment blocks outbound access to model endpoints entirely, the
pipeline will fail at the first agent call; in that case route through
your approved proxy or use a locally served model via `from_ollama`.

:::{tip} Tip
If outbound package installation is blocked, install the wheels on a
machine with access and transfer them, or point pip at your
organization's internal mirror. The lazy-import design means you
only need the wheels for the single provider you actually use.
:::
