# Glossary

| Term | Definition |
|---|---|
| Agent | One LLM-driven step in the pipeline with a fixed role, defined by a name and a system message in the manifest. |
| Manifest | A YAML file listing the agents and their system messages. The bundled default produces the standard five-agent pipeline. |
| Team preset | The AutoGen strategy for routing messages between agents (round-robin by default). |
| Critic | The agent that removes false positives by applying the exclusion rules. |
| Exclusion rule | A condition (general, field-level, or data-state) that causes a candidate issue to be removed or down-weighted. |
| Job | A handle for one submitted review, carrying its status, result, and any error. |
| Severity | An integer 1–5 reflecting an issue's impact, not the confidence that it is real. |
| Key path | A dotted reference to a field, with array indices in brackets, identifying exactly which value an issue concerns. |
| `model_client` | The injected LLM connection; the single point through which all provider communication flows. |
