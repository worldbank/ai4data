# Agentic AI Approach

This chapter explains why a multi-agent pipeline is the appropriate architecture for metadata quality assurance at scale. It traces the progression from simple API-based LLM usage—which delivers genuine capability gains—through the structural limitations that motivate a more governed, agentic design.

---

## API-Based LLMs for Metadata Quality

API-based LLM usage means sending a metadata record to a hosted or on-premises language model—such as OpenAI, Anthropic, or Azure OpenAI—via an HTTP API call and receiving a natural-language assessment in return. No custom model training is required; the model is accessed as a service and instructed through a prompt.

In the metadata QA context, a typical call includes the full metadata record in the prompt and asks the model to identify issues: fields that are missing, descriptions that are inconsistent, titles that contain typos, or values that contradict one another. The response is a structured list of detected problems, often formatted as JSON for downstream processing. Because these models are trained on large and diverse corpora, they bring strong language understanding to the task without needing domain-specific fine-tuning.

This approach represents a genuine capability leap for metadata quality work at scale. A single API call can assess a rich, free-form metadata record in seconds, surface issues a rule-based system would miss, and do so across any language or schema that the model understands. For organisations with large metadata catalogs, API-based LLM access makes systematic quality checking feasible for the first time.

---

## Limitations of API-Only Approaches

However, API-based access alone does not address all organisational requirements. A single LLM call is episodic: it has no memory of previous assessments, no awareness of what has changed since the last review, and no record of decisions made by human reviewers. Metadata quality assurance also demands explicit governance—rules about which issue types to surface, which fields to exclude, and how findings should be categorised and prioritised—and these requirements cannot be reliably enforced through prompt instructions alone. Without structured orchestration, each call is independent, results are not comparable across runs, and audit trails do not exist.

**Table 1. Structural Limits of LLM-Only Approaches**

| Limitation | Why it matters for metadata QA |
|---|---|
| Episodic execution | Metadata QA requires longitudinal awareness: what changed, what was reviewed before. |
| Weak governance | Organisational rules cannot be reliably enforced through prompts alone. |
| Lack of auditability | Decisions must be traceable to evidence, versions, and reviewers. |
| Inconsistency over time | Model and prompt drift undermine comparability across releases. |
| Poor integration with rules | Schema checks, code lists, and regex rules must be deterministic. |

---

## What Agentic AI Frameworks Are

Agentic AI frameworks are software toolkits that embed large language models within structured systems designed to support multi-step reasoning, persistence, and governance. Rather than treating an LLM call as a one-off interaction, these frameworks provide the scaffolding needed to turn LLM reasoning into repeatable, controllable processes.

In practice, agentic AI frameworks wrap LLMs with several core capabilities: orchestration of multi-step workflows, memory and context persistence across executions, integration with external tools and data sources (such as APIs, code modules, and databases), explicit planning and execution logic, and monitoring, governance, and auditability mechanisms. Together, these capabilities transform LLMs from reactive text generators into components of systems that can act autonomously or semi-autonomously while remaining accountable and inspectable.

For metadata enhancement, this distinction is critical. Metadata quality work requires repeated application of logic, awareness of prior assessments, coordination among specialised checks, and traceability of outcomes. Agentic AI frameworks provide the structural foundation needed to meet these requirements.

**Table 2. Examples of Agentic AI Frameworks**

| Framework | Language(s) | What It Does | Good For |
|---|---|---|---|
| LangChain + LangGraph | Python, JavaScript | Orchestrates multi-agent workflows; integrates memory stores, tools, and structured task graphs | Custom pipelines, conditional workflows, data-driven actions |
| Microsoft AutoGen | Python, .NET | Built-in multi-agent conversation and orchestration framework with explicit role separation (e.g., planner, analyst, executor) | Complex distributed logic and enterprise workflows |
| OpenAI Agents SDK | Python, JavaScript | SDK for building agents with tool calling, planning, and execution loops around LLMs | Rapid development of agentic workflows using hosted LLM services |

---

## How Agentic Tools Fit into a Metadata Enhancement Workflow

For metadata enhancement, the goal is not simply to generate answers but to implement repeated, conditional, and auditable logic. Agentic AI tools enable this by structuring metadata quality work as a coordinated pipeline of specialised agents.

A typical metadata quality agent pipeline begins with a **trigger mechanism**—such as a scheduled run or an event indicating that a dataset has been published or updated. A **detection agent** then scans metadata and applies quality rules to identify potential issues. A **context or memory agent** integrates persistent storage—such as a vector database or relational store—to retain knowledge of previous flags, annotations, and decisions. A **classification agent** assigns issue categories, such as consistency, completeness, or semantic error. A **severity scoring agent** applies policy-informed logic that considers rules, historical context, and prior reviewer behaviour. An **action and escalation agent** determines whether an issue should be presented as a suggestion, routed for human review, or escalated to a subject matter expert. Finally, a **logging and audit layer** records decisions, agent versions, and execution traces.

This modular, agentic structure allows metadata enhancement to be both scalable and governable, ensuring that AI support strengthens—rather than undermines—human-centred metadata quality management.

`ai4data.metadata.reviewer` implements a focused subset of this pipeline. The **primary** and **secondary** agents serve as the detection layer. The **critic** enforces governance through explicit exclusion rules. The **categorizer** and **severity_scorer** handle classification and prioritisation. Persistent storage, escalation routing, and a logging layer are outside the current scope, which is designed as a stateless, per-submission quality check that can be embedded into a broader workflow. See [implementation.md](implementation.md) for the full pipeline details.

---

## References

- [ai4data Implementation](implementation.md) — Architecture and pipeline details for `ai4data.metadata.reviewer`
- [Microsoft AutoGen documentation](https://microsoft.github.io/autogen/) — Multi-agent framework
- [autogen-agentchat](https://pypi.org/project/autogen-agentchat/) — AutoGen conversation and team orchestration library
