# Next steps after the session

## The question the session leaves you with

Given a research or data problem, which application architecture should be built, and what evidence would show that it works?

The decision guide from the deck:

| Problem | Technique | Notebook |
|---|---|---|
| The model's behaviour must be consistent | Instructions | 01 |
| The answer must come from specific documents | Context and retrieval | 01 to 03 |
| The answer needs exact, current statistics | A data tool, called by the model | 04 |
| Other software must read the answer | A structured output with a schema | 04 |
| The task needs several steps or several tools | A workflow in code, or a bounded loop | 05 |
| Readers need to trust the numbers | Claims and verification | 06 |
| The application must keep working after changes | A test on fixed questions with known answers | 03 for retrieval; the evaluation cookbook for the whole application |

The techniques combine. Most applications in research and data use instructions, retrieval, one or two tools, and a suite, and never need a general agent.

## For colleagues who will build

1. Rerun the six notebooks with a key and compare the live outputs with the recorded ones.
2. Replace the document collection with ten documents from your own work, keep the header fields, and rerun notebooks 02, 03, and 06. The retrieval table will show what your documents need.
3. Replace `get_indicator` with a tool over your own data source, keeping the `claim_id` field, and rerun notebooks 04 and 06.
4. Write ten evaluation questions for your use before writing any more code.
5. Read the [agents cookbook](https://worldbank.github.io/ai4data/cookbook/serving-statistics-to-agents/) for the full design of tools over a catalog, and the [Data360 MCP server](https://github.com/worldbank/data360-mcp) for a deployed example.

## For colleagues who commission or evaluate

Questions to ask of any application proposed to you, with the notebook that shows why:

- Where do the numbers come from, and can each one be traced to a record? (04, 06)
- What does the application do when the data does not exist, or the question is ambiguous? (01, 04)
- How is retrieval tested, separately from the answers? (03)
- What bounds the loop, and can a run be inspected afterwards? (05)
- Which fixed questions is the application tested on, and when were they last run? (03)
- What does a verified mark say, and what does it leave to the reader? (06)

An application that cannot answer these is a demonstration.

## In the program

- The [Practical Guide to Evaluation Suites for Statistical AI](https://worldbank.github.io/ai4data/cookbook/evaluation-suites/) for suites at scale.
- The [Proof-Carrying Numbers page](https://worldbank.github.io/ai4data/docs/pcn/) for the packages that put marks beside numbers in an application.
- The [end-to-end reference implementation](https://worldbank.github.io/ai4data/docs/reference-implementation/) for the same ideas applied to an organization's catalog, from documentation to verified answers.
- The [tools catalogue](https://worldbank.github.io/ai4data/tools) for what exists and its status.

## Later sessions in the series

The [series page](https://worldbank.github.io/ai4data/training/) lists the planned sessions. Three of them take up topics that this session introduces.

- **Session 03 · Evaluating AI systems.** Notebook 03, section 6, tests the search on fixed questions, and notebook 06 leaves tests of a whole application out of scope. Session 03 covers the testing of complete applications.
- **Session 05 · Building trustworthy AI applications.** This session extends the verification of figures in notebook 06.
- **Session 06 · All about context: Why AI is really a search problem.** This session returns to the preparation of documents in notebook 02 and the selection of passages in notebook 03.
