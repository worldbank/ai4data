# Glossary

Terms as the session uses them, in the order they appear.

**Language model (LLM).** A model that produces text given text. On its own it answers from what it absorbed in training, with no access to current data and no sources.

**AI application.** A system in which the model is one component. The other components select what the model sees, give it tools, check what it produces, and measure the result.

**Instructions.** The system message: what the application wants the model to do, including how to cite, what to do when information is missing, and when to ask.

**Context.** Text supplied to the model with the question: documents, passages, data. Selecting it is the job of the application.

**Context window.** The maximum amount of text a model can receive in one call. Cost grows with every token supplied.

**Token.** The unit in which models read and bill text; about four characters of English.

**Metadata.** Information about a document or a record: id, title, type, country, date, source, section. Part of what the application knows; used for filtering, ranking, and citations.

**Chunk.** The unit of text that retrieval returns and the model reads. Section chunks keep headings and meaning; fixed-size windows cut through both.

**Embedding.** A vector that places a text so that texts with similar meaning are close. Computed by an embedding model; TF-IDF is a word-based substitute.

**Index.** The chunks, their metadata, and their embeddings, held for search. In LangChain, a vector store such as `InMemoryVectorStore`, Chroma, FAISS, or pgvector.

**Retrieval.** Selecting the chunks that best match a question, by similarity of embeddings, with optional metadata filters and ranking adjustments.

**Retrieval-augmented generation (RAG).** Retrieve passages, supply them as context, generate an answer from them with citations.

**Grounding.** The property that an answer is written from the supplied evidence and cites it.

**Tool.** A function the application exposes to the model, with a description and a parameter schema. The model asks for a call; the application runs it and returns the result.

**Tool call.** The model's request to run a tool with given arguments. The application decides whether and how to run it.

**Structured output.** An answer in a fixed shape, defined by a schema and validated, so that other software can read it.

**Schema.** The definition of a shape: fields, types, required values. Pydantic models produce JSON schemas.

**Model Context Protocol (MCP).** A standard for exposing tools, resources, and prompts to AI clients, so that one server serves many clients. The session's indicator server is written with FastMCP; the Data360 MCP server is the program's deployed example.

**Workflow.** A sequence of calls written in code, with the model used at fixed points. Predictable and testable.

**Agent.** A loop in which the model chooses which tools to call and when to stop. Adapts to the question; needs a step limit, a trace, and tools that return errors as data. Built in the session with LangGraph through `create_agent`, bounded by `recursion_limit`.

**Trace.** The record of every tool call and result in a run, kept so that a reviewer can see what the application did.

**Claim.** A number in an answer tied to the record it came from, by an identifier such as `IT.NET.USER.ZS:KEN:2024`.

**Verification.** Comparing each claim with its record under a policy (rounded to a number of decimals, or exact). Verified: matches. Flagged: differs. Unmarked: no claim, so no mark.

**Fail-closed.** The rule that a number without a claim receives no mark, whether or not it happens to be right.

**Proof-Carrying Numbers (PCN).** The program's protocol and packages for claims and verification in generated text.

**Evaluation suite.** Questions with expected evidence, expected numbers, and expected behaviour, run against the application after every change.

**Retrieval hit.** For one question, whether the top-k passages include a passage from an expected document.

**Threshold.** The retrieval score below which the application treats the result as no evidence.
