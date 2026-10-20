// Builds the session 01 deck (deck.pptx) and its outline (outline.md) from one source.
//
//   npm install pptxgenjs react react-dom react-icons sharp   (once, anywhere; then set NODE_PATH)
//   node build_deck.js
//
// The outline is the reviewable text of every slide; the deck is regenerated from this file.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT_PPTX = path.join(__dirname, "deck.pptx");
const OUT_MD = path.join(__dirname, "outline.md");

const HEX = {
  text: "1F2A37",
  navy: "0B4A8B",
  teal: "0A8F8C",
  gold: "B08A2E",
  slate: "5B6B7C",
  green: "2E7D32",
  rust: "C2410C",
  tint: "EEF3F8",
  white: "FFFFFF",
  mint: "D6EEEC",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
pres.title = "Building AI applications for research and data";
const W = 13.333;
const H = 7.5;
const M = 0.6;
const FOOTER = "AI for Data – Data for AI · Training series · Session 01 · Draft for review";

pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.3, w: W - 2 * M, h: 1.5, fontSize: 40, bold: true, color: HEX.white, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.95, w: W - 2 * M, h: 1.6, fontSize: 18, color: HEX.white, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.42, w: W - 2 * M, h: 0.8, fontSize: 28, bold: true, color: HEX.navy, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: M, y: H - 0.45, w: 9, h: 0.3, fontSize: 9, color: HEX.slate, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 9, color: HEX.slate, align: "right" },
});
pres.defineSlideMaster({
  title: "DIVIDER",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: W - 2 * M, h: 1.2, fontSize: 36, bold: true, color: HEX.white, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: W - 2 * M, h: 1.6, fontSize: 18, color: HEX.white, valign: "top", margin: 0 }, text: "" } },
  ],
});

const outline = [];
let n = 0;

function text(s, str, o) {
  s.addText(str, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: HEX.text, valign: "top" }, o));
}
function bullets(s, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 5 } }));
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: HEX.text, valign: "top" }, o));
}
function card(s, x, y, w, h, name, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || HEX.tint }, line: { color: fill || HEX.tint, width: 0 }, objectName: name });
}
function box(s, x, y, w, h, label, fill, fontColor, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name || label });
  s.addText(label, { x, y, w, h, fontSize: 12, bold: true, color: fontColor || HEX.white, align: "center", valign: "middle", margin: 0.05, isTextBox: true });
}
function arrow(s, x, y, w, name) {
  s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h: 0.4, fill: { color: "C9D3DE" }, line: { color: "C9D3DE", width: 0 }, objectName: name });
}
const TH = { fill: { color: HEX.navy }, color: HEX.white, bold: true, fontSize: 12, valign: "middle", fontFace: "Calibri" };
const th = (t) => ({ text: t, options: TH });
const cell = (t, o) => ({ text: t, options: Object.assign({ fontSize: 12, color: HEX.text, valign: "top", fontFace: "Calibri" }, o || {}) });
function mono(s, str, o) {
  s.addText(str, Object.assign({ isTextBox: true, margin: 0.1, fontSize: 11, fontFace: "Courier New", color: HEX.text, valign: "top", fill: { color: HEX.tint } }, o));
}

function slide(master, title, meta) {
  const s = pres.addSlide({ masterName: master });
  n += 1;
  if (title) s.addText(title, { placeholder: "title" });
  outline.push({ n, title, body: [], notes: meta && meta.notes ? meta.notes : "", demo: meta && meta.demo ? meta.demo : "" });
  return s;
}
function record(lines) {
  outline[outline.length - 1].body.push(...lines);
}
function notes(s, t) {
  s.addNotes(t);
  outline[outline.length - 1].notes = t;
}

// -----------------------------------------------------------------------------
// The application diagram, reused with a stage highlighted
// -----------------------------------------------------------------------------
const STAGES = ["Instructions and context", "Prepared knowledge", "Retrieval", "Tools and schemas", "Workflow or agent", "Verification and evaluation"];
function diagram(s, y, highlight) {
  const gap = 0.15;
  const bw = (W - 2 * M - 5 * gap) / 6;
  STAGES.forEach((label, i) => {
    const x = M + i * (bw + gap);
    const on = highlight === undefined || highlight === i || (Array.isArray(highlight) && highlight.includes(i));
    box(s, x, y, bw, 0.7, `${i + 1}. ${label}`, on ? HEX.navy : "DCE3EA", on ? HEX.white : HEX.slate, `stage ${i + 1}`);
  });
}

// =============================================================================
// Slides
// =============================================================================

// 1 Title
{
  const s = slide("TITLE", "Building AI applications for research and data");
  s.addText(
    [
      { text: "Training series · Session 01", options: { breakLine: true } },
      { text: "90 minutes: one application built step by step, in six notebooks", options: { breakLine: true, fontSize: 14 } },
      { text: "AI for Data – Data for AI · Office of the WBG Chief Statistician and Development Data Group", options: { fontSize: 14 } },
    ],
    { placeholder: "body" }
  );
  record(["Training series · Session 01", "90 minutes: one application built step by step, in six notebooks"]);
  notes(s, "Welcome. The session builds one application from a bare model to a verified, evaluated system, in six notebooks. Two audiences: those who will build, and those who commission or evaluate. Questions at the notebook transitions.");
}

// 2 Objectives
{
  const s = slide("CONTENT", "Objectives and format");
  const lw = 6.0;
  text(s, "Two audiences", { x: M, y: 1.45, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Colleagues who will build AI applications and want each capability added to a working application, with code to rerun", "Colleagues who commission, oversee, or evaluate such applications and need to know what each capability is for and how to tell whether the result is reliable"], { x: M, y: 1.9, w: lw, h: 2.2 });
  text(s, "Afterwards, participants can", { x: M, y: 4.2, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Name the components of an AI application and the problem each solves", "Choose an architecture for a research or data question and say what evidence would show that it works", "Rerun the six notebooks, built on LangChain, LangGraph, FastMCP, and pcn-core, with a question and documents of their own"], { x: M, y: 4.65, w: lw, h: 2.0 });
  const rx = M + lw + 0.5;
  card(s, rx, 1.45, W - M - rx, 5.2, "question card");
  text(s, "The one question the session answers", { x: rx + 0.3, y: 1.6, w: W - M - rx - 0.6, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "Given a research or data problem, which application architecture should be built, and what evidence would show that it works?", { x: rx + 0.3, y: 2.1, w: W - M - rx - 0.6, h: 1.4, fontSize: 16 });
  text(s, "Format", { x: rx + 0.3, y: 3.7, w: 4, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["37 minutes of slides, 53 of notebook demonstration, then questions", "Six notebooks, one worked example, four planted problems", "Everything runs without an API key; model outputs are then recorded and labelled. Stack: LangChain, LangGraph, FastMCP, Pydantic, pcn-core"], { x: rx + 0.3, y: 4.15, w: W - M - rx - 0.6, h: 2.3 });
  record(["Two audiences: builders; commissioners and evaluators", "Afterwards: name the components; choose an architecture and the evidence; rerun the notebooks", "The one question: given a research or data problem, which architecture, and what evidence", "Format: 37 min slides, 53 min demonstration, questions; six notebooks; runs without a key"]);
  notes(s, "State the two audiences so that each knows what to listen for. The question at the right is what the session is for; return to it on the last slide.");
}

// 3 Worked example
{
  const s = slide("CONTENT", "The worked example");
  card(s, M, 1.45, W - 2 * M, 1.3, "question");
  text(s, "“How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data, summarize the main patterns, and cite the sources.”", { x: M + 0.3, y: 1.6, w: W - 2 * M - 0.6, h: 1.0, fontSize: 16, italic: true, color: HEX.navy, valign: "middle" });
  const cols = [
    ["Data", ["Five World Development Indicators series for Kenya and Rwanda, 2010 to 2024", "Retrieved from the World Bank API on 13 October 2026, with the indicator metadata", "Internet users, mobile subscriptions, GDP per capita, GDP growth, population"]],
    ["Documents", ["Four country briefs written for the workshop from the data", "One note on how the internet-users indicator is defined", "One outdated summary from 2016 and two distractors"]],
    ["Planted problems", ["An ambiguous question", "An irrelevant retrieval result and an outdated one", "A request for data that does not exist", "A numerical claim no source supports"]],
  ];
  const gap = 0.3;
  const cw = (W - 2 * M - 2 * gap) / 3;
  cols.forEach(([h, items], i) => {
    const x = M + i * (cw + gap);
    text(s, h, { x, y: 3.05, w: cw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
    bullets(s, items, { x, y: 3.5, w: cw, h: 2.6, fontSize: 13 });
  });
  text(s, "The application starts as a model answering from memory and gains one capability per notebook. The question never changes, so what changes between notebooks is the capability.", { x: M, y: 6.2, w: W - 2 * M, h: 0.6, fontSize: 12.5, color: HEX.slate });
  record(["The question (as above)", "Data: five WDI series, Kenya and Rwanda, 2010 to 2024, retrieved 13 October 2026", "Documents: four briefs, one definition note, one outdated summary, two distractors", "Planted problems: ambiguity, irrelevant and outdated retrieval, missing data, an unsupported claim"]);
  notes(s, "One question through the whole session. The planted problems are why each capability exists; name them now so that participants watch for them.");
}

// 4 Architecture
{
  const s = slide("CONTENT", "What makes a language model an application");
  text(s, "The model is one component of an engineered system. The other components select what it sees, give it tools, check what it produces, and measure the result.", { x: M, y: 1.4, w: W - 2 * M, h: 0.7, fontSize: 14 });
  diagram(s, 2.3);
  const rows = [
    [th("Component"), th("Problem it solves"), th("Notebook")],
    [cell("Instructions and context"), cell("The model behaves consistently and answers from supplied evidence"), cell("01")],
    [cell("Prepared knowledge"), cell("Documents become searchable units with metadata"), cell("02")],
    [cell("Retrieval"), cell("The right passages reach the model for each question"), cell("03")],
    [cell("Tools and schemas"), cell("Numbers come from data; answers have a shape other software can read"), cell("04")],
    [cell("Workflow or agent"), cell("Several tools and steps, bounded and inspectable"), cell("05")],
    [cell("Verification and evaluation"), cell("Each number is checked; the whole is measured across questions"), cell("06")],
  ];
  s.addTable(rows, { x: M, y: 3.3, w: W - 2 * M, colW: [3.2, 7.5, 1.43], border: { type: "solid", pt: 0.5, color: "C9D3DE" }, rowH: 0.42, margin: 0.05, objectName: "components table" });
  record(["The model is one component; the others select, equip, check, and measure", "Six components: instructions and context; prepared knowledge; retrieval; tools and schemas; workflow or agent; verification and evaluation", "One notebook per component"]);
  notes(s, "This diagram returns on every section slide with the current stage highlighted. The table is the map of the session.");
}

// 5 Section: instructions and context
{
  const s = slide("CONTENT", "Instructions, context, and the question");
  diagram(s, 1.4, 0);
  const lw = 6.2;
  text(s, "Three parts of every call", { x: M, y: 2.4, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Instructions: what the application wants, including how to cite, what to do when information is missing, when to ask", "Context: the evidence supplied with the question", "The question itself"], { x: M, y: 2.85, w: lw, h: 1.8 });
  text(s, "Kept separate because each is changed by a different person for a different reason.", { x: M, y: 4.7, w: lw, h: 0.6, fontSize: 13, color: HEX.slate });
  const rx = M + lw + 0.5;
  mono(s, "system:  You are a research assistant for development data.\n         Answer only from the supplied context. Cite the\n         document id after each figure. If the context does\n         not contain what is needed, say so.\n\nuser:    CONTEXT:\n         [kenya-digital] ... 35.0 percent in 2024 ...\n         [rwanda-digital] ... 31.7 percent in 2024 ...\n\n         QUESTION: How have internet access and ...", { x: rx, y: 2.4, w: W - M - rx, h: 3.1 });
  text(s, "What instructions can and cannot do", { x: M, y: 5.45, w: W - 2 * M, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "They shape behaviour: citations, handling of missing information, asking on ambiguity. They cannot supply data the model does not have. Notebook 01 shows the same question answered from memory and from four briefs.", { x: M, y: 5.9, w: W - 2 * M, h: 0.8, fontSize: 13 });
  record(["Three parts: instructions, context, question; kept separate", "Instructions shape behaviour and cannot supply data", "Demonstration: notebook 01, zero context against supplied context; missing information; the ambiguous question"]);
  outline[outline.length - 1].demo = "Notebook 01, sections 2 to 5";
  notes(s, "Demonstrate notebook 01: the zero-context answer against the data (wrong and unsourced), then the same question with four briefs, then the missing-year question, then the ambiguous question with and without the instruction to ask.");
}

// 6 What context costs
{
  const s = slide("CONTENT", "Why context has to be selected");
  const items = [
    ["Four briefs", "about 1,000 tokens", "fit in one call"],
    ["A real collection", "thousands of documents", "does not fit, and most of it is irrelevant to any one question"],
    ["Every token supplied", "costs money and time", "and dilutes the passages that matter"],
  ];
  items.forEach(([a, b, c], i) => {
    const y = 1.5 + i * 1.15;
    card(s, M, y, W - 2 * M, 0.95, "cost " + i);
    text(s, a, { x: M + 0.3, y, w: 3.2, h: 0.95, fontSize: 15, bold: true, color: HEX.navy, valign: "middle" });
    text(s, b, { x: M + 3.6, y, w: 3.2, h: 0.95, fontSize: 15, valign: "middle" });
    text(s, c, { x: M + 7.0, y, w: W - 2 * M - 7.3, h: 0.95, fontSize: 13, color: HEX.slate, valign: "middle" });
  });
  text(s, "So the application needs a way to select the passages that answer the question. That is what the next two components do: prepare the collection so that it can be searched, and retrieve from it per question.", { x: M, y: 5.1, w: W - 2 * M, h: 1.0, fontSize: 14 });
  record(["Four briefs fit in one call; a collection does not", "Every token costs and dilutes", "Hence preparation and retrieval"]);
  notes(s, "The bridge from notebook 01 to 02 and 03. Keep it short.");
}

// 7 Preparing knowledge
{
  const s = slide("CONTENT", "Preparing knowledge: metadata, chunks, embeddings");
  diagram(s, 1.4, 1);
  const steps = ["documents", "metadata", "chunks", "embeddings", "index"];
  const gap = 0.35;
  const bw = (W - 2 * M - 4 * gap) / 5;
  steps.forEach((t, i) => {
    const x = M + i * (bw + gap);
    box(s, x, 2.45, bw, 0.6, t, HEX.teal, HEX.white, "prep " + t);
    if (i < 4) arrow(s, x + bw + 0.03, 2.55, gap - 0.06, "prep arrow " + i);
  });
  const cols = [
    ["Metadata is knowledge", "Date, type, country, section. The 2016 summary reads as current; only its date says otherwise. Filters, ranking, and citations all run on metadata."],
    ["Chunk boundaries decide what can be found", "Fixed windows cut through sentences and drop headings. Section chunks keep the title and the heading, so a passage says what it is about."],
    ["Embeddings place text by meaning", "A vector per chunk; similar meaning, nearby vectors. A small sentence-transformer model through LangChain, held in a vector store. The index is only as good as the text and metadata it was built from."],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cols.forEach(([h, b], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 3.35, cw, 2.75, "prep card " + i);
    text(s, h, { x: x + 0.2, y: 3.5, w: cw - 0.4, h: 0.6, fontSize: 14, bold: true, color: HEX.navy });
    text(s, b, { x: x + 0.2, y: 4.15, w: cw - 0.4, h: 1.9, fontSize: 12.5 });
  });
  text(s, "Demonstration: notebook 02, the outdated summary's header, fixed windows against sections, and a search over a poorly prepared index against a prepared one.", { x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontSize: 12.5, color: HEX.slate });
  record(["Pipeline: documents, metadata, chunks, embeddings, index", "Metadata is knowledge; chunk boundaries decide what can be found; embeddings place text by meaning", "Demonstration: notebook 02"]);
  outline[outline.length - 1].demo = "Notebook 02, sections 2, 3, and 5";
  notes(s, "Show the 2016 document's header, then the fixed windows that strand a heading against the section chunks, then the poorly prepared index. Notebooks 02 and 03 are demonstrated from one file.");
}

// 8 Retrieval
{
  const s = slide("CONTENT", "Retrieval-augmented generation");
  diagram(s, 1.4, 2);
  const parts = ["question", "retriever", "top-k passages", "instructions + passages + question", "model", "answer with citations"];
  const gap = 0.25;
  const bw = (W - 2 * M - 5 * gap) / 6;
  parts.forEach((t, i) => {
    const x = M + i * (bw + gap);
    box(s, x, 2.45, bw, 0.7, t, i === 1 || i === 2 ? HEX.teal : HEX.tint, i === 1 || i === 2 ? HEX.white : HEX.text, "rag " + t);
    if (i < 5) arrow(s, x + bw + 0.02, 2.6, gap - 0.04, "rag arrow " + i);
  });
  const lw = 6.2;
  text(s, "What it gives", { x: M, y: 3.45, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Answers from the collection, each figure cited to a passage with its document, section, and year", "A collection of any size, with a few passages per call"], { x: M, y: 3.9, w: lw, h: 1.3 });
  text(s, "Where it fails", { x: M, y: 5.2, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["The 2016 summary ranks first for “latest available figure” and the model faithfully reports 2015", "Retrieval always returns something; a question outside the collection still gets three passages"], { x: M, y: 5.65, w: lw, h: 1.2 });
  const rx = M + lw + 0.5;
  card(s, rx, 3.45, W - M - rx, 3.3, "fixes card", HEX.mint);
  text(s, "Fixes live on the retrieval side", { x: rx + 0.25, y: 3.6, w: W - M - rx - 0.5, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["A metadata filter: exclude documents dated before 2020", "A recency weight in the ranking", "A reformulated query: what the passage would say", "A score threshold set from examples, and instructions to say when the passages do not answer"], { x: rx + 0.25, y: 4.05, w: W - M - rx - 0.5, h: 2.6, fontSize: 13 });
  record(["Pipeline: question, retriever, top-k passages, prompt, model, cited answer", "Gives: cited answers from a collection of any size", "Fails: outdated passage ranks first; retrieval always returns something", "Fixes on the retrieval side: metadata filter, recency weight, query reformulation, threshold plus instructions"]);
  outline[outline.length - 1].demo = "Notebook 03, sections 1 to 4";
  notes(s, "Demonstrate the worked example through the pipeline, then the retrieval failure on the latest-figure question, then the three fixes. A faithful answer to the wrong evidence is wrong; the model cannot repair a bad selection.");
}

// 9 Retrieval and generation evaluated separately
{
  const s = slide("CONTENT", "Retrieval and generation are two problems");
  const lw = 6.0;
  card(s, M, 1.45, lw, 2.4, "retrieval card");
  text(s, "Retrieval problem", { x: M + 0.25, y: 1.6, w: lw - 0.5, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The top-k passages do not include the evidence. Test it without a model: for each question, did the hits include an expected document? Fix with chunking, metadata, ranking, or the query.", { x: M + 0.25, y: 2.05, w: lw - 0.5, h: 1.7, fontSize: 13 });
  card(s, M + lw + 0.4, 1.45, W - M - (M + lw + 0.4), 2.4, "generation card");
  text(s, "Generation problem", { x: M + lw + 0.65, y: 1.6, w: W - M - (M + lw + 0.65) - 0.25, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The evidence was there and the answer did not use it, or used it wrongly. Fix with instructions or a different model. Testing it needs the answer, and so a model call.", { x: M + lw + 0.65, y: 2.05, w: W - M - (M + lw + 0.65) - 0.25, h: 1.7, fontSize: 13 });
  mono(s, "id   category           k=2    k=4    expected documents\nq1   main               hit    hit    kenya-digital, rwanda-digital, kenya-economy, rwanda-economy\nq4   missing-data       miss   hit    kenya-digital\nq8   outdated-evidence  hit    hit    kenya-digital", { x: M, y: 4.1, w: W - 2 * M, h: 1.3 });
  text(s, "The retrieval table from notebook 03: eight questions, two values of k, no model called. Mixing the two problems makes both impossible to improve, because a fix for one is invisible in a metric that blends them.", { x: M, y: 5.55, w: W - 2 * M, h: 0.9, fontSize: 13 });
  record(["Retrieval problem: evidence not in the hits; test without a model; fix chunking, metadata, ranking, query", "Generation problem: evidence present and unused; fix instructions or model", "The retrieval table from notebook 03"]);
  outline[outline.length - 1].demo = "Notebook 03, section 5";
  notes(s, "Show the retrieval table. This separation is the first evaluation idea of the session and returns in notebook 06.");
}

// 10 Tools
{
  const s = slide("CONTENT", "Tools: numbers come from data");
  diagram(s, 1.4, 3);
  const lw = 6.4;
  text(s, "A tool is a function with a description the model can read", { x: M, y: 2.4, w: lw, h: 0.45, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The model fills the parameters from the question and asks for a call. The application runs the function and returns the result. Absence of data is a result the application can test for.", { x: M, y: 2.85, w: lw, h: 1.0, fontSize: 13 });
  mono(s, 'get_indicator(indicator="NY.GDP.PCAP.CD", country="Rwanda", start=2023, end=2023)\n-> {"name": "GDP per capita (current US$)",\n    "source": "World Development Indicators",\n    "observations": [{"year": 2023, "value": 1061.06,\n                      "claim_id": "NY.GDP.PCAP.CD:RWA:2023"}]}', { x: M, y: 3.95, w: lw, h: 1.6 });
  const rx = M + lw + 0.5;
  card(s, rx, 2.4, W - M - rx, 3.15, "memory vs data");
  text(s, "From memory and from data", { x: rx + 0.25, y: 2.55, w: W - M - rx - 0.5, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "“Approximately 1,000 US dollars, please verify.”", { x: rx + 0.25, y: 3.0, w: W - M - rx - 0.5, h: 0.5, fontSize: 13, italic: true, color: HEX.rust });
  text(s, "“1,061 US dollars in 2023, World Development Indicators, NY.GDP.PCAP.CD.”", { x: rx + 0.25, y: 3.5, w: W - M - rx - 0.5, h: 0.6, fontSize: 13, italic: true, color: HEX.green });
  text(s, "The memory answer is close and hedged, with no way to tell how close. The data answer carries the indicator, the year, the source, and an identifier that the verifier will use.", { x: rx + 0.25, y: 4.2, w: W - M - rx - 0.5, h: 1.3, fontSize: 12.5 });
  text(s, "The model's memory is for language. Statistics come from a tool over authoritative data. Demonstration: notebook 04, sections 1 to 4.", { x: M, y: 5.75, w: W - 2 * M, h: 0.6, fontSize: 13, color: HEX.slate });
  record(["A tool: function plus description; the model fills parameters, the application runs it", "Memory against data: hedged approximation against a sourced value with a claim id", "Absence of data is a testable result", "Demonstration: notebook 04"]);
  outline[outline.length - 1].demo = "Notebook 04, sections 1 to 4";
  notes(s, "Demonstrate the memory answer against the tool, the tool schema, the model calling the tool, and the missing year. Point at claim_id: notebook 06 uses it.");
}

// 11 Structured outputs and MCP
{
  const s = slide("CONTENT", "Structured outputs and MCP");
  const lw = 6.2;
  text(s, "A schema makes the answer a contract", { x: M, y: 1.45, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "An answer that another program will store, display, or compare needs a fixed shape. The schema is sent with the request and the response is validated on the way back. A damaged field raises an error, so a wrong value never passes silently.", { x: M, y: 1.9, w: lw, h: 1.3, fontSize: 13 });
  mono(s, "class Observation(BaseModel):\n    indicator: str\n    country: str\n    year: int\n    value: float\n    claim_id: str\n\nclass ComparisonAnswer(BaseModel):\n    countries: list[CountrySummary]\n    comparison: str\n    sources: list[str]", { x: M, y: 3.3, w: lw, h: 2.6 });
  const rx = M + lw + 0.5;
  text(s, "MCP: one server, many clients", { x: rx, y: 1.45, w: W - M - rx, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The Model Context Protocol is a standard for exposing tools like get_indicator to any AI client. Notebook 04 serves the tool with FastMCP in a dozen lines and calls it back through the FastMCP client. The program's Data360 MCP server exposes search, metadata, and data tools over the World Bank data platform; a desktop assistant, the Data AI Chatbot, or a notebook call the same tools.", { x: rx, y: 1.9, w: W - M - rx, h: 1.7, fontSize: 13 });
  const parts = ["Data360 MCP server", "search · metadata · data"];
  box(s, rx, 3.7, W - M - rx, 0.6, parts.join("  ·  "), HEX.teal, HEX.white, "mcp server");
  const clients = ["Desktop assistant", "Data AI Chatbot", "A notebook", "An agent"];
  const cw = (W - M - rx - 3 * 0.15) / 4;
  clients.forEach((c, i) => {
    box(s, rx + i * (cw + 0.15), 4.75, cw, 0.6, c, HEX.tint, HEX.text, "client " + c);
  });
  text(s, "The notebook lists the Data360 server's sixteen tools and runs one search live, with a guard for offline sessions.", { x: rx, y: 5.5, w: W - M - rx, h: 0.7, fontSize: 12.5, color: HEX.slate });
  record(["A schema makes the answer a contract: sent with the request, validated on return", "MCP: one server, many clients; the tool served with FastMCP; the Data360 MCP server as the deployed example", "The notebook lists the Data360 tools and runs one search live"]);
  outline[outline.length - 1].demo = "Notebook 04, sections 5 and 6";
  notes(s, "Demonstrate the structured output and the validation error. Name MCP once with the Data360 server; the demonstration at the end shows it live.");
}

// 12 Decision guide
{
  const s = slide("CONTENT", "Which technique solves which problem");
  const rows = [
    [th("Problem"), th("Technique"), th("Notebook")],
    [cell("The model's behaviour must be consistent"), cell("Instructions"), cell("01")],
    [cell("The answer must come from specific documents"), cell("Context, prepared knowledge, retrieval"), cell("01 to 03")],
    [cell("The answer needs exact, current statistics"), cell("A data tool called by the model"), cell("04")],
    [cell("Other software must read the answer"), cell("A structured output with a schema"), cell("04")],
    [cell("Several steps or several tools"), cell("A workflow in code, or a bounded loop"), cell("05")],
    [cell("Readers need to trust the numbers"), cell("Claims and verification"), cell("06")],
    [cell("The application must keep working after changes"), cell("An evaluation suite"), cell("06")],
  ];
  s.addTable(rows, { x: M, y: 1.45, w: W - 2 * M, colW: [5.2, 5.5, 1.43], border: { type: "solid", pt: 0.5, color: "C9D3DE" }, rowH: 0.48, margin: 0.06, objectName: "decision table" });
  text(s, "The techniques combine. An application for research or data typically uses instructions, retrieval, one or two tools, and a suite. Few need a general agent.", { x: M, y: 5.5, w: W - 2 * M, h: 0.8, fontSize: 14 });
  record(["Table: problem, technique, notebook (seven rows)", "The techniques combine; few applications need a general agent"]);
  notes(s, "The slide to photograph. Say that the rows are complementary and that the next slides cover the last three.");
}

// 13 Workflows and agents
{
  const s = slide("CONTENT", "Workflows and agents");
  diagram(s, 1.4, 4);
  const lw = (W - 2 * M - 0.4) / 2;
  card(s, M, 2.4, lw, 3.0, "fixed card");
  text(s, "A fixed workflow", { x: M + 0.25, y: 2.55, w: lw - 0.5, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The order of calls is written in code: search, fetch four series, compose. The model is called once, to write. Every run makes the same calls. Predictable, testable, cheap.", { x: M + 0.25, y: 3.0, w: lw - 0.5, h: 1.3, fontSize: 13 });
  mono(s, "passages = search_documents.invoke({\"query\": question})\nseries   = [indicator_lookup(i, c) for i, c in PAIRS]\nanswer   = (prompt | model).invoke({...})", { x: M + 0.25, y: 4.3, w: lw - 0.5, h: 0.95 });
  card(s, M + lw + 0.4, 2.4, lw, 3.0, "agent card");
  text(s, "A model-directed loop", { x: M + lw + 0.65, y: 2.55, w: lw - 0.5, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The tools are offered with their schemas; the model chooses which to call and when to stop. Adapts to questions the designer did not anticipate, and can choose wrongly.", { x: M + lw + 0.65, y: 3.0, w: lw - 0.5, h: 1.3, fontSize: 13 });
  mono(s, "agent = create_agent(model, tools, system_prompt=...)\nresult = agent.invoke({\"messages\": [question]},\n                      config={\"recursion_limit\": 10})\n# LangGraph runs the loop; the messages are the trace", { x: M + lw + 0.65, y: 4.3, w: lw - 0.5, h: 0.95 });
  text(s, "Three controls for the loop: a recursion limit, the message trace of every call and result, and tools that return errors as data so that the model can retry. Demonstration: notebook 05, an out-of-scope question, a failed call, and the limit.", { x: M, y: 5.6, w: W - 2 * M, h: 0.9, fontSize: 13 });
  record(["Fixed workflow: order in code, one model call, predictable", "Model-directed loop: model chooses tools; adapts, can err", "Controls: step limit, trace, errors as data", "Demonstration: notebook 05"]);
  outline[outline.length - 1].demo = "Notebook 05, sections 1 to 5";
  notes(s, "Demonstrate the fixed workflow, then the loop on the same question, then the out-of-scope question, the failed call with retry, and the step limit.");
}

// 14 Choosing between them
{
  const s = slide("CONTENT", "Choosing between a workflow and an agent");
  const rows = [
    [th(""), th("Fixed workflow"), th("Model-directed loop")],
    [cell("Calls per run", { bold: true }), cell("Known in advance"), cell("Varies with the question")],
    [cell("A new kind of question", { bold: true }), cell("Same calls, possibly irrelevant"), cell("Adapts, possibly wrongly")],
    [cell("Cost", { bold: true }), cell("Predictable"), cell("Bounded by the step limit")],
    [cell("Testing", { bold: true }), cell("Test the composer"), cell("Test the trace and the answer")],
    [cell("Use when", { bold: true }), cell("The questions have one shape"), cell("The questions vary and the tools are few and safe")],
  ];
  s.addTable(rows, { x: M, y: 1.45, w: W - 2 * M, colW: [2.8, 4.6, 4.73], border: { type: "solid", pt: 0.5, color: "C9D3DE" }, rowH: 0.5, margin: 0.06, objectName: "workflow table" });
  text(s, "The worked example has one shape, so the fixed workflow is the right design for it. A loop earns its place when users ask questions the designer did not anticipate, and only with a step limit, a trace, and tools that fail as data. Prefer the simplest design that completes the task reliably.", { x: M, y: 4.75, w: W - 2 * M, h: 1.2, fontSize: 14 });
  record(["Table: calls per run, new questions, cost, testing, use when", "The worked example wants the fixed workflow", "Prefer the simplest design that completes the task reliably"]);
  notes(s, "For the commissioning audience: ask which of the two a proposal is, and why.");
}

// 15 Verification
{
  const s = slide("CONTENT", "Verification: every number checked against its record");
  diagram(s, 1.4, 5);
  const lw = 6.4;
  text(s, "Claim tags", { x: M, y: 2.4, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "The model writes each figure inside a tag that names the observation it came from. A verifier compares the number shown with the record under a policy: rounded to the stated decimals, or exact.", { x: M, y: 2.85, w: lw, h: 1.0, fontSize: 13 });
  mono(s, '<claim id="IT.NET.USER.ZS:KEN:2024" policy="rounded"\n       decimals="1">35.0</claim> percent', { x: M, y: 3.9, w: lw, h: 0.8 });
  text(s, "This is the Proof-Carrying Numbers protocol used in the program's chatbot; the notebook compares with the same pcn-core code.", { x: M, y: 4.85, w: lw, h: 0.7, fontSize: 12.5, color: HEX.slate });
  const rx = M + lw + 0.5;
  mono(s, "✓ verified   35.0  IT.NET.USER.ZS:KEN:2024  holds 34.98\n✗ flagged    35.7  IT.NET.USER.ZS:RWA:2024  holds 31.70\n✓ verified  2,133  NY.GDP.PCAP.CD:KEN:2024  holds 2133.46\n· unmarked     70                           no claim tag", { x: rx, y: 2.4, w: W - M - rx, h: 1.5, fontSize: 10.5 });
  bullets(s, ["Verified: the number matches its record", "Flagged: the model changed a number", "Unmarked: no claim, so no mark, whether or not the number happens to be right. The rule fails closed."], { x: rx, y: 4.05, w: W - M - rx, h: 1.5, fontSize: 13 });
  text(s, "What the check leaves to the reader: whether the record is the right one for the question, whether the method behind it is sound, and whether the sentence around the number interprets it correctly.", { x: M, y: 5.7, w: W - 2 * M, h: 0.8, fontSize: 13 });
  record(["Claim tags name the observation; a verifier compares under a policy", "Verified, flagged, unmarked; fail-closed", "What the check leaves to the reader", "Demonstration: notebook 06, sections 1 and 2"]);
  outline[outline.length - 1].demo = "Notebook 06, sections 1 and 2";
  notes(s, "Demonstrate the clean answer with eight verified claims, then the planted answer with a flagged and an unmarked number.");
}

// 16 Evaluation
{
  const s = slide("CONTENT", "Evaluation: the application measured across questions");
  const lw = 5.8;
  text(s, "The suite", { x: M, y: 1.45, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Eight questions with expected documents, expected numbers, and expected behaviour", "Categories: main, factual, ambiguous, missing data, out of scope, definition, outdated evidence", "Run against two configurations: retrieval only, and tools with claims", "Rerun after every change to instructions, tools, index, or model"], { x: M, y: 1.9, w: lw, h: 2.6, fontSize: 13 });
  text(s, "Columns", { x: M, y: 4.5, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["retrieval: did the hits include an expected document", "numbers: expected values found in the answer", "cited: a retrieved document or the source named", "verify: verified, flagged, unmarked counts"], { x: M, y: 4.95, w: lw, h: 1.8, fontSize: 13 });
  const rx = M + lw + 0.4;
  mono(s, "B · tools and claims\n\nid   category           retrieval  numbers  cited  verify\nq1   main               hit        4/4      yes    4v 0f 0u\nq3   ambiguous          hit        0/0      yes    no data used\nq4   missing-data       hit        0/0      yes    1v 0f 0u\nq5   out-of-scope       hit        0/0      yes    no data used\nq8   outdated-evidence  hit        1/1      yes    1v 0f 0u\n\nA · retrieval only\n\nq8   outdated-evidence  hit        0/1      yes    n/a", { x: rx, y: 1.45, w: W - M - rx, h: 3.6, fontSize: 10.5 });
  text(s, "q8 separates the configurations: retrieval alone reports 2015 from the 2016 summary; tools fetch the series and report 2024. Every number in B is verified; A has nothing to verify against.", { x: rx, y: 5.2, w: W - M - rx, h: 1.3, fontSize: 12.5 });
  record(["The suite: eight questions, expected evidence and behaviour, two configurations, rerun on every change", "Columns: retrieval, numbers, cited, verify", "q8 separates the configurations; B is verifiable, A is not"]);
  outline[outline.length - 1].demo = "Notebook 06, sections 3 to 5";
  notes(s, "Show the two tables. A table like this is what distinguishes a demonstration from a measured application.");
}

// 17 Four checks
{
  const s = slide("CONTENT", "Four checks for four questions");
  const checks = [
    ["Retrieval", "Did we find the right evidence?", "Hits against expected documents, no model needed", HEX.teal],
    ["Grounding", "Is the answer written from that evidence?", "Citations present and matching the hits", HEX.navy],
    ["Verification", "Does each number match its record?", "Claim tags checked under a policy; fail-closed", HEX.gold],
    ["Evaluation", "Does the application behave across questions?", "The suite, rerun on every change", HEX.slate],
  ];
  const gap = 0.3;
  const cw = (W - 2 * M - 3 * gap) / 4;
  checks.forEach(([h, q, how, hex], i) => {
    const x = M + i * (cw + gap);
    box(s, x, 1.5, cw, 0.6, h, hex, HEX.white, "check " + h);
    card(s, x, 2.2, cw, 2.6, "check card " + h);
    text(s, q, { x: x + 0.2, y: 2.35, w: cw - 0.4, h: 1.0, fontSize: 14, bold: true, color: HEX.navy });
    text(s, how, { x: x + 0.2, y: 3.4, w: cw - 0.4, h: 1.3, fontSize: 12.5 });
  });
  text(s, "A plausible answer passes none of these by being plausible. An appropriately validated answer passes the four, and says what it leaves to the reader.", { x: M, y: 5.1, w: W - 2 * M, h: 0.8, fontSize: 14 });
  record(["Retrieval: right evidence found; Grounding: answer written from it; Verification: numbers match records; Evaluation: behaviour across questions", "Plausible is not a check"]);
  notes(s, "The slide for the commissioning audience: four questions to ask of any application.");
}

// 18 Data360 Chat
{
  const s = slide("CONTENT", "The same ideas in the application the program runs");
  const lw = 6.2;
  text(s, "Data360 Chat", { x: M, y: 1.45, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Tools over the World Bank data catalog through the Data360 MCP server: search, metadata, data", "A bounded loop with a trace, so that an answer can be inspected", "Claim tags and a verifier in the renderer, with the Proof-Carrying Numbers packages", "An evaluation suite over the questions users ask"], { x: M, y: 1.9, w: lw, h: 2.6, fontSize: 13 });
  text(s, "Lessons from building it", { x: M, y: 4.5, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Search over a catalog misses series; the metadata has to carry what users say", "A dimension listed in metadata may have no data; the tool has to say so", "The verifier catches changed numbers; it cannot catch the wrong series", "The suite grows from use: every failure found becomes a question"], { x: M, y: 4.95, w: lw, h: 1.8, fontSize: 13 });
  const rx = M + lw + 0.5;
  card(s, rx, 1.45, W - M - rx, 5.2, "demo card", HEX.mint);
  text(s, "Live demonstration", { x: rx + 0.25, y: 1.6, w: W - M - rx - 0.5, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["The worked example on the live system", "A year with no data", "The ambiguous question", "Which indicator to use, and its definition", "One number with its source and its mark"], { x: rx + 0.25, y: 2.05, w: W - M - rx - 0.5, h: 2.6, fontSize: 13 });
  text(s, "Queries are pretested and screenshots stand in if the application or the network is unavailable. See demo/data360-chat-demo.md.", { x: rx + 0.25, y: 4.8, w: W - M - rx - 0.5, h: 1.6, fontSize: 12, color: HEX.slate });
  record(["Data360 Chat: MCP tools, bounded loop with trace, claim tags and verifier, evaluation suite", "Lessons: catalog search misses, empty dimensions, the verifier's limits, the suite grows from use", "Live demonstration: five pretested queries"]);
  outline[outline.length - 1].demo = "Data360 Chat, demo/data360-chat-demo.md";
  notes(s, "Five minutes. Point at the tool calls, the citation, and the mark beside the number. Name one limitation and how the suite catches it.");
}

// 19 Recap
{
  const s = slide("CONTENT", "Recap");
  diagram(s, 1.4);
  const rows = [
    [th("Notebook"), th("Adds"), th("The lesson")],
    [cell("01"), cell("Instructions and context"), cell("Instructions shape behaviour; context makes the answer specific. Neither supplies data the model lacks.")],
    [cell("02"), cell("Prepared knowledge"), cell("Metadata and chunk boundaries decide what retrieval can find.")],
    [cell("03"), cell("Retrieval"), cell("A faithful answer to the wrong evidence is wrong; fix retrieval on the retrieval side; test it on its own.")],
    [cell("04"), cell("Tools and schemas"), cell("Numbers come from data; absence is a testable result; a schema is a contract.")],
    [cell("05"), cell("Workflow or agent"), cell("Prefer the simplest design that completes the task; bound and trace any loop.")],
    [cell("06"), cell("Verification and evaluation"), cell("Check each number against its record; measure the whole across questions; rerun on every change.")],
  ];
  s.addTable(rows, { x: M, y: 2.35, w: W - 2 * M, colW: [1.2, 3.0, 7.93], border: { type: "solid", pt: 0.5, color: "C9D3DE" }, rowH: [0.36, 0.55, 0.45, 0.55, 0.5, 0.5, 0.55], margin: 0.05, objectName: "recap table" });
  record(["Six notebooks, six lessons (table)"]);
  notes(s, "Read the lessons column. Then return to the question of slide 2.");
}

// 20 Resources and questions
{
  const s = slide("CONTENT", "Resources and questions");
  const lw = 6.2;
  text(s, "Take home", { x: M, y: 1.45, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["The six notebooks, with cached outputs; they run in Colab without a key", "The shared module and the FastMCP server, about 450 lines on top of the libraries", "The participant guide, the glossary, the evaluation examples, and next steps by role", "The worked-example assets: data, documents, questions"], { x: M, y: 1.9, w: lw, h: 2.4, fontSize: 13 });
  text(s, "In the program", { x: M, y: 4.3, w: lw, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  bullets(s, ["Practical Guide to Serving Official Statistics to AI Agents", "Practical Guide to Evaluation Suites for Statistical AI", "Proof-Carrying Numbers", "Data360 MCP server and the Data AI Chatbot", "The end-to-end reference implementation"], { x: M, y: 4.75, w: lw, h: 2.0, fontSize: 13 });
  const rx = M + lw + 0.5;
  card(s, rx, 1.45, W - M - rx, 5.2, "question card", HEX.tint);
  text(s, "The question to leave with", { x: rx + 0.3, y: 1.6, w: W - M - rx - 0.6, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "Given a research or data problem, which application architecture should be built, and what evidence would show that it works?", { x: rx + 0.3, y: 2.1, w: W - M - rx - 0.6, h: 1.4, fontSize: 16 });
  text(s, "Questions", { x: rx + 0.3, y: 3.7, w: W - M - rx - 0.6, h: 0.4, fontSize: 15, bold: true, color: HEX.navy });
  text(s, "15 to 30 minutes. Questions on a notebook are taken with that notebook open.", { x: rx + 0.3, y: 4.15, w: W - M - rx - 0.6, h: 1.0, fontSize: 13 });
  record(["Take home: notebooks, module, guide, glossary, evaluation examples, next steps, assets", "In the program: agents cookbook, evaluation cookbook, PCN, Data360 MCP, chatbot, reference implementation", "The question to leave with", "Questions: 15 to 30 minutes"]);
  notes(s, "Links are in resources/README.md. Close on the question.");
}

// =============================================================================
// Write
// =============================================================================
async function main() {
  await pres.writeFile({ fileName: OUT_PPTX });
  const md = ["# Session 01 deck outline", "", "Generated from `build_deck.js`. This file is the reviewable text of every slide; edit the script and regenerate both.", ""];
  for (const o of outline) {
    md.push(`## ${o.n}. ${o.title}`, "");
    for (const b of o.body) md.push(`- ${b}`);
    if (o.demo) md.push("", `Demonstration: ${o.demo}`);
    if (o.notes) md.push("", `Notes: ${o.notes}`);
    md.push("");
  }
  fs.writeFileSync(OUT_MD, md.join("\n"));
  console.log("wrote", OUT_PPTX, "and", OUT_MD, "slides:", n);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
