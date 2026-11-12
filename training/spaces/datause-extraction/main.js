// Data-use extraction front end.
// The models run on the Space (ZeroGPU). This page sends the text to the Gradio
// `extract` endpoint. It streams the mentions first and then, for each named mention,
// the text that the metadata model is writing, token by token. The page renders the
// result as a tree, as JSON, or as a trace.

const MAX_WORDS = 4000;

const $ = (id) => document.getElementById(id);
const textEl = $("text");
const runBtn = $("run");
const statusEl = $("status");
const statusPill = $("status-pill");
const results = $("results");
const resultsPretty = results.querySelector(".result-pretty");
const resultsJson = results.querySelector(".result-json");
const resultsTrace = results.querySelector(".result-trace");
const segOpts = document.querySelectorAll(".seg-opt");
const segJson = document.querySelector('.seg-opt[data-mode="json"]');
const segTrace = document.querySelector('.seg-opt[data-mode="trace"]');
const steps = document.querySelectorAll(".pipeline .step");

let lastResult = null;

const SAMPLES = [
  "To assess the distributional impact of agricultural interventions, we draw on several complementary data sources that jointly cover the period of interest. The first is the Demographic and Health Survey (DHS), collected by EuStatistics in 2021, which records health and nutrition outcomes for a nationally representative sample of households across all administrative regions. The second is the Ghana Living Standards Survey (GLSS), administered by the Ghana Statistical Service in 2017, which provides detailed consumption, expenditure, and income modules at the household level. Because both surveys report household consumption using broadly comparable definitions, we are able to pool the two samples and estimate a common poverty line across years and districts. We further merge in the World Bank's World Development Indicators (WDI) to control for district-level economic conditions, population density, and access to infrastructure. Following standard practice, we restrict the estimation sample to rural households and exclude observations with implausible consumption values, and we cluster standard errors at the district level throughout. All monetary variables are expressed in constant local currency units to ensure comparability across survey rounds.",
  "Poverty rates in this paper are estimated using the National Family Health Survey (NFHS-3) for India, a large, nationally representative household survey fielded across all states and union territories. We restrict the estimation sample to ever-married women aged 15 to 49 and cluster standard errors at the district level to account for within-district correlation in household outcomes. The survey records a rich set of household characteristics, including caste, religion, asset ownership, and household size, which we use as controls throughout the analysis. Following the existing literature, we construct a household wealth index from the reported asset holdings using principal components analysis, and we measure poverty with a consumption-based welfare aggregate deflated to a common base year. To guard against measurement error, we exclude observations with missing values on any of the core variables and winsorize the dependent variable at the first and ninety-ninth percentiles. All specifications include state fixed effects and a full set of survey-month fixed effects, and we report robust standard errors clustered at the primary sampling unit level.",
  "This paper estimates the returns to secondary education using a within-family identification strategy that compares the labor-market outcomes of siblings who differ in their educational attainment. The central empirical challenge is that ability and family background jointly determine both schooling and later earnings, so naive comparisons of more- and less-educated individuals confound the causal effect of schooling with pre-existing differences in talent and resources. By restricting attention to brothers and sisters raised in the same household, we remove all time-invariant household characteristics that would otherwise bias the estimates. Our main finding is that an additional year of secondary schooling raises hourly earnings substantially, and the effect is larger for women than for men. We show that this result is robust to a wide range of alternative specifications, including controls for birth order, age spacing, and gender composition, and to alternative definitions of the outcome measure. We also document that the returns to schooling vary with local labor-market conditions, being higher in urban areas and in regions with greater industrial concentration. Finally, we provide evidence that the estimated effects are not driven by selective migration or by differential measurement error in self-reported earnings.",
  "Considerable provincial variation in insurance coverage is evident in the H8BS. Two of the surveys contain self-assessed health (in the GSCF, the assessment was actually done by the woman of the household), which at least in industrialized countries has been found to be a good predictor of mortality. We measured schooling by years of schooling in the CHNS and the GSCF, but in the H8BS have left education coded as levels of attainment. The basic CHNS results are reported in Table 4, the results for the Gansu panel in Table 6, and the results for the Health VIII baseline survey in Table 7.",
  "One of the main advantages of the data from the EAC-I is that plot size is measured by GPS. This avoids measurement errors that lead to biased results. We aggregate crop yields at the household level. We convert production to kilograms using World Bank conversion factors. Interestingly, our data indicate that crop yields vary according to the extent to which polygyny is practiced in the commune.",
  "Data are from the 2015-16 National Family Health Survey (NFHS) of India. Panels A-C present 2SLS estimates as specified by equations 1 and 2 and using cross-sectional data. Panel D presents 2SLS estimates using the retrospective woman-year panel data. Column (1) shows estimates for the full sample, while columns (2) and (3) restrict the samples to those living in rural and urban areas, respectively.",
  "A second contribution of this paper is to show the importance of decomposing the effects with respect to distance from the mines. Given the spatial heterogeneity of the results, we explore the effects in an individual-level, difference-in-differences analysis by using spatial lag models to allow for nonlinear effects with distance from mine. We also allow for spillovers across districts, in a district-level analysis. We use two complementary geocoded household data sets to analyze outcomes in Ghana: the Demographic and Health Survey (DHS) and the Ghana Living Standard Survey (GLSS), which provide information on a wide range of welfare outcomes.",
  "We use geospatial data to test the validity of the local continuity assumption around the boundaries of guerrilla-controlled areas. Elevation was obtained from NASA's Shuttle Radar Topography Mission (SRTM). Information on surface water bodies comes from the MERIT Hydro dataset. Agro-climatic yield rasters with a spatial resolution of five arc minutes (nine km) come from the Global Agro-Ecological Zones (GAEZ) project. For all yields, we are using the 30-year average beginning in 1961.",
  "The average energy spending share observed in the HBS 2019 in Bulgaria is 13.1 percent. The largest energy spending component is on electricity (59.3 percent of all energy expenditure), followed by other solid fuels (27.5 percent) and heat energy (9.3 percent). One stream of literature defines the energy poor as all households that spend more than 10 percent of their total expenditure on energy.",
  "In the COVID-19 NLPS round 4 conducted in August 2020, households were asked to compare their current income from various sources to their income in August 2019, and to indicate whether these incomes had increased, decreased, or remained the same.",
  "Natural amenities were calculated using several GIS layers from the GHS layers. Water bodies were used to calculate dummy variables for location near the sea (coast). We used the 250 meters resolution raster MODIS Water Mask (Carroll et al., 2009) for this purpose. Finally, we used the Lloyd's lists of maritime and fluvial ports to account for the presence of ports.",
  "This annex summarizes how the individual indicators from the WGI data sources are assigned to the six governance dimensions corresponding to the six aggregate WGI indicators in the most recent year in which they appear in the WGI dataset. For most data sources, this is 2023 (the most recent year available in the 2024 WGI update).",
  "The application of this fiscal incidence framework requires both microdata and macrodata. The microdata come from the second wave of the National Survey on Household Living Conditions and Agriculture (ECVMA), and the state budget and national account data provide the macrodata. The ECVMA is a survey analogous to the Living Standards Measurement Study-Integrated Survey on Agriculture (LSMS-ISA), a World Bank Group program designed to collect nationally representative data at the household and individual levels on consumption, agriculture, and other social indicators. The second wave of the panel survey in Niger was implemented in 2014, covering urban and rural areas in all regions of the country. The sample was chosen through a random two-stage process, using the 2001 General Census of Population and Housing as the base for the first stage.",
  "In order to proxy for the origin characteristics, a set of economic variables aggregated at the province (and urban/rural) level is used. Average per capita monthly household expenditures and average per capita land areas owned by households are computed by province (and urban/rural) from the VHLSS 2004 in order to capture living standards and the opportunities for agricultural, forestry, and fishery activities. To proxy for the chance of obtaining a job in the state sector, the proportion of state workers in total employment at each sending area is calculated from the VHLSS 2004. In terms of the proportion of workers employed by foreign firms, the number of workers in the latter sector for each location is taken from the Enterprise Survey data (GSO) and divided by total employment.",
  "Following the survey identification, we proceeded to review the indicators in the identified surveys that were related to norms and female labor force participation. Building on the literature on agency and empowerment, we identified a set of indicators and variables repeatedly appearing in the literature as norms-related indicators. The surveys considered included the Demographic and Health Survey (DHS), World Values Survey (WVS), Multiple Indicator Cluster Surveys (MICS), Gallup Poll, and Gender and Adolescence: Global Evidence (GAGE). WVS, though not available for all countries in the region, was selected due to the availability of gender attitude variables that could serve as a proxy for relevant norms. Upon careful review, the latter three were dropped due to limited data availability and duplication of variables already available in the DHS.",
  "According to IOM's Integrated Location Assessment 4, the rate of physical violence, threats, mistrust and returnees feeling unwelcome decreased within 12 months, but still remained. Context indicators such as blocked return, social cohesion and movement restriction remained similar, according to IOM's Return Index 12. Context indicators in Yathrib showed marked improvement, with threats between tribal groups, blocked IDP return, and returnees feeling unwelcome reported before the LPA was signed and absent three months post-signature.",
  "The SEIS 2025 dataset and its reports were published in December 2025 on the UNHCR Microdata Library. A year later, its uptake is broad. Across agencies, the Ukraine Refugee Response Plan (RRP) relied heavily on SEIS, with 200 citations confirmed and roughly 100 more potentially. In the private sector, Deloitte quoted SEIS in its European Labour Market Outlook and its CEE Ukrainian-refugee integration study, both of which draw on SEIS findings. Among governments, Poland uses SEIS to update its 2026 refugee integration strategy and adjust labour-market access provisions, while Romania draws on SEIS for its own integration planning. In research and academia, SEIS microdata pulled from the UNHCR Microdata Library has been used by several universities and cited in peer-reviewed studies on refugee labour-market outcomes and education access. Media coverage of SEIS has also been widespread.",
];

// --- status ---------------------------------------------------------------------------
function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusPill.className = "status-pill " + kind;
}

function setStep(name) {
  const order = Array.from(steps).map((s) => s.dataset.step);
  const at = order.indexOf(name);
  steps.forEach((s, i) => {
    s.classList.toggle("active", i === at);
    s.classList.toggle("done", i < at);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- the extract endpoint ---------------------------------------------------------------
// Inside the Hugging Face page, ZeroGPU attributes a request to the signed-in visitor
// through headers that the parent frame hands over on request (the same handshake that
// Gradio's own frontend uses). Without them the call counts against an anonymous quota.
const HF_PARENT = location.hostname.endsWith(".hf.space") ? "https://huggingface.co" : null;
let parentSupportsHeaders = false;
if (HF_PARENT && window.parent !== window) {
  window.addEventListener("message", (e) => {
    if (e.data === "supports-zerogpu-headers") parentSupportsHeaders = true;
  });
  window.parent.postMessage("supports-zerogpu-headers", HF_PARENT);
}

function zerogpuHeaders() {
  if (!parentSupportsHeaders) return Promise.resolve({});
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => resolve({}), 1500);
    channel.port1.onmessage = ({ data }) => {
      clearTimeout(timer);
      channel.port1.close();
      resolve(data || {});
    };
    window.parent.postMessage("zerogpu-headers", HF_PARENT, [channel.port2]);
  });
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body.detail || body.error || response.statusText;
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

// Calls `extract` through Gradio's queue and reads the server-sent events.
// `onUpdate` receives each yielded event: "spans", then "token" and "slot" events for each
// named mention, then "done".
async function callExtract(text, threshold, onUpdate) {
  const headers = { "Content-Type": "application/json", ...(await zerogpuHeaders()) };
  const join = await fetch("gradio_api/call/extract", {
    method: "POST",
    headers,
    body: JSON.stringify({ data: [text, threshold] }),
  });
  if (!join.ok) throw new Error(await errorMessage(join));
  const { event_id } = await join.json();

  const stream = await fetch(`gradio_api/call/extract/${event_id}`);
  if (!stream.ok || !stream.body) throw new Error(await errorMessage(stream));
  const reader = stream.body.pipeThrough(new TextDecoderStream()).getReader();

  let buffer = "";
  let finished = false;
  while (!finished) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      const payload = data.join("\n");
      if (event === "generating" || event === "complete") {
        onUpdate(JSON.parse(payload)[0]);
        if (event === "complete") finished = true;
      } else if (event === "error") {
        let message = "The server returned an error.";
        try {
          const body = JSON.parse(payload);
          message = typeof body === "string" ? body : (body && (body.error || body.message)) || message;
        } catch { /* keep the default */ }
        throw new Error(message);
      }
    }
  }
  if (!finished) throw new Error("The connection closed before the result arrived.");
}

// --- pretty tree rendering ----------------------------------------------------------------
const ATTRIBUTE_KEYS = ["acronym", "producer", "year", "geography", "version", "data_type", "themes"];
const USAGE_KEYS = ["role", "usage_action", "summary"];

function leafHtml(k, v) {
  return `<div class="tree-leaf"><span class="k">${escapeHtml(k)}</span><span class="arrow">→</span><span class="v">${escapeHtml(String(v))}</span></div>`;
}

function leavesHtml(obj, keys) {
  const out = [];
  for (const k of keys) {
    let v = obj && obj[k];
    if (Array.isArray(v)) v = v.join(", ");
    if (v === undefined || v === null || v === "") continue;
    out.push(leafHtml(k, v));
  }
  return out.length ? out.join("") : '<div class="tree-none">none</div>';
}

function confidenceChip(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return "";
  const pct = Math.round(score * 100);
  const level = score >= 0.85 ? "c-high" : score >= 0.7 ? "c-mid" : "c-low";
  return `<span class="tree-conf ${level}" title="confidence ${pct}%">${pct}%</span>`;
}

function mentionTreeHtml(m, index, pending) {
  const badge = `<span class="badge badge-${m.label}">${escapeHtml(m.label)}</span>`;
  const name = `<span class="tree-name">${escapeHtml(m.text)}</span>`;
  const root = `<div class="tree-root">${badge}${name}${confidenceChip(m.score)}</div>`;
  if (m.label !== "NAMED_DATA") {
    return `<div class="tree">${root}<div class="tree-none">no attributes: the metadata model reads named mentions only</div></div>`;
  }
  const waiting = '<div class="tree-stream streaming">attributing…</div>';
  const slot = (name, keys, label) =>
    `<div class="tree-group"><div class="tree-label">${label}</div>` +
    `<div class="tree-slot" data-slot="${name}">${pending ? waiting : leavesHtml(m[name], keys)}</div></div>`;
  return (
    `<div class="tree" data-mention="${index}">${root}<div class="tree-body">` +
    slot("attributes", ATTRIBUTE_KEYS, "attributes") +
    slot("usage", USAGE_KEYS, "usage") +
    `</div></div>`
  );
}

function renderPretty(payload, pending) {
  const named = payload.mentions.filter((m) => m.label === "NAMED_DATA").length;
  const t = payload.timings;
  const perf =
    `<div class="perf">` +
    `<span class="stat">span <b>${t.span_ms} ms</b></span>` +
    `<span class="stat">mentions <b>${payload.mentions.length}</b></span>` +
    `<span class="stat">named <b>${named}</b></span>` +
    `<span class="stat">attribution <b>${pending ? "…" : t.attribution_ms + " ms"}</b></span>` +
    `</div>`;
  const trees = payload.mentions.map((m, i) => mentionTreeHtml(m, i, pending)).join("");
  resultsPretty.innerHTML = payload.mentions.length ? perf + trees : perf + '<div class="empty">No mentions found.</div>';
}

function slotElement(index, slot) {
  return resultsPretty.querySelector(`.tree[data-mention="${index}"] [data-slot="${slot}"]`);
}

// Writes the model's text into its slot as it arrives, then swaps in the parsed fields.
function handleEvent(event) {
  if (event.stage === "spans" || event.stage === "done") {
    renderPretty(event, event.stage === "spans");
    if (event.stage === "spans") setStatus("Reading named mentions", "busy");
  } else if (event.stage === "token") {
    const el = slotElement(event.index, event.slot);
    if (el) el.innerHTML = `<div class="tree-stream streaming">${escapeHtml(event.text)}</div>`;
  } else if (event.stage === "slot") {
    const el = slotElement(event.index, event.slot);
    if (el) el.innerHTML = leavesHtml(event.value, event.slot === "attributes" ? ATTRIBUTE_KEYS : USAGE_KEYS);
  }
}

function renderError(message) {
  resultsPretty.innerHTML =
    `<div class="empty"><div class="empty-title">The extraction did not run</div><div class="empty-sub">${escapeHtml(message)}</div></div>`;
}

// --- JSON and trace views -----------------------------------------------------------------
function colorizeJson(json) {
  return escapeHtml(json)
    .replace(/(&quot;[^&]*?&quot;)(\s*:)/g, '<span class="jk">$1</span>$2')
    .replace(/:\s*(&quot;[^&]*?&quot;)/g, ': <span class="js">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="jn">$1</span>')
    .replace(/:\s*(true|false|null)\b/g, ': <span class="jb">$1</span>');
}

function cleanResult() {
  return {
    input: lastResult.input,
    mentions: lastResult.mentions.map(({ text, label, attributes, usage }) => ({ text, label, attributes, usage })),
  };
}

function renderJson() {
  if (!lastResult) return;
  resultsJson.innerHTML = `<pre>${colorizeJson(JSON.stringify(cleanResult(), null, 2))}</pre>`;
}

function renderTrace() {
  if (!lastResult) return;
  const t = lastResult.timings;
  const meta =
    `<div class="json-meta">` +
    `<span>input <b>${lastResult.input.length}</b> chars</span>` +
    `<span>span <b>${t.span_ms} ms</b></span>` +
    `<span>attribution <b>${t.attribution_ms} ms</b></span>` +
    `<span>server <b>${t.total_ms} ms</b></span>` +
    `<span>round trip <b>${t.round_trip_ms} ms</b></span>` +
    `<span>mentions <b>${lastResult.mentions.length}</b></span>` +
    `</div>`;
  resultsTrace.innerHTML = meta + `<pre>${colorizeJson(JSON.stringify(lastResult, null, 2))}</pre>`;
}

const PANES = [["pretty", resultsPretty], ["json", resultsJson], ["trace", resultsTrace]];

function setMode(mode) {
  if (mode !== "pretty" && !lastResult) return;
  segOpts.forEach((o) => {
    const active = o.dataset.mode === mode;
    o.classList.toggle("active", active);
    o.setAttribute("aria-selected", String(active));
  });
  for (const [m, pane] of PANES) {
    pane.hidden = m !== mode;
    pane.setAttribute("aria-live", m === mode ? "polite" : "off");
  }
  if (mode === "json") renderJson();
  else if (mode === "trace") renderTrace();
}

segOpts.forEach((o) => o.addEventListener("click", () => setMode(o.dataset.mode)));

// --- run ------------------------------------------------------------------------------------
runBtn.addEventListener("click", async () => {
  const text = textEl.value.trim();
  if (!text) return;
  if (text.split(/\s+/).length > MAX_WORDS) {
    setStatus("Text too long", "err");
    renderError(`The text is over ${MAX_WORDS.toLocaleString()} words. Paste a shorter passage.`);
    return;
  }

  lastResult = null;
  [segJson, segTrace].forEach((s) => { if (s) s.disabled = true; });
  setMode("pretty");
  resultsJson.innerHTML = "";
  resultsTrace.innerHTML = "";
  resultsPretty.innerHTML = '<div class="empty"><div class="empty-title">Waiting for a GPU…</div><div class="empty-sub">The models are loaded; the GPU is attached for the few seconds that the request runs.</div></div>';
  runBtn.disabled = true;
  setStatus("Running", "busy");
  setStep("extract");

  const started = performance.now();
  let latest = null;
  try {
    await callExtract(text, Number($("span-threshold").value), (event) => {
      if (event.stage === "done") latest = event;
      handleEvent(event);
    });
    if (!latest) throw new Error("The server finished without a result.");
    const roundTrip = Math.round(performance.now() - started);
    const labelCounts = {};
    for (const m of latest.mentions) labelCounts[m.label] = (labelCounts[m.label] || 0) + 1;
    lastResult = {
      input: { text, length: text.length },
      timings: { ...latest.timings, round_trip_ms: roundTrip },
      label_counts: labelCounts,
      mentions: latest.mentions,
    };
    [segJson, segTrace].forEach((s) => { if (s) s.disabled = false; });
    setStatus("Done", "ready");
    setStep("results");
  } catch (e) {
    const message = (e && (e.message || e.name)) || String(e) || "unknown error";
    setStatus("Error", "err");
    renderError(message);
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
});

// --- samples --------------------------------------------------------------------------------
const sampleSegBtns = document.querySelectorAll(".sample-seg-btn");
const sampleGroups = document.querySelectorAll(".samples-group");
const sampleChips = document.querySelectorAll(".sample-chip");
const shuffleBtn = $("sample-shuffle");

function selectSample(btn) {
  if (!btn) return;
  const idx = Number(btn.dataset.sample);
  if (isNaN(idx) || !SAMPLES[idx]) return;
  textEl.value = SAMPLES[idx];
  sampleChips.forEach((c) => c.classList.toggle("active", c === btn));
}

sampleSegBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const cat = btn.dataset.category;
    sampleSegBtns.forEach((b) => {
      const active = b === btn;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
    });
    sampleGroups.forEach((g) => { g.hidden = g.dataset.group !== cat; });
  });
});

sampleChips.forEach((btn) => btn.addEventListener("click", () => selectSample(btn)));

shuffleBtn?.addEventListener("click", () => {
  const activeGroup = document.querySelector(".samples-group:not([hidden])");
  if (!activeGroup) return;
  const chips = Array.from(activeGroup.querySelectorAll(".sample-chip"));
  if (!chips.length) return;
  const current = chips.find((c) => c.classList.contains("active"));
  const pool = chips.filter((c) => c !== current);
  const choices = pool.length ? pool : chips;
  selectSample(choices[Math.floor(Math.random() * choices.length)]);
});

const thresholdInput = $("span-threshold");
const thresholdVal = $("span-threshold-val");
thresholdInput?.addEventListener("input", () => {
  thresholdVal.textContent = Number(thresholdInput.value).toFixed(2);
});

textEl.addEventListener("input", () => setStep("input"));

// The page opens with a real-world example loaded: "Spatial (DHS + GLSS)".
selectSample(document.querySelector('.sample-chip[data-sample="6"]'));
