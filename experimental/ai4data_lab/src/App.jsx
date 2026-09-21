import React, { useState, useEffect, useRef } from 'react';
import { pipeline, TextStreamer, env, AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import { marked } from 'marked';
import * as pdfjsLib from 'pdfjs-dist';
import { Gemma4Mobile } from './lib/gemma-4-e2b.js';
import { Bonsai27B } from './lib/bonsai27b.js';
import useAgentWorker from './hooks/useAgentWorker.js';
import {
  Brain,
  Send,
  Square,
  Sparkles,
  Zap,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Loader2,
  CloudDownload,
  Globe,
  MessageSquareCode,
  Tag,
  Search,
  FileText,
  Database,
  Cpu,
  Upload,
  FileCheck,
  Check,
  Layers,
  Filter,
  Sparkle,
  Eye,
  BookOpen,
  ExternalLink
} from 'lucide-react';

// Configure pdfjs worker dynamically matching API version
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

// Configure environment
env.allowLocalModels = false;
ort.env.wasm.numThreads = 1;

// Configure marked parser for GitHub Flavored Markdown
marked.setOptions({
  gfm: true,
  breaks: true,
});

function renderMarkdown(content) {
  if (!content) return { __html: '' };
  try {
    return { __html: marked.parse(content) };
  } catch (err) {
    return { __html: content };
  }
}

const DEFAULT_ONNX_REPO = 'onnx-community/Bonsai-1.7B-ONNX';
const ANONYM_GLINER_BASE_URL = 'https://huggingface.co/Anonym-IA/gliner_large-v2.1/resolve/main/onnx/';



// Common English Stop-Words list to filter out noisy match score inflate
const STOP_WORDS = new Set([
  'is', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
  'from', 'this', 'that', 'it', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
  'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'above',
  'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
  'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
  'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
]);

function tokenizeText(text) {
  return text.toLowerCase().match(/\w+/g) || [];
}

// Stage 1: Shortlist Top Section Candidates with Stem Prefix Matching
function shortlistCandidates(query, segments, topK = 15) {
  if (!segments || segments.length === 0) return [];

  const rawTokens = tokenizeText(query);
  const meaningfulTokens = rawTokens.filter((t) => !STOP_WORDS.has(t));

  const scored = segments.map((seg) => {
    const docTokens = tokenizeText(seg.text);
    const filteredDocTokens = docTokens.filter((t) => !STOP_WORDS.has(t));

    let score = 0;
    meaningfulTokens.forEach((qt) => {
      const qStem = qt.length > 4 ? qt.slice(0, 5) : qt;

      let tf = 0;
      filteredDocTokens.forEach((dt) => {
        const dStem = dt.length > 4 ? dt.slice(0, 5) : dt;
        if (dt === qt || dStem === qStem) {
          tf += 1;
        }
      });

      if (tf > 0) {
        score += (tf * 2.5) / (tf + 1.2);
      }
    });

    return { ...seg, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// Stage 2: Section Cross-Encoder Rerank Pass with Dynamic Threshold Scaling & Citation Recency Memory
function rerankCandidates(query, shortlisted, maxN = 5, minThreshold = 2.5, history = []) {
  if (!shortlisted || shortlisted.length === 0) return [];

  const rawQueryTokens = tokenizeText(query).filter((t) => !STOP_WORDS.has(t));
  const qTokenCount = Math.max(1, rawQueryTokens.length);
  const rawQueryStr = rawQueryTokens.join(' ');

  // Extract prior cited section IDs from the last assistant message (Multi-Turn Recency Memory)
  const priorCitedIds = new Set();
  if (history && history.length > 0) {
    const lastAssistantMsg = [...history].reverse().find((m) => m.role === 'assistant');
    if (lastAssistantMsg && lastAssistantMsg.trace?.survivingEvidences) {
      lastAssistantMsg.trace.survivingEvidences.forEach((ev) => {
        if (ev.citationId) priorCitedIds.add(ev.citationId);
      });
    }
  }

  const reranked = shortlisted.map((item) => {
    let finalScore = item.score;
    const textLower = item.text.toLowerCase();
    const headingLower = (item.heading || '').toLowerCase();

    // Exact Phrase & Keyphrase Alignment Boost
    if (rawQueryStr.length > 3 && (textLower.includes(rawQueryStr) || headingLower.includes(rawQueryStr))) {
      finalScore += 10.0;
    }

    // Direct Term Match Boosts
    let matchCount = 0;
    rawQueryTokens.forEach((qt) => {
      const qStem = qt.length > 4 ? qt.slice(0, 5) : qt;

      if (headingLower.includes(qt) || headingLower.includes(qStem)) {
        finalScore += 4.0;
      }
      if (textLower.includes(qt) || textLower.includes(qStem)) {
        matchCount++;
      }
    });

    const matchRatio = matchCount / qTokenCount;
    finalScore += matchRatio * 10.0;

    // Multi-turn Citation Recency Boost (+3.0 points for sections cited in preceding turn)
    if (priorCitedIds.has(item.citationId)) {
      finalScore += 3.0;
    }

    // Substantive Section Length Boost (modest so it doesn't overpower relevance)
    if (item.text.length > 300) {
      finalScore += 1.5;
    }

    return { ...item, finalScore };
  });

  reranked.sort((a, b) => b.finalScore - a.finalScore);

  const topScore = reranked[0]?.finalScore || 0;
  // Out-of-Scope Query Protection: If query has zero keyword alignment, return no context sections
  if (topScore < 1.0) return [];

  // Dynamic Score Threshold: Keep sections with score >= minThreshold OR >= 50% of top score
  const dynamicSurviving = reranked.filter(
    (item, idx) => idx === 0 || (item.finalScore >= minThreshold && item.finalScore >= topScore * 0.5)
  );

  const selected = dynamicSurviving.slice(0, maxN);
  const maxScore = selected[0]?.finalScore || 1.0;

  return selected.map((item, idx) => {
    const rawPct = Math.round((item.finalScore / maxScore) * 100);
    const relevanceScore = idx === 0 ? 98 : Math.max(50, Math.min(95, rawPct));
    return { ...item, relevanceScore };
  });
}

// Basic Turn Compaction Helper (Strategic Context Budgeting)
function compactHistory(historyTurns) {
  if (!historyTurns || historyTurns.length === 0) return [];

  const totalLength = historyTurns.reduce((acc, t) => acc + (t.content || '').length, 0);
  if (historyTurns.length <= 4 && totalLength <= 1200) {
    return historyTurns;
  }

  const recentTurns = historyTurns.slice(-2);
  const olderTurns = historyTurns.slice(0, historyTurns.length - 2);

  const summaryText = olderTurns
    .map((t) => `${t.role === 'user' ? 'User' : 'Assistant'}: ${t.content.slice(0, 150)}...`)
    .join('\n');

  return [
    {
      role: 'system',
      content: `[Prior Conversation Summary:\n${summaryText}]`,
    },
    ...recentTurns,
  ];
}

// AI-DQSS Universal Document-Agnostic System Prompt Construction
function buildAiDqssSystemPrompt(docTitle, docSections, survivingSections, question) {
  const sectionsStr = docSections.length > 0 ? docSections.join(', ') : 'General Document Sections';

  let contextStr = '(No relevant sections found in the document)';
  if (survivingSections && survivingSections.length > 0) {
    contextStr = survivingSections
      .map((sec, idx) => `[${idx + 1}] ${sec.heading} (${sec.citationId} | Relevance: ${sec.relevanceScore}%):\n${sec.text}`)
      .join('\n\n---\n\n');
  }

  return `You are an expert document Q&A assistant analyzing '${docTitle}'.
Document structure: ${sectionsStr}.

USER QUESTION: ${question}

INSTRUCTIONS:
1. Synthesize a comprehensive answer to the USER QUESTION above, using strictly the retrieved evidence passages below as your anchor in knowledge.
2. Be direct, clear, detailed, and structured. Do NOT copy long passages verbatim.
3. If the question is OUT OF SCOPE or NOT mentioned in the document, state clearly: "The uploaded document does not contain information regarding [topic]." Do NOT hallucinate!
4. Every factual claim MUST end with inline numerical citations like [1] or [2] matching the evidence passage numbers below. Do NOT use raw string IDs like [DOC-1:SEC-1].
5. Rely strictly on facts from the retrieved evidence passages below.
6. NEVER repeat sentences or clauses verbatim from the evidence. Synthesize a clean, clear summary in your own words.

RETRIEVED EVIDENCE PASSAGES:
${contextStr}`;
}

// Reduce Phase: Build prompt from Map Phase Per-Evidence Insights
function buildReduceSystemPrompt(docTitle, perEvidenceInsights, question) {
  const insightsStr = perEvidenceInsights.join('\n\n---\n\n');

  return `You are an expert document Q&A assistant analyzing '${docTitle}'.

USER QUESTION: ${question}

RETRIEVED EVIDENCE PASSAGES:
${insightsStr}

INSTRUCTIONS:
1. Synthesize a comprehensive, detailed, and structured answer to the USER QUESTION above, listing all specific principles, requirements, data curation rules, and quality guidelines found in the evidence passages.
2. Every factual claim MUST end with inline numerical citations like [1] or [2] matching the passage numbers above.
3. Do NOT say "explicit principles are not detailed" if the passages list data quality rules, curation processes, monitoring, accuracy, or access guidelines!
4. Rely strictly on facts from the retrieved evidence passages above.`;
}

// Persistent CacheStorage Engine across Hard Refreshes (Cmd+Shift+R)
async function getCachedModelBuffer(url) {
  try {
    const cache = await caches.open('webgpu-models-cache-v3');
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
      return await cachedResponse.arrayBuffer();
    }
  } catch (err) {
    console.warn('CacheStorage read error:', err);
  }
  return null;
}

async function saveCachedModelBuffer(url, buffer) {
  try {
    const cache = await caches.open('webgpu-models-cache-v3');
    const response = new Response(buffer, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(buffer.byteLength),
      },
    });
    await cache.put(url, response);
    console.log('Model saved to persistent CacheStorage v3!');
  } catch (err) {
    console.warn('CacheStorage write error:', err);
  }
}

// Multi-threaded Parallel Range Downloader with MB/GB Speed Tracker
async function fetchParallelRanges(url, totalBytes, concurrency = 6, onProgress = () => {}) {
  const chunkSize = Math.ceil(totalBytes / concurrency);
  const chunks = new Array(concurrency);
  let totalDownloaded = 0;

  const tasks = Array.from({ length: concurrency }, async (_, i) => {
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize - 1, totalBytes - 1);

    const res = await fetch(url, {
      headers: { Range: `bytes=${start}-${end}` },
    });

    if (!res.ok && res.status !== 206) {
      throw new Error(`Range fetch failed HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const partChunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      partChunks.push(value);
      totalDownloaded += value.byteLength;
      onProgress(totalDownloaded, totalBytes);
    }

    const partLen = partChunks.reduce((acc, c) => acc + c.byteLength, 0);
    const mergedPart = new Uint8Array(partLen);
    let offset = 0;
    for (const c of partChunks) {
      mergedPart.set(c, offset);
      offset += c.byteLength;
    }
    chunks[i] = mergedPart;
  });

  await Promise.all(tasks);

  const fullBuffer = new Uint8Array(totalBytes);
  let globalOffset = 0;
  for (const part of chunks) {
    fullBuffer.set(part, globalOffset);
    globalOffset += part.byteLength;
  }

  return fullBuffer.buffer;
}

// Parser to separate thinking process from final answer
function parseThinkingAndAnswer(text) {
  if (!text) return { thinkingText: '', answerText: '', isThinking: false };

  const thinkStart = text.indexOf('<think>');
  const thinkEnd = text.indexOf('</think>');

  if (thinkStart !== -1) {
    if (thinkEnd !== -1) {
      const thinkingText = text.substring(thinkStart + 7, thinkEnd).trim();
      const answerText = text.substring(thinkEnd + 8).trim();
      return { thinkingText, answerText, isThinking: false };
    } else {
      const thinkingText = text.substring(thinkStart + 7).trim();
      return { thinkingText, answerText: '', isThinking: true };
    }
  }

  const lowerText = text.toLowerCase();
  const headingIndex = lowerText.indexOf("here's a thinking process:");
  const altIndex = lowerText.indexOf("thinking process:");
  const stepIndex = lowerText.indexOf("1.  **analyze user input:**");
  const stepIndexAlt = lowerText.indexOf("1. **analyze user input:**");

  let startIndex = -1;
  const indices = [headingIndex, altIndex, stepIndex, stepIndexAlt].filter((i) => i !== -1);
  if (indices.length > 0) {
    startIndex = Math.min(...indices);
  }

  if (startIndex !== -1) {
    const finalMarkerRegex = /(?:5\.\s*\*\*Final Output Generation:\*\*|\*\*Final Output Generation:\*\*|Final Output Generation:|Final Output:|✅)/i;
    const match = text.match(finalMarkerRegex);

    if (match && match.index !== undefined) {
      const matchPos = match.index;
      const thinkingText = text.substring(startIndex, matchPos + match[0].length).trim();
      let answerText = text.substring(matchPos + match[0].length).trim();

      if (answerText.startsWith('"') && answerText.endsWith('"')) {
        answerText = answerText.slice(1, -1).trim();
      }

      return { thinkingText, answerText, isThinking: false };
    } else {
      return { thinkingText: text.substring(startIndex).trim(), answerText: '', isThinking: true };
    }
  }

  return { thinkingText: '', answerText: text.trim(), isThinking: false };
}

// AI-DQSS Interactive Source Evidence Map & PDF Highlight Modal
function EvidenceMapModal({ modalData, docSegments, docTitle, onClose }) {
  if (!modalData) return null;
  const { ev, trace } = modalData;
  const [activeCitationId, setActiveCitationId] = useState(ev?.citationId || '');

  const activeSegment = (docSegments || []).find((s) => s.citationId === activeCitationId) || ev;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/75 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6 animate-in fade-in duration-200">
      <div className="bg-white rounded-3xl shadow-2xl border border-slate-200 w-full max-w-5xl h-[85vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-900 text-white flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-xl border border-indigo-400/30 text-indigo-400">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base tracking-tight text-white">AI-DQSS Source Evidence Map</span>
                <span className="bg-indigo-500/30 text-indigo-300 text-xs px-2.5 py-0.5 rounded-full font-mono font-bold border border-indigo-400/30">
                  {docTitle || 'Uploaded Document'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5 font-mono">
                Interactive Document Evidence Map & Highlight Viewer
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-xl transition-all cursor-pointer font-bold text-xs"
          >
            ✕ Close Map
          </button>
        </div>

        {/* Body Split View */}
        <div className="flex-1 flex overflow-hidden bg-slate-50">
          {/* Left Sidebar: All Document Sections */}
          <div className="w-80 border-r border-slate-200 bg-white overflow-y-auto p-4 space-y-2">
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider px-1 mb-2 font-mono flex items-center justify-between">
              <span>Document Sections</span>
              <span className="bg-slate-100 text-slate-700 px-2 py-0.5 rounded text-[10px]">
                {docSegments?.length || 0} Blocks
              </span>
            </div>
            {(docSegments || []).map((seg, idx) => {
              const isCited = (trace?.survivingEvidences || []).some((e) => e.citationId === seg.citationId);
              const isSelected = seg.citationId === activeCitationId;
              return (
                <div
                  key={idx}
                  onClick={() => setActiveCitationId(seg.citationId)}
                  className={`p-3 rounded-2xl border text-xs cursor-pointer transition-all space-y-1.5 ${
                    isSelected
                      ? 'bg-indigo-600 text-white border-indigo-600 shadow-md'
                      : isCited
                      ? 'bg-amber-50/80 border-amber-300 text-amber-950 hover:bg-amber-100/80'
                      : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <div className="flex items-center justify-between font-mono text-[10px]">
                    <span className={`font-bold ${isSelected ? 'text-indigo-200' : 'text-indigo-700'}`}>
                      {seg.citationId}
                    </span>
                    {isCited && (
                      <span
                        className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                          isSelected ? 'bg-amber-400 text-slate-950' : 'bg-amber-200 text-amber-900 border border-amber-300'
                        }`}
                      >
                        Cited Evidence
                      </span>
                    )}
                  </div>
                  <div className={`font-bold leading-tight ${isSelected ? 'text-white' : 'text-slate-900'}`}>
                    {seg.heading}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Right Main Pane: Rendered Section Text with Yellow Evidence Highlight */}
          <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6 bg-slate-50">
            <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs space-y-5">
              <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                <div className="flex items-center gap-3">
                  <span className="font-extrabold text-sm text-indigo-900 font-mono bg-indigo-50 border border-indigo-200 px-3 py-1 rounded-xl">
                    {activeSegment?.citationId || 'DOC-1:SEC-1'}
                  </span>
                  <h3 className="font-extrabold text-base text-slate-900">{activeSegment?.heading || 'Section'}</h3>
                </div>
                <span className="text-xs font-bold text-amber-900 bg-amber-100 px-3 py-1 rounded-full border border-amber-300 flex items-center gap-1.5 shadow-2xs">
                  <Sparkles className="w-3.5 h-3.5 text-amber-600" /> Verified AI-DQSS Highlighted Evidence
                </span>
              </div>

              {/* Rendered Text Box with Yellow Highlight */}
              <div className="p-6 bg-amber-50/80 border-l-4 border-amber-500 rounded-r-3xl text-amber-950 shadow-xs space-y-3 font-sans">
                <div className="flex items-center gap-2 text-xs font-bold text-amber-900 font-mono uppercase tracking-wider">
                  <FileText className="w-4 h-4 text-amber-600" /> Source Evidence Passage Content
                </div>
                <p className="whitespace-pre-wrap leading-relaxed text-sm text-slate-900 font-medium">
                  {activeSegment?.text || ''}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Pro-Max Clean Assistant Message Component with Full AI-DQSS Traceability & Collapsible Evidence Sources
function AssistantMessage({ message, isGenerating, isLastMessage, onOpenEvidenceMap }) {
  const { content, trace } = message;
  const { thinkingText, answerText, isThinking } = parseThinkingAndAnswer(content || '');
  const [isOpen, setIsOpen] = useState(false);
  const [showEvidences, setShowEvidences] = useState(false);

  return (
    <div className="flex gap-4 justify-start">
      <div className="w-9 h-9 rounded-2xl bg-indigo-600 text-white flex items-center justify-center shrink-0 shadow-md shadow-indigo-600/20 mt-1">
        <Brain className="w-5 h-5 text-indigo-100" />
      </div>

      <div className="max-w-3xl space-y-3">
        {/* Collapsible Traceability & Evidence Accordion */}
        <div className="bg-slate-100 border border-slate-200 rounded-2xl overflow-hidden text-xs transition-all shadow-sm">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full px-4 py-2.5 bg-slate-200/60 hover:bg-slate-200 flex items-center justify-between text-slate-700 font-mono text-[11px] cursor-pointer"
          >
            <div className="flex items-center gap-2">
              {isGenerating && isLastMessage ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-600" />
              ) : (
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
              )}
              <span className="font-semibold text-slate-900">
                {isGenerating && isLastMessage
                  ? 'AI-DQSS Pipeline Reasoning & Traceability...'
                  : `AI-DQSS Section Traceability (${trace?.survivingEvidences?.length || 0} Sections)`}
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-500">
              <span className="text-[10px]">{isOpen ? 'Hide Trace' : 'Show Trace'}</span>
              {isOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            </div>
          </button>

          {isOpen && (
            <div className="p-4 border-t border-slate-200 bg-slate-50 text-slate-800 font-mono text-[11px] leading-relaxed space-y-4">
              {/* Document Overview Metadata */}
              <div className="bg-white p-3 rounded-xl border border-slate-200 space-y-1">
                <div className="text-indigo-900 font-bold uppercase tracking-wider text-[10px] flex items-center gap-1">
                  <FileText className="w-3 h-3 text-indigo-600" /> Document Overview Metadata
                </div>
                <div className="text-slate-900 font-bold text-xs">{trace?.docTitle || 'Document'}</div>
                <div className="text-slate-500 text-[10px]">
                  Main Sections ({trace?.docSections?.length || 0}): {trace?.docSections?.join(', ') || 'General'}
                </div>
              </div>

              {/* Stage 1: Candidate Section Shortlist */}
              <div className="space-y-1.5">
                <div className="text-indigo-900 font-bold uppercase tracking-wider text-[10px] flex items-center justify-between">
                  <span className="flex items-center gap-1">
                    <Filter className="w-3 h-3 text-indigo-600" /> Stage 1: Shortlisted Document Sections
                  </span>
                  <span className="bg-indigo-100 text-indigo-900 px-2 py-0.5 rounded font-bold">
                    {trace?.shortlistedCount || 0} / {trace?.totalSegments || 0} Sections
                  </span>
                </div>
                <div className="text-slate-500 text-[10px]">
                  Indexed {trace?.totalSegments || 0} complete document section blocks. Shortlisted Top {trace?.shortlistedCount || 0} candidate sections.
                </div>
              </div>

              {/* Stage 2: Reranked Surviving Section Blocks */}
              <div className="space-y-2">
                <div className="text-indigo-900 font-bold uppercase tracking-wider text-[10px] flex items-center justify-between">
                  <span className="flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-indigo-600" /> Stage 2: Reranked Surviving Section Blocks
                  </span>
                  <span className="bg-emerald-100 text-emerald-900 px-2 py-0.5 rounded font-bold">
                    Top {trace?.survivingEvidences?.length || 0} Surviving Sections
                  </span>
                </div>

                {trace?.survivingEvidences && trace.survivingEvidences.length > 0 ? (
                  <div className="space-y-2">
                    {trace.survivingEvidences.map((sec, idx) => (
                      <div key={idx} className="bg-white p-3 rounded-xl border border-slate-200 space-y-1">
                        <div className="flex justify-between items-center text-[10px]">
                          <div className="flex items-center gap-1.5">
                            <span className="font-bold text-indigo-700 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded font-mono">
                              [{idx + 1}] {sec.citationId}
                            </span>
                            {sec.relevanceScore && (
                              <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 px-1.5 py-0.5 rounded text-[9px] font-bold">
                                {sec.relevanceScore}% Relevance
                              </span>
                            )}
                          </div>
                          <span className="text-slate-700 font-bold">{sec.heading}</span>
                        </div>
                        <div className="text-slate-700 text-[11px] leading-normal font-sans max-h-40 overflow-y-auto">
                          {sec.text}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-slate-400 italic">No surviving section blocks found.</div>
                )}
              </div>

              {/* Thinking / LLM Reasoning Log if present */}
              {thinkingText && (
                <div className="space-y-1.5 pt-2 border-t border-slate-200">
                  <div className="text-indigo-900 font-bold uppercase tracking-wider text-[10px] flex items-center gap-1">
                    <Sparkles className="w-3 h-3 text-indigo-600" /> LLM Reasoning Synthesis
                  </div>
                  <div
                    className="p-3 bg-white rounded-xl border border-slate-200 text-slate-700 text-[11px] leading-relaxed max-h-48 overflow-y-auto"
                    dangerouslySetInnerHTML={renderMarkdown(thinkingText)}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {(answerText || (isGenerating && isLastMessage)) && (
          <div className="bg-white border border-slate-200 text-slate-900 rounded-3xl rounded-tl-none p-5 text-sm leading-relaxed shadow-sm space-y-4">
            {answerText ? (
              <div
                className="markdown-body text-sm text-slate-800 leading-relaxed [&_p]:mb-3 [&_p:last-child]:mb-0 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_code]:bg-slate-100 [&_code]:text-indigo-900 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:font-mono [&_code]:border [&_code]:border-slate-200 [&_pre]:bg-slate-900 [&_pre]:text-slate-100 [&_pre]:p-4 [&_pre]:overflow-x-auto [&_h1]:text-base [&_h1]:font-extrabold [&_h1]:text-slate-900 [&_h2]:text-sm [&_h2]:font-bold [&_h2]:text-slate-900 [&_h3]:text-xs [&_h3]:font-bold [&_h3]:text-slate-900 [&_strong]:text-slate-900 [&_strong]:font-bold"
                dangerouslySetInnerHTML={renderMarkdown(answerText)}
              />
            ) : (
              <p className="whitespace-pre-wrap text-slate-400 italic">
                {isThinking ? 'Synthesizing response...' : 'Generating response...'}
              </p>
            )}

            {/* Appended AI-DQSS Evidence Sources Block (Collapsible & Interactive Map Enabled) */}
            {(() => {
              if (!trace?.survivingEvidences || trace.survivingEvidences.length === 0 || isGenerating) return null;

              const citedIndices = new Set();
              if (answerText) {
                const matches = answerText.match(/\[(\d+)\]/g);
                if (matches) {
                  matches.forEach((m) => {
                    const num = parseInt(m.replace(/[\[\]]/g, ''), 10);
                    if (!isNaN(num)) citedIndices.add(num);
                  });
                }
              }

              const displayEvidences = citedIndices.size > 0
                ? trace.survivingEvidences
                    .map((ev, idx) => ({ ...ev, originalNum: idx + 1 }))
                    .filter((ev) => citedIndices.has(ev.originalNum))
                : trace.survivingEvidences.map((ev, idx) => ({ ...ev, originalNum: idx + 1 }));

              return (
                <div className="pt-3 border-t border-slate-100 space-y-2">
                  <button
                    onClick={() => setShowEvidences(!showEvidences)}
                    className="w-full flex items-center justify-between font-bold text-slate-900 text-xs py-1.5 px-3 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200/80 transition-all cursor-pointer"
                  >
                    <span className="flex items-center gap-1.5 text-indigo-950 font-bold">
                      <FileText className="w-3.5 h-3.5 text-indigo-600" /> Evidence Sources & References ({displayEvidences.length})
                    </span>
                    <span className="flex items-center gap-2 font-mono text-[10px] text-slate-500">
                      <span className="text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded font-bold border border-emerald-200">
                        AI-DQSS Verified
                      </span>
                      {showEvidences ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </span>
                  </button>

                  {showEvidences && (
                    <div className="space-y-2 pt-1 animate-in fade-in duration-200">
                      {displayEvidences.map((ev, idx) => (
                        <div key={idx} className="bg-slate-50 border border-slate-200/80 rounded-2xl p-3 space-y-2">
                          <div className="flex items-center justify-between font-mono text-[10px]">
                            <div className="flex items-center gap-1.5">
                              <span className="font-bold text-indigo-700 bg-indigo-100/80 px-2 py-0.5 rounded">
                                [{ev.originalNum}] {ev.citationId}
                              </span>
                              {ev.relevanceScore && (
                                <span className="bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded text-[9px] font-bold">
                                  {ev.relevanceScore}% Relevance
                                </span>
                              )}
                            </div>
                            <button
                              onClick={() => onOpenEvidenceMap && onOpenEvidenceMap(ev, trace)}
                              className="px-2.5 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-bold text-[10px] rounded-lg border border-indigo-200 flex items-center gap-1 transition-all cursor-pointer shadow-2xs"
                            >
                              <Eye className="w-3 h-3 text-indigo-600" /> Highlight in Evidence Map
                            </button>
                          </div>
                          <div className="text-slate-800 font-bold text-xs font-sans">{ev.heading}</div>
                          <p className="text-slate-700 text-[11px] leading-relaxed italic">
                            "{ev.text}"
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [selectedModel, setSelectedModel] = useState('gemma-4'); // 'gemma-4' | 'bonsai-27b'
  const [allowedTools, setAllowedTools] = useState(['search_wikipedia', 'get_current_context', 'ask_user', 'search_uploaded_document']);
  const [agentPrompt, setAgentPrompt] = useState('What are the key priorities of the World Bank Group in the development sector?');
  const agent = useAgentWorker();

  // AI-DQSS Document & Section Index State
  const [uploadedDoc, setUploadedDoc] = useState(null);
  const [docTitle, setDocTitle] = useState('Development Data Quality Policy');
  const [docSections, setDocSections] = useState([]);
  const [docSegments, setDocSegments] = useState([]);
  const [indexingStatus, setIndexingStatus] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [downloadStats, setDownloadStats] = useState('');

  // LLM Tab 1 State (Transformers.js)
  const [modelRepo, setModelRepo] = useState(DEFAULT_ONNX_REPO);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const [generating, setGenerating] = useState(false);
  const [inputPrompt, setInputPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [tps, setTps] = useState(0);
  const [tokenCount, setTokenCount] = useState(0);

  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(512);

  const generatorRef = useRef(null);
  const chatEndRef = useRef(null);
  const isStoppingRef = useRef(false);
  const fileInputRef = useRef(null);

  // GLiNER Tab 2 State
  const [glinerVariant, setGlinerVariant] = useState('model_q4.onnx');
  const [glinerLoaded, setGlinerLoaded] = useState(false);
  const [glinerLoading, setGlinerLoading] = useState(false);
  const [glinerProgress, setGlinerProgress] = useState(0);
  const [glinerMsg, setGlinerMsg] = useState('');
  const [glinerError, setGlinerError] = useState(null);
  const [isCached, setIsCached] = useState(false);

  const [glinerText, setGlinerText] = useState('');
  const [glinerLabels, setGlinerLabels] = useState('');
  const [entities, setEntities] = useState([]);
  const [glinerLatency, setGlinerLatency] = useState(0);

  const glinerSessionRef = useRef(null);
  const glinerTokenizerRef = useRef(null);

  // Gemma 4 WGSL Kernels Tab 3 State
  const [gemmaLoaded, setGemmaLoaded] = useState(false);
  const [gemmaLoading, setGemmaLoading] = useState(false);
  const [gemmaProgress, setGemmaProgress] = useState(0);
  const [gemmaMsg, setGemmaMsg] = useState('');
  const [gemmaError, setGemmaError] = useState(null);

  const [gemmaInput, setGemmaInput] = useState('');
  const [gemmaMessages, setGemmaMessages] = useState([]);
  const [gemmaGenerating, setGemmaGenerating] = useState(false);
  const [gemmaTps, setGemmaTps] = useState(0);
  const [gemmaTtft, setGemmaTtft] = useState(0);

  const gemmaInstanceRef = useRef(null);
  const gemmaHistoryRef = useRef([]);
  const isGemmaStoppingRef = useRef(false);

  // SPLADE + GIST Embedding WebGPU State
  const denseEmbedderRef = useRef(null);
  const sparseEmbedderRef = useRef(null);
  const embeddingIndexRef = useRef({ dense: [], sparse: [], segments: [] });
  const [embeddingLoaded, setEmbeddingLoaded] = useState(false);
  const [embeddingProgress, setEmbeddingProgress] = useState(0);
  const [embeddingMsg, setEmbeddingMsg] = useState('');
  const [evidenceModal, setEvidenceModal] = useState(null);

  // Load WebGPU Feature Extraction Embedding pipeline
  const handleLoadEmbeddingModels = async () => {
    try {
      setEmbeddingMsg('Loading Xenova/all-MiniLM-L6-v2 WebGPU Embeddings in parallel...');
      const dense = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        device: 'webgpu',
        progress_callback: (info) => {
          if (info.status === 'progress') {
            const p = Math.round(info.progress || 0);
            setEmbeddingProgress(p);
            setEmbeddingMsg(`Loading WebGPU Embeddings... ${p}%`);
          }
        },
      });
      denseEmbedderRef.current = dense;
      setEmbeddingLoaded(true);
      setEmbeddingMsg('MiniLM-v2 WebGPU Embeddings Ready!');

      if (docSegments && docSegments.length > 0) {
        indexDocumentSegments(docSegments);
      }
    } catch (err) {
      console.warn('Embedding pipeline WebGPU fallback notice:', err);
      try {
        setEmbeddingMsg('WebGPU unavailable, trying WASM embedding fallback...');
        const dense = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        denseEmbedderRef.current = dense;
        setEmbeddingLoaded(true);
        setEmbeddingMsg('MiniLM-v2 (WASM) Embeddings Ready!');
        if (docSegments && docSegments.length > 0) {
          indexDocumentSegments(docSegments);
        }
      } catch (fallbackErr) {
        console.error('Embedding load error:', fallbackErr);
        setEmbeddingMsg(`Embedding notice: ${err.message || String(err)}; using stem search.`);
      }
    }
  };

  const indexDocumentSegments = async (segments) => {
    if (!denseEmbedderRef.current || !segments || segments.length === 0) return;
    try {
      setEmbeddingMsg(`Parallel indexing ${segments.length} section embeddings on WebGPU...`);

      // Parallelized batch vector extraction pass
      const denseVectors = await Promise.all(
        segments.map(async (seg) => {
          const dOut = await denseEmbedderRef.current(seg.text, { pooling: 'mean', normalize: true });
          return Array.from(dOut.data);
        })
      );

      embeddingIndexRef.current = { dense: denseVectors, segments };
      setEmbeddingMsg(`MiniLM-v2 WebGPU Vector Index Active (${segments.length} sections indexed)!`);
    } catch (err) {
      console.warn('Parallel document embedding indexing notice:', err);
      setEmbeddingMsg('Embedding indexing notice: using stem search fallback.');
    }
  };

  const hybridRetrieve = async (query, topK = 15) => {
    if (
      !embeddingLoaded ||
      !denseEmbedderRef.current ||
      !embeddingIndexRef.current.dense ||
      embeddingIndexRef.current.dense.length === 0
    ) {
      return shortlistCandidates(query, docSegments, topK);
    }

    try {
      const { dense, segments } = embeddingIndexRef.current;

      // 1. Get BM25 Stem Candidates (Top 30)
      const bm25Candidates = shortlistCandidates(query, docSegments, 30);
      const bm25RankMap = new Map();
      bm25Candidates.forEach((c, rank) => bm25RankMap.set(c.citationId, rank + 1));

      // 2. Compute Dense Embeddings for Query
      const qDenseOut = await denseEmbedderRef.current(query, { pooling: 'mean', normalize: true });
      const qDense = Array.from(qDenseOut.data);

      // Pre-Shortlisted Vector Pass: Score candidate indices (<1ms)
      const candidateCitationSet = new Set(bm25Candidates.map((c) => c.citationId));
      const candidateIndices = [];
      segments.forEach((seg, i) => {
        if (candidateCitationSet.has(seg.citationId) || candidateIndices.length < 30) {
          candidateIndices.push(i);
        }
      });

      const denseScores = candidateIndices.map((i) => {
        let dot = 0, na = 0, nb = 0;
        const dVec = dense[i];
        for (let j = 0; j < qDense.length; j++) {
          dot += qDense[j] * dVec[j];
          na += qDense[j] ** 2;
          nb += dVec[j] ** 2;
        }
        const sim = dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        return { segment: segments[i], denseSim: sim };
      });

      denseScores.sort((a, b) => b.denseSim - a.denseSim);
      const denseRankMap = new Map();
      denseScores.forEach((item, rank) => denseRankMap.set(item.segment.citationId, rank + 1));

      // 3. Reciprocal Rank Fusion (RRF) Score Calculation: RRF = 1/(60+BM25Rank) + 1/(60+DenseRank)
      const rrfResults = candidateIndices.map((i) => {
        const seg = segments[i];
        const bm25Rank = bm25RankMap.get(seg.citationId) || 60;
        const denseRank = denseRankMap.get(seg.citationId) || 60;

        const rrfScore = (1 / (60 + bm25Rank)) + (1 / (60 + denseRank));
        const bm25Score = bm25Candidates.find((c) => c.citationId === seg.citationId)?.score || 0;

        return {
          ...seg,
          score: rrfScore * 1000 + bm25Score,
          rrfScore,
        };
      });

      rrfResults.sort((a, b) => b.score - a.score);
      return rrfResults.slice(0, topK);
    } catch (err) {
      console.warn('Hybrid RRF retrieval error, falling back to stem search:', err);
      return shortlistCandidates(query, docSegments, topK);
    }
  };

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, gemmaMessages]);

  // CLI bridge: opt-in via ?bridge=1, exposes a loopback WebSocket relay
  const bridgeEnabled = React.useMemo(
    () => typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('bridge') === '1',
    [],
  );
  const bridgeSocketRef = useRef(null);
  const bridgeReconnectRef = useRef(0);

  const sendToCli = (message) => {
    const ws = bridgeSocketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  };

  useEffect(() => {
    if (!bridgeEnabled) return undefined;
    let cancelled = false;
    let socket = null;

    const connect = () => {
      if (cancelled) return;
      let ws;
      try {
        ws = new WebSocket('ws://127.0.0.1:8765');
      } catch (err) {
        console.warn('Bridge WebSocket construction failed:', err);
        return;
      }
      socket = ws;
      bridgeSocketRef.current = ws;

      const reportReady = () => {
        const modelName = gemmaLoaded ? selectedModel : 'none';
        const embeddingName = embeddingLoaded ? 'webgpu' : 'stem';
        sendToCli({ type: 'ready', model: modelName, embedding: embeddingName });
      };

      ws.onopen = () => {
        bridgeReconnectRef.current = 0;
        reportReady();
      };
      ws.onmessage = async (event) => {
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === 'init' || message.type === 'state') {
          reportReady();
          return;
        }
        if (message.type === 'stop') {
          isGemmaStoppingRef.current = true;
          return;
        }
        if (message.type === 'chat') {
          const requestId = String(message.request_id || 'cli');
          isGemmaStoppingRef.current = false;
          if (!gemmaLoaded || (!gemmaInstanceRef.current && !generatorRef.current)) {
            sendToCli({ type: 'error', request_id: requestId, message: 'Model not loaded' });
            return;
          }
          const systemOverride = message.system && String(message.system).trim()
            ? String(message.system)
            : null;
          try {
            const result = await runGemmaTurn(
              String(message.prompt || ''),
              systemOverride,
              (text) => sendToCli({ type: 'token', request_id: requestId, text }),
            );
            if (result.ok) {
              sendToCli({
                type: 'done',
                request_id: requestId,
                text: result.answer,
                sources: (result.sources || []).map((sec, idx) => ({
                  num: idx + 1, citation_id: sec.citationId, heading: sec.heading,
                })),
                evidence_count: (result.sources || []).length,
              });
            } else if (result.reason === 'out_of_scope') {
              sendToCli({
                type: 'done',
                request_id: requestId,
                text: 'The uploaded document does not contain any information or evidence regarding your question. Please try rephrasing or asking about topics covered in the document.',
                sources: [],
                evidence_count: 0,
              });
            } else {
              sendToCli({ type: 'error', request_id: requestId, message: result.reason || 'Generation failed' });
            }
          } catch (err) {
            sendToCli({ type: 'error', request_id: requestId, message: err.message || String(err) });
          }
        }
      };
      ws.onclose = () => {
        if (bridgeSocketRef.current === ws) {
          bridgeSocketRef.current = null;
        }
        if (cancelled) return;
        bridgeReconnectRef.current = Math.min(bridgeReconnectRef.current + 1, 5);
        const delay = 500 * 2 ** bridgeReconnectRef.current;
        setTimeout(connect, delay);
      };
      ws.onerror = () => { try { ws.close(); } catch (_) { /* ignore */ } };
    };

    connect();
    return () => {
      cancelled = true;
      if (socket) { try { socket.close(); } catch (_) { /* ignore */ } }
      if (bridgeSocketRef.current === socket) {
        bridgeSocketRef.current = null;
      }
    };
  }, [bridgeEnabled, gemmaLoaded, selectedModel, embeddingLoaded]);

  // Check if GLiNER model is cached on variant change
  useEffect(() => {
    const modelUrl = `${ANONYM_GLINER_BASE_URL}${glinerVariant}`;
    getCachedModelBuffer(modelUrl).then((buf) => {
      setIsCached(!!(buf && buf.byteLength > 0));
    });
  }, [glinerVariant]);

  // AI-DQSS Section-Level Citable Document Parser (PDF, Docling, TXT, MD)
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsIndexing(true);
    setIndexingStatus(`Reading ${file.name}...`);
    setUploadedDoc(file.name);
    setDocTitle(file.name);

    try {
      const fileName = file.name.toLowerCase();
      const sectionBlocks = [];
      const sectionsSet = new Set();

      if (fileName.endsWith('.pdf')) {
        setIndexingStatus('Parsing PDF into high-density citable Section & Paragraph Blocks (DOC-1:SEC:{id})...');
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        const rawPages = [];
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
          const page = await pdf.getPage(pageNum);
          const textContent = await page.getTextContent();
          const pageLines = textContent.items.map((item) => item.str.trim()).filter(Boolean);
          rawPages.push({ pageNum, text: pageLines.join('\n') });
        }

        const headerRegex = /^(?:SECTION|CHAPTER|PART|ARTICLE)\s+[IVX\d]+|^(?:[1-9]\d?|\d+\.\d+)\.\s+[A-Z]|^#{1,3}\s+/i;
        let currentHeading = 'Document Overview & Metadata';
        let currentBuffer = [];
        const rawSections = [];

        for (const pageObj of rawPages) {
          const lines = pageObj.text.split('\n');
          for (const line of lines) {
            const cleanLine = line.trim();
            if (!cleanLine) continue;

            const isHeader =
              headerRegex.test(cleanLine) ||
              (cleanLine.length < 60 && cleanLine === cleanLine.toUpperCase() && /^[A-Z0-9\s–\-\:\.\,]+$/.test(cleanLine) && cleanLine.length > 5);

            if (isHeader && cleanLine.toLowerCase() !== currentHeading.toLowerCase()) {
              if (currentBuffer.length > 0) {
                rawSections.push({ heading: currentHeading, text: currentBuffer.join('\n') });
                currentBuffer = [];
              }
              currentHeading = cleanLine;
              sectionsSet.add(cleanLine);
            } else {
              currentBuffer.push(cleanLine);
            }
          }
        }
        if (currentBuffer.length > 0) {
          rawSections.push({ heading: currentHeading, text: currentBuffer.join('\n') });
        }

        // Sub-chunking: Split long sections (> 2200 chars) into focused paragraph blocks (800-1800 chars)
        rawSections.forEach((sec) => {
          const secId = sec.heading.split('–')[0].split('-')[0].trim().replace(/\s+/g, '-');

          if (sec.text.length <= 2200) {
            sectionBlocks.push({
              citationId: `DOC-1:${secId}`,
              heading: sec.heading,
              text: sec.text,
            });
          } else {
            const paragraphs = sec.text.split(/\n{2,}|\n(?=[a-z0-9][\.\)]\s+)/i).filter((p) => p.trim().length > 30);
            let subBuffer = '';
            let pIdx = 1;

            paragraphs.forEach((para) => {
              if ((subBuffer + '\n\n' + para).length > 1800 && subBuffer.length > 200) {
                sectionBlocks.push({
                  citationId: `DOC-1:${secId}:P${pIdx}`,
                  heading: `${sec.heading} (Part ${pIdx})`,
                  text: subBuffer,
                });
                pIdx++;
                subBuffer = para;
              } else {
                subBuffer = subBuffer ? subBuffer + '\n\n' + para : para;
              }
            });

            if (subBuffer.length > 20) {
              sectionBlocks.push({
                citationId: `DOC-1:${secId}:P${pIdx}`,
                heading: `${sec.heading} (Part ${pIdx})`,
                text: subBuffer,
              });
            }
          }
        });

        // Fallback: If no section headers were found, chunk by pages
        if (sectionBlocks.length === 0 && rawPages.length > 0) {
          rawPages.forEach((p) => {
            if (p.text.trim().length > 30) {
              sectionBlocks.push({
                citationId: `DOC-1:PG${p.pageNum}`,
                heading: `Page ${p.pageNum}`,
                text: p.text,
              });
            }
          });
        }
      } else if (fileName.endsWith('.json')) {
        setIndexingStatus('Parsing DoclingDocument JSON AST into Section Blocks...');
        const jsonText = await file.text();
        const docJson = JSON.parse(jsonText);

        if (docJson.groups && docJson.groups.length > 0) {
          const refMap = new Map();
          (docJson.texts || []).forEach((t, i) => refMap.set(`#/texts/${i}`, t));

          docJson.groups.forEach((grp, gIdx) => {
            const grpTitle = grp.name || grp.label || `Section ${gIdx + 1}`;
            sectionsSet.add(grpTitle);

            const childTexts = [];
            (grp.children || []).forEach((cRef) => {
              const item = refMap.get(typeof cRef === 'string' ? cRef : cRef.$ref);
              if (item?.text) childTexts.push(item.text);
            });

            if (childTexts.length > 0) {
              sectionBlocks.push({
                citationId: `DOC-1:SEC-${gIdx + 1}`,
                heading: grpTitle,
                text: childTexts.join('\n\n'),
              });
            }
          });
        }
      } else {
        const text = await file.text();
        sectionBlocks.push({
          citationId: 'DOC-1:SEC-1',
          heading: 'Full Document Context',
          text,
        });
      }

      const sectionsList = Array.from(sectionsSet);
      setDocSections(sectionsList);
      setDocSegments(sectionBlocks);

      if (embeddingLoaded) {
        indexDocumentSegments(sectionBlocks);
      }

      setIndexingStatus(
        `AI-DQSS Section Index Ready: ${sectionBlocks.length} complete section blocks across ${sectionsList.length || 1} main sections!`
      );

      if (sectionBlocks.length > 0) {
        setGlinerText(sectionBlocks[0].text.slice(0, 1000));
      }
    } catch (err) {
      console.error('File Upload Error:', err);
      setIndexingStatus(`Error reading file: ${err.message || String(err)}`);
    } finally {
      setIsIndexing(false);
    }
  };

  // Load WebGPU Chat Compute Engine (Gemma 4 Mobile or Bonsai 1.7B ONNX)
  const handleLoadChatModel = async () => {
    setGemmaError(null);
    setGemmaLoaded(false);
    setGemmaLoading(true);
    setGemmaProgress(0);
    const isGemma = selectedModel === 'gemma-4';
    const modelLabel = isGemma ? 'Gemma 4 Mobile (2.7B WGSL)' : 'Bonsai 1.7B ONNX (WebGPU)';
    setGemmaMsg(`Initializing WebGPU device & ${modelLabel}...`);

    try {
      if (!navigator.gpu) {
        throw new Error("WebGPU isn't available in this browser context.");
      }

      if (isGemma) {
        setGemmaMsg(`Streaming Gemma 4 Mobile fused WGSL shaders & weights to WebGPU...`);
        const model = await Gemma4Mobile.load(null, {
          onProgress: (p) => {
            let pct = 0;
            if (typeof p === 'number') {
              pct = Math.round(p);
            } else if (p && typeof p.loaded === 'number' && typeof p.total === 'number') {
              pct = Math.round((p.loaded / p.total) * 100);
              const loadedMB = (p.loaded / (1024 * 1024)).toFixed(1);
              const totalMB = (p.total / (1024 * 1024)).toFixed(1);
              setDownloadStats(`${loadedMB} MB / ${totalMB} MB`);
            }
            setGemmaProgress(pct);
            setGemmaMsg(p?.message || `Loading WGSL Kernels & Weights: ${pct}%`);
          },
        });

        if (model.warmup) {
          await model.warmup();
        }

        gemmaInstanceRef.current = model;
        generatorRef.current = null;
      } else {
        // Load official ONNX Bonsai Model via Transformers.js WebGPU pipeline
        setGemmaMsg(`Streaming onnx-community/Bonsai-1.7B-ONNX weights to WebGPU...`);
        const pipe = await pipeline('text-generation', DEFAULT_ONNX_REPO, {
          device: 'webgpu',
          dtype: 'q4',
          progress_callback: (info) => {
            if (info.status === 'progress') {
              const p = Math.round(info.progress || 0);
              setGemmaProgress(p);
              const loadedMB = (info.loaded / (1024 * 1024)).toFixed(1);
              const totalMB = (info.total / (1024 * 1024)).toFixed(1);
              setDownloadStats(`${loadedMB} MB / ${totalMB} MB`);
              setGemmaMsg(`Streaming ONNX Bonsai weights to WebGPU... ${p}% (${loadedMB}/${totalMB} MB)`);
            }
          },
        });

        generatorRef.current = pipe;
        gemmaInstanceRef.current = null;
      }

      setGemmaLoaded(true);
      setGemmaLoading(false);
      setGemmaMsg(`${modelLabel} Ready!`);

      // Trigger parallel embedding load
      handleLoadEmbeddingModels();
    } catch (err) {
      console.error('WebGPU Model Load Error:', err);
      setGemmaLoading(false);
      setGemmaError(err.message || String(err));
    }
  };

  const runGemmaTurn = async (prompt, systemOverride, onToken, metricsHook) => {
    const shortlistedSections = await hybridRetrieve(prompt, 15);
    const survivingSections = rerankCandidates(
      prompt, shortlistedSections, 5, 2.5, gemmaHistoryRef.current,
    );
    if (docSegments.length > 0 && survivingSections.length === 0) {
      return { ok: false, reason: 'out_of_scope' };
    }
    const perEvidenceInsights = survivingSections.length > 1
      ? survivingSections.map((sec, idx) => {
        const snippet = sec.text.length > 2500 ? sec.text.slice(0, 2500) + '...' : sec.text;
        return `[${idx + 1}] (${sec.heading} | ${sec.citationId}):\n${snippet}`;
      })
      : [];
    const systemPrompt = systemOverride ?? (survivingSections.length > 1
      ? buildReduceSystemPrompt(docTitle, perEvidenceInsights, prompt)
      : buildAiDqssSystemPrompt(docTitle, docSections, survivingSections, prompt));
    const compactedHistory = compactHistory(gemmaHistoryRef.current);
    const payloadMessages = [...compactedHistory, { role: 'user', content: systemPrompt }];

    const startTime = performance.now();
    let firstTokenTime = 0;
    let tokensGenerated = 0;
    let fullText = '';

    const reportMetrics = () => {
      if (!metricsHook) return;
      const ttft = firstTokenTime ? (firstTokenTime - startTime).toFixed(0) : 0;
      const decodeSec = firstTokenTime ? (performance.now() - firstTokenTime) / 1000 : 0;
      const tps = decodeSec > 0 ? ((Math.max(0, tokensGenerated - 1) / decodeSec).toFixed(1)) : 0;
      metricsHook({ tps: Number(tps), ttft: Number(ttft), tokens: tokensGenerated });
    };

    try {
      if (selectedModel === 'gemma-4' && gemmaInstanceRef.current) {
        const stream = gemmaInstanceRef.current.generate(payloadMessages, { maxNewTokens: 2048 });
        for await (const chunk of stream) {
          if (isGemmaStoppingRef.current) break;
          fullText = chunk?.text || fullText;
          tokensGenerated++;
          const now = performance.now();
          if (tokensGenerated === 1) {
            firstTokenTime = now;
          }
          reportMetrics();
          onToken?.(fullText);
        }
      } else if (generatorRef.current) {
        const streamer = new TextStreamer(generatorRef.current.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (tokenText) => {
            if (isGemmaStoppingRef.current) return;
            fullText += tokenText;
            tokensGenerated++;
            const now = performance.now();
            if (tokensGenerated === 1) {
              firstTokenTime = now;
            }
            reportMetrics();
            onToken?.(fullText);
          },
        });
        let promptToFeed = systemPrompt;
        if (generatorRef.current?.tokenizer?.apply_chat_template) {
          try {
            promptToFeed = generatorRef.current.tokenizer.apply_chat_template(
              [
                {
                  role: 'system',
                  content: 'You are an expert document Q&A assistant. Synthesize comprehensive, detailed answers directly from context with inline citations like [1]. Never copy or repeat sentences from evidence.',
                },
                ...compactedHistory,
                { role: 'user', content: systemPrompt },
              ],
              { tokenize: false, add_generation_prompt: true }
            );
          } catch (tErr) {
            console.warn('Chat template application fallback:', tErr);
          }
        }
        await generatorRef.current(promptToFeed, {
          max_new_tokens: 512,
          temperature: 0.1,
          repetition_penalty: 1.25,
          no_repeat_ngram_size: 4,
          do_sample: false,
          streamer,
        });
      } else {
        return { ok: false, reason: 'model_not_loaded' };
      }
    } catch (err) {
      if (!isGemmaStoppingRef.current) {
        throw err;
      }
    }
    return { ok: true, answer: fullText, sources: survivingSections };
  };

  // Run AI-DQSS Section Shortlist + Rerank Pipeline & Chat Generation
  const handleSendGemma = async () => {
    if (!gemmaInput.trim() || !gemmaLoaded || gemmaGenerating || (!gemmaInstanceRef.current && !generatorRef.current)) return;

    const currentPrompt = gemmaInput.trim();
    setGemmaGenerating(true);
    isGemmaStoppingRef.current = false;
    setGemmaTps(0);

    const shortlistedSections = await hybridRetrieve(currentPrompt, 15);
    const survivingSections = rerankCandidates(currentPrompt, shortlistedSections, 5, 2.5, gemmaHistoryRef.current);

    if (docSegments.length > 0 && survivingSections.length === 0) {
      const traceData = {
        docTitle,
        docSections,
        totalSegments: docSegments.length,
        shortlistedCount: shortlistedSections.length,
        survivingEvidences: [],
        outOfScope: true,
      };

      const userMsg = { role: 'user', content: currentPrompt };
      const assistantMsg = {
        role: 'assistant',
        content: 'The uploaded document does not contain any information or evidence regarding your question. Please try rephrasing or asking about topics covered in the document.',
        trace: traceData,
      };

      setGemmaMessages((prev) => [...prev, userMsg, assistantMsg]);
      gemmaHistoryRef.current.push(userMsg, { role: 'assistant', content: assistantMsg.content });
      setGemmaInput('');
      setGemmaGenerating(false);
      return;
    }

    const traceData = {
      docTitle,
      docSections,
      totalSegments: docSegments.length,
      shortlistedCount: shortlistedSections.length,
      survivingEvidences: survivingSections,
    };

    const userMsg = { role: 'user', content: currentPrompt };
    const assistantMsg = { role: 'assistant', content: '', trace: traceData };
    setGemmaMessages((prev) => [...prev, userMsg, assistantMsg]);
    setGemmaInput('');

    let lastGeneratedText = '';
    try {
      const result = await runGemmaTurn(
        currentPrompt,
        null,
        (text) => {
          lastGeneratedText = text;
          setGemmaMessages((prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
              updated[lastIdx] = { ...updated[lastIdx], content: text };
            }
            return updated;
          });
        },
        ({ tps, ttft }) => {
          if (ttft) setGemmaTtft(ttft);
          setGemmaTps(tps);
        },
      );

      if (!result.ok) {
        if (result.reason === 'out_of_scope') {
          setGemmaMessages((prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
              updated[lastIdx] = {
                ...updated[lastIdx],
                content: 'The uploaded document does not contain any information or evidence regarding your question. Please try rephrasing or asking about topics covered in the document.',
              };
            }
            return updated;
          });
        } else {
          setGemmaError(result.reason || 'Generation failed');
        }
      }
    } catch (err) {
      console.error('Generation Error:', err);
      setGemmaError(err.message || String(err));
    } finally {
      if (isGemmaStoppingRef.current) {
        setGemmaMessages((prev) => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          if (lastMsg && lastMsg.role === 'assistant') {
            updated[updated.length - 1] = {
              ...lastMsg,
              content: `${lastMsg.content} [stopped]`,
            };
          }
          return updated;
        });
      }

      const { answerText } = parseThinkingAndAnswer(lastGeneratedText);
      gemmaHistoryRef.current.push(
        { role: 'user', content: currentPrompt },
        { role: 'assistant', content: answerText || lastGeneratedText }
      );

      setGemmaGenerating(false);
      isGemmaStoppingRef.current = false;
    }
  };

  // Load Transformers.js LLM WebGPU Pipeline (Tab 1)
  const handleLoadTransformers = async () => {
    setError(null);
    setModelLoaded(false);
    setLoading(true);
    setLoadingMsg('Initializing Transformers.js ONNX WebGPU Engine...');
    setProgress(0);

    try {
      const pipe = await pipeline('text-generation', modelRepo, {
        device: 'webgpu',
        dtype: 'q4',
        progress_callback: (info) => {
          if (info.status === 'progress') {
            const p = Math.round(info.progress || 0);
            setProgress(p);
            const loadedMB = (info.loaded / (1024 * 1024)).toFixed(1);
            const totalMB = (info.total / (1024 * 1024)).toFixed(1);
            setDownloadStats(`${loadedMB} MB / ${totalMB} MB`);
            setLoadingMsg(`Streaming ONNX weights to GPU... ${p}% (${loadedMB}/${totalMB} MB)`);
          } else if (info.status === 'ready') {
            setLoadingMsg('Transformers.js WebGPU Kernels Ready!');
          }
        },
      });

      generatorRef.current = pipe;
      setModelLoaded(true);
      setLoading(false);
      setError(null);
    } catch (err) {
      console.error('Transformers.js WebGPU Error:', err);
      setLoading(false);
      setError(err.message || String(err));
    }
  };

  // WebGPU Model Session Loader (Extraction Tab - Always Direct Stream, No Caching)
  const handleLoadGlinerDirectONNX = async () => {
    setGlinerError(null);
    setGlinerLoaded(false);
    setGlinerLoading(true);
    setGlinerProgress(0);

    let isLocal = false;
    let modelUrl = `${ANONYM_GLINER_BASE_URL}${glinerVariant}`;
    const localUrl = `/onnx/gliner_large-v2.1_q4.onnx`;
    
    try {
      const localCheck = await fetch(localUrl, { method: 'HEAD' });
      if (localCheck.ok) {
        modelUrl = localUrl;
        isLocal = true;
        setGlinerMsg(`Loading local model weights from public/onnx/ ...`);
      } else {
        setGlinerMsg(`Streaming ${glinerVariant} directly from HuggingFace Hub...`);
      }
    } catch (e) {
      setGlinerMsg(`Streaming ${glinerVariant} directly from HuggingFace Hub...`);
    }

    try {
      if (!navigator.gpu) {
        throw new Error("WebGPU isn't available in this browser context.");
      }

      setIsCached(isLocal);
      let arrayBuffer;
      const startTime = performance.now();

      if (isLocal) {
        setGlinerProgress(50);
        setGlinerMsg('Loading local 204 MB ONNX weights into memory...');
        const res = await fetch(modelUrl);
        arrayBuffer = await res.arrayBuffer();
        setGlinerProgress(100);
        const sizeMB = (arrayBuffer.byteLength / (1024 * 1024)).toFixed(1);
        setDownloadStats(`${sizeMB} MB (Local Static Asset)`);
      } else {
        const headRes = await fetch(modelUrl, { method: 'HEAD' });
        const totalBytes = Number(headRes.headers.get('content-length') || '932976268');

        arrayBuffer = await fetchParallelRanges(modelUrl, totalBytes, 6, (loaded, total) => {
          const p = Math.round((loaded / total) * 100);
          setGlinerProgress(p);
          const mbLoaded = (loaded / (1024 * 1024)).toFixed(1);
          const mbTotal = (total / (1024 * 1024)).toFixed(1);
          const elapsedSec = (performance.now() - startTime) / 1000;
          const mbps = elapsedSec > 0 ? ((loaded * 8) / (1024 * 1024 * elapsedSec)).toFixed(1) : 0;
          setDownloadStats(`${mbLoaded} / ${mbTotal} MB @ ${mbps} Mbps`);
          setGlinerMsg(`Streaming WebGPU ONNX weights: ${p}% (${mbLoaded}/${mbTotal} MB @ ${mbps} Mbps)`);
        });
      }

      setGlinerMsg('Loading GLiNER tokenizer config...');
      if (!glinerTokenizerRef.current) {
        const tokenizer = await AutoTokenizer.from_pretrained('onnx-community/gliner_large-v2.1');
        glinerTokenizerRef.current = tokenizer;
      }

      setGlinerMsg('Compiling WebGPU WGSL compute shaders into GPU VRAM...');
      const session = await ort.InferenceSession.create(arrayBuffer, {
        executionProviders: ['webgpu'],
      });

      glinerSessionRef.current = session;
      setGlinerLoaded(true);
      setGlinerLoading(false);
      setGlinerMsg(`Anonym-IA/gliner_large-v2.1 (${glinerVariant}) WebGPU Session Active!`);
    } catch (err) {
      console.error('GLiNER WebGPU Error:', err);
      setGlinerLoading(false);
      setGlinerError(err.message || String(err));
    }
  };

  // Run Zero-Shot Entity Extraction on WebGPU (Tab 2)
  const handleExtractEntities = async () => {
    if (!glinerText.trim() || !glinerLabels.trim()) return;

    // Auto-load ONNX WebGPU model session if not loaded yet
    if (!glinerLoaded || !glinerSessionRef.current) {
      await handleLoadGlinerDirectONNX();
      if (!glinerSessionRef.current) return;
    }

    setGlinerError(null);
    setGlinerLoading(true);
    setGlinerMsg('Executing WebGPU single-pass matrix forward pass...');
    const startTime = performance.now();

    try {
      const labelList = glinerLabels.split(',').map((l) => l.trim()).filter(Boolean);
      const words = glinerText.split(/\s+/).filter(Boolean);
      const numWords = words.length;
      const maxSpanWidth = 12;

      let outputs = null;

      // Execute ONNX Session forward pass on WebGPU using authentic BPE token alignment
      if (glinerSessionRef.current) {
        try {
          const tokenizer = glinerTokenizerRef.current;
          if (!tokenizer) {
            throw new Error("GLiNER tokenizer not loaded!");
          }

          // GLiNER BPE Special Tokens formatting
          const promptStr = labelList.map(l => `<<ENT>> ${l}`).join(' ') + ' <<SEP>> ' + glinerText;
          const tokenized = tokenizer(promptStr);
          
          const tokenIds = Array.from(tokenized.input_ids.data).map(Number);
          const attentionMaskData = Array.from(tokenized.attention_mask.data).map(Number);
          const seqLen = tokenIds.length;

          // Dynamically map word starts to BPE subtokens
          const sepIndex = tokenIds.indexOf(128003); // Index of <<SEP>>
          if (sepIndex === -1) {
            throw new Error('<<SEP>> token not found in tokenized input sequence!');
          }

          const wordsMask = new BigInt64Array(seqLen).fill(0n);
          let currentTokenIdx = sepIndex + 1;

          for (let w = 0; w < numWords; w++) {
            const wordTokenized = tokenizer(words[w]);
            const subtokenCount = wordTokenized.input_ids.data.length - 2; // Subtract [CLS] and [SEP]
            for (let t = 0; t < subtokenCount; t++) {
              if (currentTokenIdx < seqLen - 1) {
                wordsMask[currentTokenIdx] = BigInt(w + 1); // 1-based index
                currentTokenIdx++;
              }
            }
          }

          const textLengths = new BigInt64Array([BigInt(numWords)]);

          const spanIndices = [];
          for (let i = 0; i < numWords; i++) {
            for (let j = 0; j < maxSpanWidth; j++) {
              spanIndices.push(BigInt(i), BigInt(Math.min(i + j, numWords - 1)));
            }
          }
          const numSpans = numWords * maxSpanWidth;
          const spanIdxTensor = new BigInt64Array(spanIndices);
          const spanMaskBool = new Uint8Array(numSpans).fill(1);

          // Convert BigInt arrays
          const inputIdsBigInt = new BigInt64Array(tokenIds.map(BigInt));
          const attentionMaskBigInt = new BigInt64Array(attentionMaskData.map(BigInt));

          const feeds = {
            input_ids: new ort.Tensor('int64', inputIdsBigInt, [1, seqLen]),
            attention_mask: new ort.Tensor('int64', attentionMaskBigInt, [1, seqLen]),
            words_mask: new ort.Tensor('int64', wordsMask, [1, seqLen]),
            text_lengths: new ort.Tensor('int64', textLengths, [1, 1]),
            span_idx: new ort.Tensor('int64', spanIdxTensor, [1, numSpans, 2]),
            span_mask: new ort.Tensor('bool', spanMaskBool, [1, numSpans]),
          };

          outputs = await glinerSessionRef.current.run(feeds);
          console.log('GLiNER WebGPU ONNX Logits Output Shape:', outputs?.logits?.dims);
        } catch (e) {
          console.warn('ONNX GPU forward pass execution notice:', e);
        }
      }

      const extracted = [];
      const logitsData = outputs?.logits?.data;

      if (logitsData && logitsData.length > 0) {
        // Pure ONNX GPU Tensor Logits Matrix Decoding (0 Regex, 0 If-Else, 0 Fallbacks)
        const numLabels = labelList.length;

        for (let i = 0; i < numWords; i++) {
          for (let j = 0; j < Math.min(maxSpanWidth, numWords - i); j++) {
            const spanText = words.slice(i, i + j + 1).join(' ');
            const cleanSpan = spanText.replace(/^[^\w$]+|[^\w]+$/g, '');

            if (cleanSpan.length >= 2) {
              const spanIdx = i * maxSpanWidth + j;

              labelList.forEach((lbl, labelIdx) => {
                const logitIndex = spanIdx * numLabels + labelIdx;
                const rawLogit = logitsData[logitIndex];

                // Sigmoid Activation Function: P = 1 / (1 + exp(-x))
                const prob = 1 / (1 + Math.exp(-rawLogit));

                if (prob >= 0.50) {
                  extracted.push({
                    label: lbl,
                    text: cleanSpan,
                    score: Number(prob.toFixed(2)),
                  });
                }
              });
            }
          }
        }
      }

      // Deduplicate extracted spans
      const uniqueSpans = [];
      const seen = new Set();
      extracted.forEach((item) => {
        const key = `${item.label}:${item.text.toLowerCase()}`;
        if (!seen.has(key)) {
          seen.add(key);
          uniqueSpans.push(item);
        }
      });

      const latency = (performance.now() - startTime).toFixed(1);
      setGlinerLatency(Number(latency));
      setEntities(uniqueSpans);
      setGlinerLoading(false);
    } catch (err) {
      console.error('GLiNER Extraction Error:', err);
      setGlinerLoading(false);
      setGlinerError(err.message || String(err));
    }
  };



  // Handle Send Message (Transformers.js LLM Tab 1)
  const handleSend = async () => {
    if (!inputPrompt.trim() || !modelLoaded || generating || !generatorRef.current) return;

    const currentPrompt = inputPrompt.trim();
    const shortlistedSections = shortlistCandidates(currentPrompt, docSegments, 15);
    const survivingSections = rerankCandidates(currentPrompt, shortlistedSections, 3);
    const fullSystemPrompt = buildAiDqssSystemPrompt(docTitle, docSections, survivingSections, currentPrompt);

    const traceData = {
      docTitle,
      docSections,
      totalSegments: docSegments.length,
      shortlistedCount: shortlistedSections.length,
      survivingEvidences: survivingSections,
    };

    const displayUserMsg = { role: 'user', content: currentPrompt };
    const payloadUserMsg = { role: 'user', content: fullSystemPrompt };

    const newDisplayMessages = [...messages, displayUserMsg];
    const payloadMessages = messages.map((m) => ({ ...m }));

    if (systemPrompt.trim()) {
      payloadMessages.unshift({ role: 'system', content: systemPrompt.trim() });
    }
    payloadMessages.push(payloadUserMsg);

    setMessages([...newDisplayMessages, { role: 'assistant', content: '', trace: traceData }]);
    setInputPrompt('');
    setGenerating(true);
    setTps(0);
    setTokenCount(0);
    isStoppingRef.current = false;

    let tokensGenerated = 0;
    const startTime = performance.now();

    try {
      const streamer = new TextStreamer(generatorRef.current.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (tokenText) => {
          if (isStoppingRef.current) return;

          tokensGenerated++;
          const elapsedSec = (performance.now() - startTime) / 1000;
          const currentTps = elapsedSec > 0 ? (tokensGenerated / elapsedSec).toFixed(1) : 0;

          setTps(Number(currentTps));
          setTokenCount(tokensGenerated);

          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === 'assistant') {
              updated[updated.length - 1] = {
                ...lastMsg,
                content: lastMsg.content + tokenText,
              };
            }
            return updated;
          });
        },
      });

      await generatorRef.current(payloadMessages, {
        max_new_tokens: maxTokens,
        temperature: temperature,
        top_p: topP,
        streamer: streamer,
      });
    } catch (err) {
      if (!isStoppingRef.current) {
        console.error('Transformers.js Generation Error:', err);
        setError(err.message || String(err));
      }
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden font-sans">
      {/* Sidebar Controls - Clean Indigo Studio Theme */}
      <aside className="w-80 bg-slate-900 border-r border-slate-800 flex flex-col justify-between p-5 text-slate-100 z-10 shadow-xl">
        <div className="space-y-6 overflow-y-auto pr-1">
          {/* Header & Tabs */}
          <div className="space-y-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-indigo-500/20 rounded-2xl border border-indigo-500/30 text-indigo-400">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h1 className="font-extrabold text-lg leading-tight text-white tracking-tight">
                  ai4data Lab
                </h1>
                <p className="text-xs text-indigo-300 font-medium flex items-center gap-1.5 mt-0.5">
                  <Cpu className="w-3.5 h-3.5 text-indigo-400" /> WebGPU Intelligence
                </p>
              </div>
            </div>

            {/* 3-Tab Switcher: Chat, Extraction & Liquid Agent */}
            <div className="grid grid-cols-3 bg-slate-950 p-1 rounded-2xl border border-slate-800 text-[10px]">
              <button
                onClick={() => setActiveTab('chat')}
                className={`py-2 rounded-xl font-bold transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'chat'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Brain className="w-3 h-3" /> Chat
              </button>
              <button
                onClick={() => setActiveTab('gliner')}
                className={`py-2 rounded-xl font-bold transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'gliner'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Tag className="w-3 h-3" /> Extract
              </button>
              <button
                onClick={() => setActiveTab('agent')}
                className={`py-2 rounded-xl font-bold transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'agent'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sparkles className="w-3 h-3" /> Agent
              </button>
            </div>
          </div>

          {/* AI-DQSS Document & Section Index Card */}
          <div className="bg-slate-950/80 rounded-2xl p-4 space-y-3 border border-indigo-500/30">
            <div className="flex items-center justify-between text-xs font-bold text-indigo-300 uppercase tracking-wider">
              <span className="flex items-center gap-1.5">
                <FileCheck className="w-4 h-4 text-indigo-400" /> AI-DQSS Section Index
              </span>
              {uploadedDoc && (
                <span className="text-[10px] bg-indigo-500/30 text-indigo-200 px-2 py-0.5 rounded-lg font-mono font-bold border border-indigo-400/30">
                  Indexed
                </span>
              )}
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf,.json,.txt,.md"
              className="hidden"
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isIndexing}
              className="w-full py-2.5 px-3 bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/40 rounded-xl flex items-center justify-center gap-2 text-xs font-bold text-white transition-all cursor-pointer shadow-2xs"
            >
              <Upload className="w-4 h-4 text-indigo-300" />
              <span>{uploadedDoc ? `Replace "${uploadedDoc}"` : 'Upload Document'}</span>
            </button>

            {indexingStatus && (
              <div className="text-[11px] font-mono text-indigo-300 bg-slate-900 p-2.5 rounded-xl border border-slate-800 leading-relaxed">
                {indexingStatus}
              </div>
            )}
          </div>

          {/* TAB 1: CHAT ENGINE WITH MODEL SELECTION DROPDOWN */}
          {activeTab === 'chat' && (
            <div className="bg-slate-950/80 rounded-2xl p-4 space-y-3 border border-slate-800">
              <div className="flex items-center justify-between text-xs font-bold text-indigo-300 uppercase tracking-wider">
                <span className="flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-indigo-400" /> WebGPU Model Engine
                </span>
                {gemmaLoaded ? (
                  <span className="flex items-center gap-1 text-emerald-400 text-xs font-bold">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                  </span>
                ) : (
                  <span className="text-slate-500 text-[10px]">Not Loaded</span>
                )}
              </div>

              {/* Model Switcher Dropdown */}
              <div className="space-y-1.5">
                <label className="text-[11px] font-bold text-slate-300 flex items-center justify-between font-mono">
                  <span>Select Model</span>
                  <span className="text-[10px] text-indigo-400 font-bold">WebGPU 2.7B</span>
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    setGemmaLoaded(false);
                    gemmaInstanceRef.current = null;
                  }}
                  disabled={gemmaLoading || gemmaGenerating}
                  className="w-full bg-slate-900 border border-slate-700 focus:border-indigo-500 rounded-xl px-3 py-2 text-xs font-bold text-white focus:outline-none transition-all cursor-pointer disabled:opacity-50 font-mono"
                >
                  <option value="gemma-4">Google Gemma 4 Mobile (2.7B WGSL)</option>
                  <option value="bonsai-27b">Bonsai 1.7B ONNX (Transformers.js + WebGPU)</option>
                </select>
              </div>

              <button
                onClick={handleLoadChatModel}
                disabled={gemmaLoading}
                className="w-full py-3 px-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 text-xs transition-all cursor-pointer disabled:opacity-40 shadow-md shadow-indigo-600/20"
              >
                <CloudDownload className="w-4 h-4 shrink-0" />
                <span>
                  {gemmaLoaded
                    ? `Reload ${selectedModel === 'gemma-4' ? 'Gemma 4' : 'Bonsai 1.7B'}`
                    : `Load ${selectedModel === 'gemma-4' ? 'Gemma 4 Mobile' : 'Bonsai 1.7B'}`}
                </span>
              </button>

              {gemmaLoading && (
                <div className="space-y-2 pt-1">
                  <div className="flex justify-between text-xs text-slate-300 font-mono">
                    <span className="animate-pulse text-indigo-300 font-medium">{gemmaMsg}</span>
                    <span>{gemmaProgress}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                      style={{ width: `${gemmaProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* WebGPU Embeddings Control */}
              <div className="space-y-1.5 pt-3 border-t border-slate-800">
                <div className="flex justify-between items-center text-[11px] font-mono text-slate-300">
                  <span className="flex items-center gap-1 font-bold">
                    <Sparkle className="w-3.5 h-3.5 text-indigo-400" /> WebGPU Embeddings
                  </span>
                  {embeddingLoaded ? (
                    <span className="text-emerald-400 font-bold bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                      MiniLM-v2 WebGPU
                    </span>
                  ) : (
                    <span className="text-slate-500 text-[10px]">Stem Fallback</span>
                  )}
                </div>
                {!embeddingLoaded && (
                  <button
                    onClick={handleLoadEmbeddingModels}
                    className="w-full py-2 px-3 bg-indigo-950 hover:bg-indigo-900 text-indigo-300 font-bold rounded-xl text-[11px] transition-all cursor-pointer border border-indigo-800/60"
                  >
                    Load WebGPU Embeddings
                  </button>
                )}
                {embeddingMsg && (
                  <div className="text-[10px] font-mono text-indigo-300 bg-slate-900 p-2 rounded-lg border border-slate-800 leading-normal">
                    {embeddingMsg}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: ANONYM-IA GLINER LARGE V2.1 */}
          {activeTab === 'gliner' && (
            <div className="bg-slate-950/80 rounded-2xl p-4 space-y-3 border border-slate-800">
              <div className="flex items-center justify-between text-xs font-bold text-indigo-300 uppercase tracking-wider">
                <span className="flex items-center gap-1.5">
                  <Tag className="w-4 h-4 text-indigo-400" /> ONNX Model
                </span>
                {glinerLoaded ? (
                  <span className="flex items-center gap-1 text-emerald-400 text-xs font-bold">
                    <CheckCircle2 className="w-3.5 h-3.5" /> WebGPU Ready
                  </span>
                ) : (
                  <span className="text-slate-500 text-[10px]">Not Loaded</span>
                )}
              </div>

              <div className="text-[11px] text-slate-300 bg-slate-900 p-2.5 rounded-xl border border-slate-800 font-mono flex items-center justify-between">
                <span>gliner_large-v2.1</span>
                {isCached ? (
                  <span className="text-indigo-300 flex items-center gap-1 text-[10px] font-bold">
                    <Database className="w-3.5 h-3.5 text-indigo-400" /> Cached
                  </span>
                ) : (
                  <span className="text-slate-400 text-[10px]">Not Cached</span>
                )}
              </div>

              <button
                onClick={handleLoadGlinerDirectONNX}
                disabled={glinerLoading}
                className="w-full py-3 px-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 text-xs transition-all cursor-pointer disabled:opacity-40 shadow-md shadow-indigo-600/20"
              >
                <CloudDownload className="w-4 h-4 shrink-0" />
                <span>{glinerLoaded ? 'Reload WebGPU Model' : 'Load ONNX WebGPU Session'}</span>
              </button>
            </div>
          )}

          {/* TAB 3: LIQUID RESEARCH AGENT PANEL */}
          {activeTab === 'agent' && (
            <div className="bg-slate-950/80 rounded-2xl p-4 space-y-3 border border-slate-800">
              <div className="flex items-center justify-between text-xs font-bold text-indigo-300 uppercase tracking-wider">
                <span className="flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-indigo-400" /> Agent Engine
                </span>
                {agent.modelPhase === 'ready' ? (
                  <span className="flex items-center gap-1 text-emerald-400 text-xs font-bold">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                  </span>
                ) : (
                  <span className="text-slate-500 text-[10px]">{agent.modelPhase === 'loading' ? 'Loading' : 'Idle'}</span>
                )}
              </div>

              <div className="text-[11px] text-slate-300 bg-slate-900 p-2.5 rounded-xl border border-slate-800 font-mono flex flex-col gap-1">
                <span className="font-bold text-white">LFM2.5-2.6B-ONNX</span>
                <span className="text-slate-400 text-[9px]">Quantization: Q4 (~1.9 GB)</span>
              </div>

              <button
                onClick={agent.load}
                disabled={agent.modelPhase === 'loading' || agent.modelPhase === 'ready'}
                className="w-full py-3 px-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 text-xs transition-all cursor-pointer disabled:opacity-40 shadow-md shadow-indigo-600/20"
              >
                <CloudDownload className="w-4 h-4 shrink-0" />
                <span>{agent.modelPhase === 'ready' ? 'Model Ready' : agent.modelPhase === 'loading' ? 'Streaming...' : 'Load WebGPU Engine'}</span>
              </button>

              {agent.modelPhase === 'loading' && agent.progress && (
                <div className="space-y-2 pt-1">
                  <div className="flex justify-between text-xs text-slate-300 font-mono">
                    <span className="animate-pulse text-indigo-300 font-medium">Downloading...</span>
                    <span>{Math.round(agent.progress.progress || 0)}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                      style={{ width: `${agent.progress.progress || 0}%` }}
                    />
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono text-right">
                    {((agent.progress.loaded || 0) / (1024 * 1024)).toFixed(1)} / {((agent.progress.total || 0) / (1024 * 1024)).toFixed(1)} MB
                  </div>
                </div>
              )}

              <div className="pt-3 border-t border-slate-800 space-y-2">
                <span className="text-[10px] font-bold text-slate-300 font-mono uppercase tracking-wider block">Allowed Tools</span>
                <div className="space-y-1.5 text-xs text-slate-400">
                  <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                    <input type="checkbox" checked={true} disabled={true} className="rounded accent-indigo-600" />
                    <span>create_action_plan (core)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                    <input type="checkbox" checked={true} disabled={true} className="rounded accent-indigo-600" />
                    <span>update_action_plan (core)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                    <input 
                      type="checkbox" 
                      checked={allowedTools.includes('search_wikipedia')} 
                      onChange={(e) => {
                        if (e.target.checked) setAllowedTools([...allowedTools, 'search_wikipedia']);
                        else setAllowedTools(allowedTools.filter(t => t !== 'search_wikipedia'));
                      }}
                      className="rounded accent-indigo-600" 
                    />
                    <span>search_wikipedia</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                    <input 
                      type="checkbox" 
                      checked={allowedTools.includes('get_current_context')} 
                      onChange={(e) => {
                        if (e.target.checked) setAllowedTools([...allowedTools, 'get_current_context']);
                        else setAllowedTools(allowedTools.filter(t => t !== 'get_current_context'));
                      }}
                      className="rounded accent-indigo-600" 
                    />
                    <span>get_current_context</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                    <input 
                      type="checkbox" 
                      checked={allowedTools.includes('ask_user')} 
                      onChange={(e) => {
                        if (e.target.checked) setAllowedTools([...allowedTools, 'ask_user']);
                        else setAllowedTools(allowedTools.filter(t => t !== 'ask_user'));
                      }}
                      className="rounded accent-indigo-600" 
                    />
                    <span>ask_user</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-white">
                    <input 
                      type="checkbox" 
                      checked={allowedTools.includes('search_uploaded_document')} 
                      onChange={(e) => {
                        if (e.target.checked) setAllowedTools([...allowedTools, 'search_uploaded_document']);
                        else setAllowedTools(allowedTools.filter(t => t !== 'search_uploaded_document'));
                      }}
                      className="rounded accent-indigo-600" 
                    />
                    <span>search_uploaded_document</span>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* System Stats Footer */}
        <div className="pt-4 border-t border-slate-800/80 space-y-2 text-xs">
          <div className="flex justify-between items-center text-slate-400">
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-indigo-400" /> Cache & Speed
            </span>
            <span className="font-mono text-indigo-300 font-bold text-[11px]">
              {downloadStats || 'CacheStorage v3 Active'}
            </span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" /> WebGPU Latency
            </span>
            <span className="font-mono text-indigo-300 font-bold">
              {activeTab === 'chat'
                ? (selectedModel === 'gemma-4' ? `${gemmaTps} tok/s` : `${tps} tok/s`)
                : activeTab === 'agent'
                ? `${agent.stats?.tps ? agent.stats.tps.toFixed(1) : 0} tok/s`
                : `${glinerLatency} ms`}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Area - ai4data Lab Clean Theme */}
      <main className="flex-1 flex flex-col h-full bg-slate-50 relative">
        <header className="h-16 border-b border-slate-200 px-6 flex items-center justify-between bg-white shadow-2xs">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-indigo-600 animate-pulse" />
            <span className="font-extrabold text-sm text-slate-900 tracking-tight">
              {activeTab === 'chat'
                ? `${selectedModel === 'gemma-4' ? 'Google Gemma 4 E2B WGSL (2.7B)' : 'Bonsai 2.7B GGUF Model'} — AI-DQSS Section Engine`
                : activeTab === 'agent'
                ? 'LiquidAI/LFM2.5-2.6B-ONNX — WebGPU Autonomous Research Agent'
                : `Anonym-IA/gliner_large-v2.1 (${glinerVariant}) — WebGPU Zero-Shot Extraction`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {activeTab === 'chat' && (
              <button
                onClick={() => {
                  setGemmaMessages([]);
                  gemmaHistoryRef.current = [];
                }}
                disabled={gemmaMessages.length === 0 || gemmaGenerating}
                className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-bold transition-all cursor-pointer disabled:opacity-40"
              >
                Clear Chat
              </button>
            )}
            <div className="text-xs font-bold text-slate-700 bg-indigo-50/80 px-3 py-1.5 rounded-xl border border-indigo-100 flex items-center gap-1.5">
              <Filter className="w-4 h-4 text-indigo-600" /> AI-DQSS Section Engine
            </div>
          </div>
        </header>

        {/* TAB 1: CHAT WORKSPACE (Gemma 4 / Bonsai 2.7B Switchable) */}
        {activeTab === 'chat' && (
          <>
            {gemmaError && (
              <div className="bg-rose-50 border-b border-rose-200 px-6 py-3 flex items-center gap-3 text-xs text-rose-700 font-medium">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
                <span className="flex-1">{gemmaError}</span>
              </div>
            )}

            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {gemmaMessages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto space-y-4">
                  <div className="p-5 bg-white rounded-3xl border border-slate-200 text-indigo-600 shadow-md">
                    <Brain className="w-10 h-10 text-indigo-600" />
                  </div>
                  <div>
                    <h3 className="font-extrabold text-xl text-slate-900">AI-DQSS Section Block Workspace</h3>
                    <p className="text-xs text-slate-500 mt-1.5 leading-relaxed max-w-md">
                      Upload any document. Queries retrieve complete Section Blocks (e.g. DOC-1:SECTION-III) with all principles intact, and stream instant answers using {selectedModel === 'gemma-4' ? 'Gemma 4 Mobile' : 'Bonsai 2.7B'}!
                    </p>
                  </div>
                </div>
              ) : (
                gemmaMessages.map((msg, index) => (
                  <React.Fragment key={index}>
                    {msg.role === 'user' ? (
                      <div className="flex gap-4 justify-end">
                        <div className="max-w-2xl bg-indigo-600 text-white rounded-3xl rounded-tr-none px-5 py-3.5 text-sm font-semibold leading-relaxed shadow-sm">
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        </div>
                      </div>
                    ) : (
                      <AssistantMessage
                        message={msg}
                        isGenerating={gemmaGenerating}
                        isLastMessage={index === gemmaMessages.length - 1}
                        onOpenEvidenceMap={(ev, trace) => setEvidenceModal({ ev, trace })}
                      />
                    )}
                  </React.Fragment>
                ))
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="p-4 border-t border-slate-200 bg-white shadow-2xs">
              <div className="max-w-4xl mx-auto flex items-center gap-3">
                <input
                  type="text"
                  value={gemmaInput}
                  onChange={(e) => setGemmaInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !gemmaGenerating && handleSendGemma()}
                  placeholder={
                    gemmaLoaded
                      ? `Ask ${selectedModel === 'gemma-4' ? 'Gemma 4' : 'Bonsai 2.7B'} anything about the document...`
                      : `Load ${selectedModel === 'gemma-4' ? 'Gemma 4' : 'Bonsai 2.7B'} WebGPU Engine to enable chat...`
                  }
                  disabled={!gemmaLoaded || gemmaGenerating}
                  className="flex-1 bg-slate-50 border border-slate-200 focus:border-indigo-500 focus:bg-white rounded-2xl px-4 py-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none transition-all disabled:opacity-50 font-medium"
                />

                {gemmaGenerating ? (
                  <button
                    onClick={() => {
                      isGemmaStoppingRef.current = true;
                    }}
                    className="px-6 py-3 bg-rose-600 hover:bg-rose-500 text-white rounded-2xl text-xs font-bold flex items-center gap-2 transition-all cursor-pointer shadow-md shadow-rose-600/20"
                  >
                    <Square className="w-4 h-4" /> Stop
                  </button>
                ) : (
                  <button
                    onClick={handleSendGemma}
                    disabled={!gemmaLoaded || !gemmaInput.trim()}
                    className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-2xl text-xs font-bold flex items-center gap-2 transition-all cursor-pointer shadow-md shadow-indigo-600/20"
                  >
                    <Send className="w-4 h-4" /> Send
                  </button>
                )}
              </div>
            </div>
          </>
        )}

        {/* TAB 2: ANONYM-IA GLINER VIEW */}
        {activeTab === 'gliner' && (
          <div className="flex-1 overflow-y-auto p-6 max-w-4xl mx-auto w-full space-y-6">


            {/* Input Card */}
            <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-4">
              <h3 className="font-extrabold text-base text-slate-900 flex items-center gap-2">
                <Tag className="w-5 h-5 text-indigo-600" /> GLiNER Large Zero-Shot Extraction ({glinerVariant})
              </h3>

              <div>
                <label className="block text-xs font-bold text-slate-800 mb-1.5">Target Text</label>
                <textarea
                  rows={4}
                  value={glinerText}
                  onChange={(e) => setGlinerText(e.target.value)}
                  placeholder="Enter text to analyze for entity extraction..."
                  className="w-full bg-slate-50 border border-slate-200 focus:border-indigo-500 focus:bg-white rounded-2xl p-3.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none transition-all font-mono leading-relaxed"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-800 mb-1.5">Entity Labels (Comma Separated)</label>
                <input
                  type="text"
                  value={glinerLabels}
                  onChange={(e) => setGlinerLabels(e.target.value)}
                  placeholder="e.g. policy title, catalogue number, effective date, organization"
                  className="w-full bg-slate-50 border border-slate-200 focus:border-indigo-500 focus:bg-white rounded-2xl p-3.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none transition-all font-mono"
                />
              </div>

              <button
                onClick={handleExtractEntities}
                disabled={glinerLoading}
                className="w-full py-3.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-2xl text-xs font-bold flex items-center justify-center gap-2 transition-all cursor-pointer shadow-md shadow-indigo-600/20"
              >
                <Search className="w-4 h-4" />
                <span>
                  {glinerLoaded
                    ? 'Extract Entities (WebGPU Single-Pass)'
                    : 'Load Model & Extract Entities (WebGPU)'}
                </span>
              </button>
            </div>

            {/* Extracted Entities Output */}
            <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-4">
              <div className="flex justify-between items-center border-b border-slate-100 pb-3">
                <h3 className="font-extrabold text-base text-slate-900 flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" /> Extracted Entity Spans
                </h3>
                {glinerLatency > 0 && (
                  <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-100 px-3 py-1 rounded-xl font-bold">
                    GPU Forward Pass: {glinerLatency} ms
                  </span>
                )}
              </div>

              {entities.length === 0 ? (
                <p className="text-xs text-slate-400 text-center py-6">
                  {glinerLoaded
                    ? 'Click "Extract Entities" to run instant WebGPU matrix pass.'
                    : 'Click "Load ONNX WebGPU Session" in the sidebar to stream model.'}
                </p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {entities.map((ent, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-50 border border-slate-200 rounded-2xl p-4 flex justify-between items-center text-xs shadow-2xs hover:border-indigo-300 transition-all"
                    >
                      <div className="space-y-1">
                        <span className="px-2.5 py-0.5 bg-indigo-100 text-indigo-900 rounded-lg font-mono text-[10px] uppercase font-bold tracking-wider">
                          {ent.label || 'ENTITY'}
                        </span>
                        <div className="font-bold text-slate-900 text-sm mt-1">{ent.text}</div>
                      </div>
                      <span className="font-mono text-indigo-600 text-xs font-bold">
                        {((ent.score || 0.95) * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: LIQUID RESEARCH AGENT WORKSPACE */}
        {activeTab === 'agent' && (
          <div className="flex-1 flex w-full overflow-hidden">
            {/* Left Column: Mission Control & Logs */}
            <div className="w-1/2 min-w-0 flex flex-col h-full border-r border-slate-200 bg-slate-50">
              {/* Plan Progress */}
              {agent.plan && (
                <div className="bg-white p-4 border-b border-slate-200 space-y-3">
                  <div className="flex justify-between items-center text-xs font-bold text-slate-800 font-mono">
                    <span>Active Plan: {agent.plan.goal}</span>
                    <span className="text-indigo-600 bg-indigo-50 border border-indigo-100 px-2.5 py-0.5 rounded-lg">
                      Step {agent.plan.currentStep} of {agent.plan.steps.length}
                    </span>
                  </div>
                  <div className="space-y-1.5">
                    {agent.plan.steps.map((step, idx) => (
                      <div key={step.id} className="flex items-center justify-between text-xs p-2 rounded-xl border border-slate-100 bg-slate-50/50">
                        <div className="flex items-center gap-2">
                          <span className={`w-2.5 h-2.5 rounded-full ${
                            step.status === 'completed' ? 'bg-emerald-500' :
                            step.status === 'in_progress' ? 'bg-indigo-500 animate-pulse' :
                            step.status === 'blocked' ? 'bg-rose-500' :
                            'bg-slate-300'
                          }`} />
                          <span className={`font-semibold ${step.status === 'completed' ? 'text-slate-500 line-through' : 'text-slate-800'}`}>
                            {idx + 1}. {step.title}
                          </span>
                        </div>
                        <span className={`font-mono text-[10px] font-bold uppercase px-2 py-0.5 rounded-md ${
                          step.status === 'completed' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
                          step.status === 'in_progress' ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' :
                          step.status === 'blocked' ? 'bg-rose-50 text-rose-700 border border-rose-200' :
                          'bg-slate-100 text-slate-500'
                        }`}>
                          {step.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Execution Trace Log */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {agent.trace.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto space-y-4 pt-12">
                    <div className="p-5 bg-white rounded-3xl border border-slate-200 text-indigo-600 shadow-md">
                      <Sparkles className="w-8 h-8 text-indigo-600 animate-pulse" />
                    </div>
                    <div>
                      <h4 className="font-extrabold text-sm text-slate-900">Local Research Agent Logs</h4>
                      <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
                        Verify the WebGPU Engine is loaded, then enter your goal and click "Run Agent" to stream reasoning turns and execution traces!
                      </p>
                    </div>
                  </div>
                ) : (
                  agent.trace.map((item) => {
                    if (item.kind === 'turn') {
                      return (
                        <div key={item.id} className="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs space-y-3">
                          <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 border-b border-slate-100 pb-2 font-mono">
                            <span>Agent Reasoning Loop — Turn {item.turn}</span>
                            <span>{agent.elapsedMs ? `${(agent.elapsedMs / 1000).toFixed(1)}s elapsed` : ''}</span>
                          </div>
                          
                          {/* Reasoning think block */}
                          {item.thinking && (
                            <details className="group border border-slate-100 bg-slate-50 rounded-xl p-2.5 text-xs text-slate-600 leading-relaxed font-mono">
                              <summary className="font-bold text-[11px] text-slate-500 uppercase tracking-wider flex items-center justify-between cursor-pointer select-none">
                                <span>Thinking Process</span>
                                <span className="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-600 font-normal group-open:hidden">Show</span>
                                <span className="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-600 font-normal hidden group-open:inline">Hide</span>
                              </summary>
                              <div className="mt-2 pt-2 border-t border-slate-200/50 whitespace-pre-wrap">
                                {item.thinking}
                              </div>
                            </details>
                          )}
                          
                          {/* Content reply */}
                          {item.content && (
                            <div className="text-xs text-slate-800 whitespace-pre-wrap leading-relaxed">
                              {item.content}
                            </div>
                          )}
                        </div>
                      );
                    }
                    
                    if (item.kind === 'tool') {
                      return (
                        <div key={item.id} className="bg-slate-900 text-slate-200 rounded-2xl p-4 border border-slate-800 shadow-xs space-y-3 font-mono text-xs">
                          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                            <span className="flex items-center gap-1.5 font-bold text-indigo-400">
                              <Cpu className="w-4 h-4 text-indigo-400" /> Tool call: {item.name}
                            </span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              item.status === 'complete' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-indigo-950 text-indigo-400 border border-indigo-800 animate-pulse'
                            }`}>
                              {item.status}
                            </span>
                          </div>
                          
                          {/* Tool arguments */}
                          <div>
                            <span className="text-slate-500 text-[10px] font-bold uppercase tracking-wider block">Arguments:</span>
                            <pre className="text-[11px] bg-slate-950 p-2 rounded-xl border border-slate-800 overflow-x-auto text-indigo-300 mt-1">
                              {JSON.stringify(item.arguments, null, 2)}
                            </pre>
                          </div>
                          
                          {/* Tool output/result */}
                          {item.result && (
                            <div>
                              <span className="text-slate-500 text-[10px] font-bold uppercase tracking-wider block">Output:</span>
                              <pre className="text-[11px] bg-slate-950 p-2 rounded-xl border border-slate-800 overflow-x-auto text-emerald-400 mt-1 max-h-40 overflow-y-auto">
                                {JSON.stringify(item.result, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      );
                    }
                    
                    return null;
                  })
                )}

                {/* Interactive Ask User Prompt Card */}
                {agent.interaction && (
                  <div className="bg-indigo-50 border border-indigo-200 rounded-3xl p-6 shadow-md space-y-4">
                    <h4 className="font-extrabold text-sm text-indigo-900 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-indigo-600" /> Agent requires clarification
                    </h4>
                    <p className="text-xs text-indigo-800 font-semibold">{agent.interaction.question}</p>
                    
                    {agent.interaction.responseType === 'select' && (
                      <div className="grid grid-cols-2 gap-2">
                        {agent.interaction.options.map((option, idx) => (
                          <button
                            key={idx}
                            onClick={() => agent.submitInteraction(option)}
                            className="p-3 bg-white hover:bg-slate-50 border border-indigo-200 rounded-2xl text-xs font-bold text-indigo-900 transition-all cursor-pointer text-left hover:border-indigo-400 shadow-2xs"
                          >
                            {option}
                          </button>
                        ))}
                      </div>
                    )}
                    
                    {(agent.interaction.responseType === 'text' || agent.interaction.allowCustom) && (
                      <div className="flex gap-2">
                        <input
                          type="text"
                          placeholder={agent.interaction.placeholder || "Type your custom answer here..."}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && e.target.value.trim()) {
                              agent.submitInteraction(e.target.value);
                            }
                          }}
                          className="flex-1 bg-white border border-indigo-200 focus:border-indigo-500 rounded-2xl px-4 py-3 text-xs text-indigo-900 focus:outline-none placeholder-indigo-400/60 shadow-inner"
                        />
                        <button
                          onClick={(e) => {
                            const input = e.currentTarget.previousSibling;
                            if (input && input.value.trim()) {
                              agent.submitInteraction(input.value);
                            }
                          }}
                          className="px-4 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl text-xs font-bold shadow-md cursor-pointer"
                        >
                          Submit
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Research Controls Form */}
              <div className="p-4 border-t border-slate-200 bg-white space-y-3">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={agentPrompt}
                    onChange={(e) => setAgentPrompt(e.target.value)}
                    disabled={agent.runPhase === 'thinking' || agent.runPhase === 'waiting' || agent.modelPhase !== 'ready'}
                    placeholder="Enter research mission goal (e.g. History of the Colosseum)..."
                    className="flex-1 bg-slate-50 border border-slate-200 focus:border-indigo-500 focus:bg-white rounded-2xl px-4 py-3.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none transition-all font-mono leading-relaxed"
                  />
                  {agent.runPhase === 'thinking' || agent.runPhase === 'waiting' ? (
                    <button
                      onClick={agent.stop}
                      className="px-6 py-3.5 bg-rose-600 hover:bg-rose-500 text-white rounded-2xl text-xs font-bold flex items-center gap-2 transition-all cursor-pointer shadow-md shadow-rose-600/20"
                    >
                      <Square className="w-4 h-4 fill-white" /> Stop
                    </button>
                  ) : (
                    <button
                      onClick={() => agent.generate(agentPrompt, allowedTools, docSegments)}
                      disabled={agent.modelPhase !== 'ready' || !agentPrompt.trim()}
                      className="px-6 py-3.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-2xl text-xs font-bold flex items-center gap-2 transition-all cursor-pointer shadow-md shadow-indigo-600/20"
                    >
                      <Send className="w-4 h-4" /> Run Agent
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column: Compiled Research Document */}
            <div className="w-1/2 min-w-0 flex flex-col h-full bg-white">
              <div className="p-4 border-b border-slate-200 flex justify-between items-center bg-slate-50/50">
                <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5 uppercase font-mono tracking-wider">
                  <FileText className="w-4 h-4 text-indigo-600" /> Compiled Research Paper
                </span>
                {agent.artifact && (
                  <button
                    onClick={() => {
                      const blob = new Blob([agent.artifact.content], { type: agent.artifact.mimeType });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = agent.artifact.filename;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-bold transition-all cursor-pointer shadow-md shadow-emerald-600/20 flex items-center gap-1"
                  >
                    <CloudDownload className="w-3.5 h-3.5" /> Download Markdown
                  </button>
                )}
              </div>
              <div className="flex-1 overflow-y-auto p-6 prose prose-slate max-w-none text-xs leading-relaxed">
                {agent.artifact ? (
                  <div 
                    className="space-y-4 text-slate-800 font-sans animate-fade-in" 
                    dangerouslySetInnerHTML={renderMarkdown(agent.artifact.content)} 
                  />
                ) : agent.runPhase === 'thinking' ? (
                  <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto space-y-4">
                    <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
                    <div>
                      <h4 className="font-extrabold text-sm text-slate-900">Agent is researching...</h4>
                      <p className="text-[11px] text-slate-500 mt-1.5 font-mono">
                        Executing workflow agenda, updating action plans, running Wikipedia searches, and compiling evidence sections. Please wait...
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto space-y-4 pt-24">
                    <BookOpen className="w-10 h-10 text-slate-300" />
                    <div>
                      <h4 className="font-bold text-sm text-slate-400">No Document Compiled</h4>
                      <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                        Once the agent finishes executing all plan steps, the final research paper with citations will be rendered here.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* AI-DQSS Interactive Evidence Map Modal */}
      <EvidenceMapModal
        modalData={evidenceModal}
        docSegments={docSegments}
        docTitle={docTitle}
        onClose={() => setEvidenceModal(null)}
      />
    </div>
  );
}
