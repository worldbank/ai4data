import * as ort from 'onnxruntime-node';
import { AutoTokenizer } from '@huggingface/transformers';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import * as path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const modelPath = path.resolve(__dirname, '../public/onnx/gliner_large-v2.1_q4.onnx');

console.log('=== GLiNER Real ONNX Model Forward Pass CLI Runner ===');

// Parse CLI Flags
const args = process.argv.slice(2);
let sampleText = 'Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.';
let targetLabels = ['company', 'person', 'product', 'location'];

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--input' && args[i + 1]) {
    sampleText = args[i + 1];
    i++;
  } else if (args[i] === '--labels' && args[i + 1]) {
    targetLabels = args[i + 1].split(',').map(l => l.trim()).filter(Boolean);
    i++;
  }
}

console.log('Sample Text:', sampleText);
console.log('Target Labels:', targetLabels);

async function runCliInference(sampleText, targetLabels) {
  const startTime = performance.now();

  console.log('\n[1/4] Loading AutoTokenizer for onnx-community/gliner_large-v2.1...');
  const tokenizer = await AutoTokenizer.from_pretrained('onnx-community/gliner_large-v2.1');

  console.log(`[2/4] Reading ONNX model weights dynamically from ${modelPath}...`);
  const buffer = fs.readFileSync(modelPath);

  console.log('[3/4] Initializing ONNX InferenceSession...');
  const session = await ort.InferenceSession.create(buffer);

  console.log('[4/4] Tokenizing inputs and building feed tensors...');
  const words = sampleText.split(/\s+/).filter(Boolean);
  const numWords = words.length;
  const maxSpanWidth = 12;

  // GLiNER BPE Special Tokens formatting
  const prompt = targetLabels.map(l => `<<ENT>> ${l}`).join(' ') + ' <<SEP>> ' + sampleText;
  const tokenized = tokenizer(prompt);

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

  console.log('\n--- BPE Token Alignment Debugger ---');
  for (let w = 0; w < numWords; w++) {
    const wordTokenized = tokenizer(words[w]);
    const subtokenCount = wordTokenized.input_ids.data.length - 2; // Subtract [CLS] and [SEP]
    console.log(`Word [${words[w]}]: Token index start: ${currentTokenIdx}, Subtokens: ${subtokenCount}, Assigned index: ${w + 1}`);
    for (let t = 0; t < subtokenCount; t++) {
      if (currentTokenIdx < seqLen - 1) {
        wordsMask[currentTokenIdx] = BigInt(w + 1); // 1-based index!
        currentTokenIdx++;
      }
    }
  }

  console.log('Sequence Length:', seqLen);
  console.log('Token IDs:', tokenIds);
  console.log('Words Mask:', Array.from(wordsMask).map(Number));

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

  const outputs = await session.run(feeds);
  const latency = (performance.now() - startTime).toFixed(1);

  console.log(`\n================================================================`);
  console.log(`🎉 ONNX MODEL FORWARD PASS SUCCESSFUL! (Latency: ${latency} ms)`);
  console.log(`Logits Tensor Output Shape: [${outputs.logits.dims.join(', ')}]`);
  console.log(`================================================================`);

  // Parse extracted entity spans dynamically from pure Float32Array logits
  const extracted = [];
  const logitsData = outputs.logits.data;
  const numLabels = targetLabels.length;
  const allSpans = [];

  for (let i = 0; i < numWords; i++) {
    for (let j = 0; j < Math.min(maxSpanWidth, numWords - i); j++) {
      const spanText = words.slice(i, i + j + 1).join(' ');
      const cleanSpan = spanText.replace(/^[^\w$]+|[^\w]+$/g, '');

      if (cleanSpan.length >= 2) {
        const spanIdx = i * maxSpanWidth + j;

        targetLabels.forEach((lbl, labelIdx) => {
          const logitIndex = spanIdx * numLabels + labelIdx;
          const rawLogit = logitsData[logitIndex];
          const prob = 1 / (1 + Math.exp(-rawLogit));

          allSpans.push({
            label: lbl,
            text: cleanSpan,
            logit: rawLogit,
            score: Number(prob.toFixed(2)),
          });

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

  // Log top 10 highest logits
  allSpans.sort((a, b) => b.logit - a.logit);
  console.log('\n--- Top 10 Candidate Spans by Logit Score ---');
  console.table(allSpans.slice(0, 10));

  // Deduplicate overlapping/duplicate spans
  const uniqueSpans = [];
  const seen = new Set();
  extracted.forEach((item) => {
    const key = `${item.label}:${item.text.toLowerCase()}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniqueSpans.push(item);
    }
  });

  console.log('\n--- Extracted Entity Spans (ONNX Tensor Inference) ---');
  console.table(uniqueSpans);
}

runCliInference(sampleText, targetLabels).catch((err) => console.error('CLI Execution Error:', err));
