/**
 * rank.worker.ts
 *
 * Cross-encoder reranking worker using jina-reranker-v1-turbo-en.
 * Accepts a query + list of document strings and returns scored, sorted results.
 */

import { env, AutoTokenizer, XLMRobertaModel } from '@huggingface/transformers'

env.allowRemoteModels = true

const MODEL_ID = 'jinaai/jina-reranker-v1-turbo-en'

let model: XLMRobertaModel | null = null
let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null

async function init(): Promise<void> {
  model = await XLMRobertaModel.from_pretrained(MODEL_ID, { dtype: 'q8' } as any)
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID)
  console.log('[rank.worker] Model and tokenizer loaded')
}

await init()

interface RankRequest {
  query: string
  documents: string[]
  top_k?: number
  return_documents?: boolean
}

interface RankResult {
  corpus_id: number
  score: number
  text?: string
}

self.onmessage = async (e: MessageEvent<RankRequest | { ping: boolean }>) => {
  try {
    if ('ping' in e.data && e.data.ping) {
      self.postMessage('pong')
      return
    }

    const { query, documents, top_k, return_documents } = e.data as RankRequest

    const inputs = await tokenizer!(
      new Array(documents.length).fill(query),
      { text_pair: documents, padding: true, truncation: true },
    )

    const { logits } = await model!(inputs)

    const scores: RankResult[] = (logits as any)
      .sigmoid()
      .tolist()
      .map(([score]: [number], i: number) => ({
        corpus_id: i,
        score,
        ...(return_documents ? { text: documents[i] } : {}),
      }))

    scores.sort((a, b) => b.score - a.score)

    self.postMessage(top_k ? scores.slice(0, top_k) : scores)
  } catch (err) {
    console.error('[rank.worker] Error:', err)
    self.postMessage([])
  }
}
