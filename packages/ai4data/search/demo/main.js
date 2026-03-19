/**
 * Minimal demo for @ai4data/search.
 * Served from the package root so that ../dist/index.mjs and the default worker path resolve.
 *
 * Run from packages/ai4data/search:
 *   npm run build && npx serve . -p 5173
 * Then open http://localhost:5173/demo/
 */

import { SearchClient } from '../dist/index.mjs'

const manifestInput = document.getElementById('manifestUrl')
const connectBtn = document.getElementById('connectBtn')
const statusEl = document.getElementById('status')
const queryInput = document.getElementById('query')
const searchBtn = document.getElementById('searchBtn')
const resultsList = document.getElementById('results')

let client = null

function setStatus(text) {
  statusEl.textContent = text
}

function setResults(items) {
  resultsList.innerHTML = items
    .map(
      (r) =>
        `<li><strong>${escapeHtml(r.title || String(r.id))}</strong><br><small>id: ${r.id} · score: ${(r.score ?? 0).toFixed(4)}</small></li>`
    )
    .join('')
}

function escapeHtml(s) {
  const div = document.createElement('div')
  div.textContent = s
  return div.innerHTML
}

function updateUI() {
  if (!client) return
  setStatus(client.loadingMessage)
  const ready = client.isIndexReady
  queryInput.disabled = !ready
  searchBtn.disabled = !ready
  if (ready) {
    setStatus(client.isModelReady ? 'Ready (semantic + hybrid)' : 'Index ready (lexical only)')
  }
}

connectBtn.addEventListener('click', () => {
  const url = manifestInput.value.trim()
  if (!url) {
    setStatus('Please enter a manifest URL.')
    return
  }

  if (client) {
    client.destroy()
    client = null
  }

  setStatus('Connecting…')
  setResults([])
  queryInput.value = ''
  connectBtn.disabled = true

  try {
    client = new SearchClient(url)

    client.on('progress', () => updateUI())
    client.on('index_ready', () => updateUI())
    client.on('ready', () => updateUI())
    client.on('error', () => updateUI())

    client.on('results', ({ data }) => {
      setResults(data || [])
      updateUI()
    })

    // Poll loading message until ready (worker doesn’t emit progress on every tick)
    const interval = setInterval(() => {
      updateUI()
      if (client.isIndexReady) clearInterval(interval)
    }, 500)
    setTimeout(() => clearInterval(interval), 120000)
  } catch (err) {
    setStatus(`Error: ${err.message}`)
  }
  connectBtn.disabled = false
})

searchBtn.addEventListener('click', () => {
  const q = queryInput.value.trim()
  if (!q || !client) return
  client.search(q, { topK: 10, mode: 'hybrid' })
  setStatus('Searching…')
})

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') searchBtn.click()
})
