import { defineConfig } from 'tsup'
import { readFileSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const pkg = JSON.parse(readFileSync(join(__dirname, 'package.json'), 'utf8'))
const version = pkg.version ?? '0.0.0'

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    worker: 'src/workers/search.worker.ts',
    'rank-worker': 'src/workers/rank.worker.ts',
  },
  format: ['esm'],
  target: 'es2022', // top-level await (rank.worker) requires ES2022+
  platform: 'browser', // avoid bundling Node .node addons (onnxruntime, sharp) into dist/
  outDir: 'dist',
  dts: { entry: 'src/index.ts' },
  splitting: false,
  minify: true,
  sourcemap: true,
  clean: true,
  // Bundle all dependencies into worker outputs (no externals for worker entries)
  noExternal: [
    /^@xenova\/transformers/,
    /^wink-bm25-text-search/,
    /^wink-nlp/,
    /^wink-eng-lite-web-model/,
    /^@huggingface\/transformers/,
  ],
  esbuildOptions(options) {
    options.define = {
      ...options.define,
      __PACKAGE_VERSION__: JSON.stringify(version),
    }
  },
})
