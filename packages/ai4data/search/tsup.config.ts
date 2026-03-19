import { defineConfig } from 'tsup'

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
})
