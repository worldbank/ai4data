import { Wllama } from '@wllama/wllama';

let wllama = null;
let isAbortRequested = false;

const CONFIG_PATHS = {
  default: '/wllama/wllama.wasm',
};

self.addEventListener('message', async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case 'load-file': {
      try {
        const { file, nCtx = 2048, nThreads = 4, nGpuLayers = 0 } = data;

        self.postMessage({ status: 'loading', message: 'Initializing Wllama engine...' });

        if (wllama) {
          await wllama.exit();
        }

        wllama = new Wllama(CONFIG_PATHS, {
          suppressNativeLog: false,
          logger: {
            debug: () => {},
            log: (...args) => console.log('[Wllama]', ...args),
            warn: (...args) => console.warn('[Wllama]', ...args),
            error: (...args) => console.error('[Wllama]', ...args),
          },
        });

        self.postMessage({ status: 'loading', message: 'Loading 1-bit GGUF model into memory...' });

        // loadModel takes an array of Blobs/Files
        await wllama.loadModel([file], {
          n_ctx: nCtx,
          n_threads: nThreads,
          n_gpu_layers: nGpuLayers,
          progressCallback: ({ loaded, total }) => {
            const progress = total ? Math.round((loaded / total) * 100) : 0;
            self.postMessage({
              status: 'progress',
              loaded,
              total,
              progress,
              message: `Loading GGUF... ${progress}%`,
            });
          },
        });

        const meta = wllama.getModelMetadata();
        self.postMessage({
          status: 'ready',
          meta,
          message: 'Model ready for inference!',
        });
      } catch (err) {
        console.error('Error loading GGUF model:', err);
        self.postMessage({ status: 'error', error: err.message || String(err) });
      }
      break;
    }

    case 'generate': {
      if (!wllama || !wllama.isModelLoaded()) {
        self.postMessage({ status: 'error', error: 'Model is not loaded yet.' });
        return;
      }

      isAbortRequested = false;
      const { messages, temperature = 0.7, topP = 0.9, maxTokens = 1024 } = data;

      let tokenCount = 0;
      const startTime = performance.now();

      self.postMessage({ status: 'start' });

      try {
        await wllama.createChatCompletion({
          messages,
          sampling: {
            temp: temperature,
            top_p: topP,
          },
          max_tokens: maxTokens,
          stream: true,
          onData: (chunk) => {
            if (isAbortRequested) {
              return;
            }
            const tokenText = chunk.choices?.[0]?.delta?.content || '';
            tokenCount++;
            const elapsedSec = (performance.now() - startTime) / 1000;
            const tps = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : 0;

            self.postMessage({
              status: 'update',
              content: tokenText,
              tps: Number(tps),
              tokenCount,
            });
          },
        });

        self.postMessage({ status: 'complete' });
      } catch (err) {
        if (!isAbortRequested) {
          self.postMessage({ status: 'error', error: err.message || String(err) });
        }
      }
      break;
    }

    case 'interrupt': {
      isAbortRequested = true;
      self.postMessage({ status: 'interrupted' });
      break;
    }

    case 'unload': {
      if (wllama) {
        await wllama.exit();
        wllama = null;
      }
      self.postMessage({ status: 'unloaded' });
      break;
    }
  }
});
