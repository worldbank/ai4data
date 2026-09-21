import { AutoTokenizer, AutoModel } from '@huggingface/transformers';

console.log('=== Testing GLiNER with Transformers.js AutoModel (q4) ===');

async function testTransformersJsGliner() {
  try {
    console.log('[1/3] Loading AutoTokenizer for onnx-community/gliner_large-v2.1...');
    const tokenizer = await AutoTokenizer.from_pretrained('onnx-community/gliner_large-v2.1');
    console.log('Tokenizer loaded successfully!');

    console.log('[2/3] Loading AutoModel (q4) for onnx-community/gliner_large-v2.1...');
    const model = await AutoModel.from_pretrained('onnx-community/gliner_large-v2.1', {
      dtype: 'q4',
    });
    console.log('AutoModel (q4) loaded successfully!');

    const prompt = '<<ENT>> company <<ENT>> person <<ENT>> product <<ENT>> location <<SEP>> Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.';
    const inputs = tokenizer(prompt);
    console.log('Tokenized input IDs using accurate special tokens:', Array.from(inputs.input_ids.data));
    console.log('Tokenized inputs keys:', Object.keys(inputs));

    const outputs = await model(inputs);
    console.log('\n======================================================');
    console.log('🎉 Transformers.js AutoModel Forward Pass Successful!');
    console.log('Output Keys:', Object.keys(outputs));
    console.log('======================================================');
  } catch (err) {
    console.error('Transformers.js AutoModel Error:', err);
  }
}

testTransformersJsGliner();
