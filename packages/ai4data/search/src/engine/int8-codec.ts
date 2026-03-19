/**
 * int8-codec.ts
 *
 * Quantization scheme: vectors are stored as Int8 values in [-127, 127].
 * Each vector is accompanied by a scalar `scale` such that the original
 * float value ≈ int8_value * scale.  Dot products are computed in mixed
 * precision (Float32 query × dequantized Int8 stored vector) to keep
 * both accuracy and memory efficiency.
 */

/**
 * Compute the dot product between a Float32 query vector and a stored
 * Int8-quantized vector, dequantizing on the fly.
 *
 * @param queryF32   - L2-normalised query vector (Float32Array)
 * @param storedQV   - Int8-quantized stored vector
 * @param storedScale - Per-vector dequantization scale factor
 * @returns Approximate cosine similarity score
 */
export function dotProductMixed(
  queryF32: Float32Array,
  storedQV: Int8Array,
  storedScale: number,
): number {
  let dot = 0.0;
  const len = queryF32.length;
  for (let i = 0; i < len; i++) {
    dot += queryF32[i] * (storedQV[i] * storedScale);
  }
  return dot;
}

/**
 * Dequantize an Int8 vector back to Float32 using the stored scale.
 *
 * @param qv    - Int8-quantized vector
 * @param scale - Per-vector dequantization scale factor
 * @returns Reconstructed Float32 vector
 */
export function dequantize(qv: Int8Array, scale: number): Float32Array {
  const out = new Float32Array(qv.length);
  for (let i = 0; i < qv.length; i++) {
    out[i] = qv[i] * scale;
  }
  return out;
}

/**
 * L2-normalise a Float32 vector in place.
 * Vectors whose norm is below 1e-9 are left unchanged to avoid division by zero.
 *
 * @param vec - Vector to normalise (mutated in place)
 * @returns The same (now normalised) vector
 */
export function l2NormalizeInPlace(vec: Float32Array): Float32Array {
  let norm = 0.0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-9) return vec;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

/**
 * Convert a plain number array (or an existing Int8Array) to an Int8Array.
 * Values outside [-128, 127] are silently truncated by the typed-array constructor.
 *
 * @param arr - Source values
 * @returns An Int8Array view / copy of the input
 */
export function toInt8Array(arr: number[] | Int8Array): Int8Array {
  return new Int8Array(arr);
}
