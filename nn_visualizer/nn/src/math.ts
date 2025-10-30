import type { Shape } from './types.js';

export function sizeOf(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function normalizeAxis(axis: number, rank: number): number {
  return axis < 0 ? axis + rank : axis;
}

export function normalizeAxes(axes: number | number[] | undefined, rank: number): number[] {
  if (axes === undefined || axes === null) return [];
  const arr = Array.isArray(axes) ? axes : [axes];
  return arr.map(a => normalizeAxis(a, rank)).sort((a, b) => a - b);
}

export function reshape(data: Float32Array | Int16Array, oldShape: Shape, newShape: Shape): Float32Array | Int16Array {
  if (sizeOf(oldShape) !== sizeOf(newShape)) {
    throw new Error(`Cannot reshape of size ${sizeOf(oldShape)} to ${sizeOf(newShape)}`);
  }
  // Views are fine; caller wraps in tensor object
  return data;
}

export function computeStrides(shape: Shape): number[] {
  const rank = shape.length;
  const strides = new Array(rank).fill(0);
  let acc = 1;
  for (let i = rank - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

export function indexToOffset(indices: number[], strides: number[]): number {
  return indices.reduce((sum, idx, i) => sum + idx * strides[i], 0);
}

export function padNHWC(
  input: Float32Array,
  shape: [number, number, number, number],
  pad: number
): { data: Float32Array; shape: [number, number, number, number] } {
  const [B, H, W, C] = shape;
  const outH = H + 2 * pad;
  const outW = W + 2 * pad;
  const out = new Float32Array(B * outH * outW * C);
  const inStrides = computeStrides(shape);
  const outStrides = computeStrides([B, outH, outW, C]);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        for (let c = 0; c < C; c++) {
          const vin = b * inStrides[0] + h * inStrides[1] + w * inStrides[2] + c;
          const vout = b * outStrides[0] + (h + pad) * outStrides[1] + (w + pad) * outStrides[2] + c;
          out[vout] = input[vin];
        }
      }
    }
  }
  return { data: out, shape: [B, outH, outW, C] };
}

export function slidingWindow2D(
  input: Float32Array,
  shape: [number, number, number, number], // NHWC
  window: number,
  stride: number
): { data: Float32Array; shape: [number, number, number, number, number, number] } {
  const [B, H, W, C] = shape;
  const outH = Math.floor((H - window) / stride) + 1;
  const outW = Math.floor((W - window) / stride) + 1;
  const outShape: [number, number, number, number, number, number] = [B, outH, outW, C, window, window];
  const out = new Float32Array(B * outH * outW * C * window * window);
  const inStrides = computeStrides(shape);
  const outStrides = computeStrides(outShape);

  for (let b = 0; b < B; b++) {
    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        const h0 = oh * stride;
        const w0 = ow * stride;
        for (let c = 0; c < C; c++) {
          for (let wh = 0; wh < window; wh++) {
            for (let ww = 0; ww < window; ww++) {
              const ih = h0 + wh;
              const iw = w0 + ww;
              const vin = b * inStrides[0] + ih * inStrides[1] + iw * inStrides[2] + c;
              const vout = b * outStrides[0] + oh * outStrides[1] + ow * outStrides[2] + c * outStrides[3] + wh * outStrides[4] + ww;
              out[vout] = input[vin];
            }
          }
        }
      }
    }
  }
  return { data: out, shape: outShape };
}

export function broadcastShapes(shapeA: Shape, shapeB: Shape): Shape {
  const rankA = shapeA.length;
  const rankB = shapeB.length;
  const maxRank = Math.max(rankA, rankB);
  const outShape: number[] = [];

  for (let i = 0; i < maxRank; i++) {
    const dimA = i < rankA ? shapeA[rankA - 1 - i] : 1;
    const dimB = i < rankB ? shapeB[rankB - 1 - i] : 1;

    if (dimA === dimB) {
      outShape.unshift(dimA);
    } else if (dimA === 1) {
      outShape.unshift(dimB);
    } else if (dimB === 1) {
      outShape.unshift(dimA);
    } else {
      throw new Error(`Cannot broadcast shapes [${shapeA.join(', ')}] and [${shapeB.join(', ')}]`);
    }
  }

  return outShape;
}

export function getBroadcastedValue(
  a: Float32Array,
  shapeA: Shape,
  stridesA: number[],
  b: Float32Array,
  shapeB: Shape,
  stridesB: number[],
  outShape: Shape,
  outStrides: number[],
  indices: number[]
): { a: number; b: number } {
  let idxA = 0;
  let idxB = 0;

  const rankA = shapeA.length;
  const rankB = shapeB.length;
  const rankOut = outShape.length;

  for (let i = 0; i < rankOut; i++) {
    const outIdx = indices[i];
    const outDim = outShape[i];

    // Map output index to input A
    const dimA = i < rankA ? shapeA[i] : 1;
    const aIdx = dimA === 1 ? 0 : outIdx;
    if (i < rankA) idxA += aIdx * stridesA[i];

    // Map output index to input B
    const dimB = i < rankB ? shapeB[i] : 1;
    const bIdx = dimB === 1 ? 0 : outIdx;
    if (i < rankB) idxB += bIdx * stridesB[i];
  }

  return { a: a[idxA] || 0, b: b[idxB] || 0 };
}

export function f32(x: number): number {
  return Math.fround(x);
}


