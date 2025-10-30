import { f32, normalizeAxes, padNHWC, sizeOf, slidingWindow2D, broadcastShapes, computeStrides } from './math.js';
import type { DType, Shape, TensorBase, TensorLike } from './types.js';

abstract class Tensor<TArray extends Float32Array | Int16Array> implements TensorBase<TArray> {
  abstract readonly dtype: DType;
  readonly data: TArray;
  readonly shape: Shape;

  constructor(raw: TArray, shape: Shape) {
    this.data = raw;
    this.shape = [...shape];
  }

  get size(): number { return sizeOf(this.shape); }

  abstract asFloat(): Float32Array;

  get(index: number | number[]): number {
    const idx = Array.isArray(index) ? index : [index];
    const strides = ((): number[] => {
      const s = new Array(this.shape.length);
      let acc = 1;
      for (let i = this.shape.length - 1; i >= 0; i--) {
        s[i] = acc; acc *= this.shape[i];
      }
      return s;
    })();
    let off = 0;
    for (let i = 0; i < idx.length; i++) off += idx[i] * strides[i];
    return this.asFloat()[off];
  }

  add(other: Tensor<TArray> | number): this {
    const o = other instanceof Tensor ? other : this.constructorFromNumber(other, [...this.shape]);
    return this.binaryOp(o, (a, b) => f32(a + b));
  }
  radd(other: Tensor<TArray> | number): this { return this.add(other); }

  sub(other: Tensor<TArray> | number): this {
    const o = other instanceof Tensor ? other : this.constructorFromNumber(other, [...this.shape]);
    return this.binaryOp(o, (a, b) => f32(a - b));
  }
  rsub(other: Tensor<TArray> | number): this {
    const o = other instanceof Tensor ? other : this.constructorFromNumber(other, [...this.shape]);
    return o.sub(this) as this;
  }

  neg(): this { return this.unaryOp(a => f32(-a)); }

  sum(axis?: number | number[], keepdims = false): this {
    if (axis === undefined || axis === null) {
      const s = this.asFloat().reduce((acc, v) => acc + v, 0);
      return this.constructorFromArray(new Float32Array([f32(s)]), keepdims ? [1] : []) as this;
    }
    const rank = this.shape.length;
    const axes = normalizeAxes(axis, rank);
    
    // Compute output shape (with reduced dimensions set to 1 if keepdims)
    const outShape = this.shape.slice();
    for (const ax of axes) outShape[ax] = 1;
    
    const out = new Float32Array(sizeOf(outShape));
    const inF = this.asFloat();
    const inStrides = ((): number[] => {
      const s = new Array(rank); let acc = 1; for (let i = rank - 1; i >= 0; i--) { s[i] = acc; acc *= this.shape[i]; } return s;
    })();
    const outStrides = ((): number[] => {
      const s = new Array(rank); let acc = 1; for (let i = rank - 1; i >= 0; i--) { s[i] = acc; acc *= outShape[i]; } return s;
    })();
    
    const coords = new Array(rank).fill(0);
    const outSize = sizeOf(outShape);
    for (let outIndex = 0; outIndex < outSize; outIndex++) {
      // Decode output index to coordinates (all dimensions, including reduced ones as 0)
      let rem = outIndex;
      for (let i = 0; i < rank; i++) {
        const s = outStrides[i];
        const v = Math.floor(rem / s);
        coords[i] = v;
        rem -= v * s;
      }
      
      // Iterate over reduced axes and sum
      let sumVal = 0;
      const iter = (axIdx: number) => {
        if (axIdx === axes.length) {
          // All reduced axes have been iterated, compute input offset
          let off = 0;
          for (let i = 0; i < rank; i++) {
            off += coords[i] * inStrides[i];
          }
          sumVal += inF[off];
          return;
        }
        const ax = axes[axIdx];
        const orig = coords[ax];
        // Iterate over all values in the reduced dimension
        for (let v = 0; v < this.shape[ax]; v++) {
          coords[ax] = v;
          iter(axIdx + 1);
        }
        coords[ax] = orig;
      };
      iter(0);
      out[outIndex] = f32(sumVal);
    }
    
    const finalShape = keepdims ? outShape : outShape.filter((d, i) => !axes.includes(i));
    // When filtering shape, we need to ensure the data matches
    // If keepdims=false, out has size sizeOf(outShape), but finalShape has fewer dimensions
    // However, sizeOf(outShape) == sizeOf(finalShape) because reduced dims are 1
    const finalData = keepdims ? out : out.slice(0, sizeOf(finalShape));
    return this.constructorFromArray(finalData, [...finalShape]) as this;
  }

  mean(axis?: number | number[], keepdims = false): this {
    if (axis === undefined || axis === null) {
      const n = this.size;
      const s = this.asFloat().reduce((acc, v) => acc + v, 0);
      return this.constructorFromArray(new Float32Array([f32(s / n)]), keepdims ? [1] : []) as this;
    }
    const rank = this.shape.length;
    const axes = normalizeAxes(axis, rank);
    const denom = axes.reduce((acc, ax) => acc * this.shape[ax], 1);
    
    // Compute output shape (with reduced dimensions set to 1 if keepdims)
    const outShape = this.shape.slice();
    for (const ax of axes) outShape[ax] = 1;
    
    const out = new Float32Array(sizeOf(outShape));
    const inF = this.asFloat();
    const inStrides = ((): number[] => {
      const s = new Array(rank); let acc = 1; for (let i = rank - 1; i >= 0; i--) { s[i] = acc; acc *= this.shape[i]; } return s;
    })();
    const outStrides = ((): number[] => {
      const s = new Array(rank); let acc = 1; for (let i = rank - 1; i >= 0; i--) { s[i] = acc; acc *= outShape[i]; } return s;
    })();
    
    const coords = new Array(rank).fill(0);
    const outSize = sizeOf(outShape);
    for (let outIndex = 0; outIndex < outSize; outIndex++) {
      // Decode output index to coordinates (all dimensions, including reduced ones as 0)
      let rem = outIndex;
      for (let i = 0; i < rank; i++) {
        const s = outStrides[i];
        const v = Math.floor(rem / s);
        coords[i] = v;
        rem -= v * s;
      }
      
      // Compute mean by dividing first, then summing (avoids intermediate values > 1)
      let meanVal = 0;
      const iter = (axIdx: number) => {
        if (axIdx === axes.length) {
          // All reduced axes have been iterated, compute input offset
          let off = 0;
          for (let i = 0; i < rank; i++) {
            off += coords[i] * inStrides[i];
          }
          // Divide by denom first, then add (keeps values in [-1, 1] range)
          meanVal += f32(inF[off] / denom);
          return;
        }
        const ax = axes[axIdx];
        const orig = coords[ax];
        // Iterate over all values in the reduced dimension
        for (let v = 0; v < this.shape[ax]; v++) {
          coords[ax] = v;
          iter(axIdx + 1);
        }
        coords[ax] = orig;
      };
      iter(0);
      out[outIndex] = meanVal;
    }
    
    const finalShape = keepdims ? outShape : outShape.filter((d, i) => !axes.includes(i));
    const finalData = keepdims ? out : out.slice(0, sizeOf(finalShape));
    return this.constructorFromArray(finalData, [...finalShape]) as this;
  }

  reshape(shape: number[]): this {
    if (sizeOf(shape) !== this.size) throw new Error('Invalid reshape size');
    return this.constructorFromArray(this.asFloat().slice(), shape) as this;
  }

  flatten(...axes: number[]): this {
    const rank = this.shape.length;
    const set = new Set(axes.map(a => (a < 0 ? a + rank : a)));
    const outShape: number[] = [];
    let remaining = 1;
    for (let i = 0; i < rank; i++) {
      if (set.has(i)) {
        remaining *= this.shape[i];
      } else {
        outShape.push(this.shape[i] * remaining);
        remaining = 1;
      }
    }
    if (remaining !== 1) {
      throw new Error(`Cannot flatten tensor with shape ${this.shape} along last dimension`);
    }
    return this.reshape(outShape);
  }

  unsqueeze(axis: number): this {
    const rank = this.shape.length;
    const ax = axis < 0 ? axis + rank + 1 : axis;
    const shape = [...this.shape];
    shape.splice(ax, 0, 1);
    return this.constructorFromArray(this.asFloat().slice(), shape) as this;
  }

  squeeze(axis: number): this {
    const rank = this.shape.length;
    const ax = axis < 0 ? axis + rank : axis;
    if (this.shape[ax] !== 1) throw new Error('Cannot squeeze non-1 dimension');
    const shape = [...this.shape];
    shape.splice(ax, 1);
    return this.constructorFromArray(this.asFloat().slice(), shape) as this;
  }

  sliding_window_2d(window: number, padding = 0, stride = 1): this {
    // Expect NHWC
    const [B, H, W, C] = this.shape as [number, number, number, number];
    if (this.shape.length !== 4) throw new Error('sliding_window_2d expects NHWC rank-4');
    const base = this.asFloat();
    const { data: padded, shape: pshape } = padding > 0 ? padNHWC(base, [B, H, W, C], padding) : { data: base, shape: [B, H, W, C] as [number, number, number, number] };
    const sw = slidingWindow2D(padded, pshape, window, stride);
    return this.constructorFromArray(sw.data, sw.shape) as this;
  }

  stack(...tensors: this[]): this {
    const rank = this.shape.length;
    const outShape = [tensors.length, ...this.shape];
    const out = new Float32Array(sizeOf(outShape));
    const singleSize = this.size;
    tensors.forEach((t, i) => out.set(t.asFloat(), i * singleSize));
    return this.constructorFromArray(out, outShape) as this;
  }

  protected abstract constructorFromArray(data: Float32Array, shape: number[]): this;
  protected abstract constructorFromNumber(value: number, shape: number[]): this;
  protected abstract unaryOp(f: (a: number) => number): this;
  protected abstract binaryOp(other: Tensor<TArray>, f: (a: number, b: number) => number): this;
}

const RING_MIN = -32768;

export class RingTensor extends Tensor<Int16Array> {
  readonly dtype: DType = 'int16';

  static rand(shape: number[]): RingTensor {
    const size = sizeOf(shape);
    const raw = new Int16Array(size);
    for (let i = 0; i < size; i++) raw[i] = (Math.random() * 65536 | 0) + RING_MIN; // wrap later
    return new RingTensor(raw, shape);
  }

  constructor(rawOrData: Int16Array | number[] | Float32Array, shape: number[]) {
    let raw: Int16Array;
    if (rawOrData instanceof Int16Array) {
      raw = rawOrData;
    } else {
      const arr = rawOrData instanceof Float32Array ? rawOrData : new Float32Array(rawOrData as number[]);
      // map [-1,1] to [min,max]
      raw = new Int16Array(arr.length);
      for (let i = 0; i < arr.length; i++) {
        let v = Math.round(arr[i] * -RING_MIN);
        raw[i] = toInt16(v);
      }
    }
    super(raw, shape);
  }

  asFloat(): Float32Array {
    const out = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) out[i] = this.data[i] / -RING_MIN;
    return out;
  }

  real(): RealTensor { return new RealTensor(this.asFloat(), [...this.shape]); }

  poly(order: number): RingTensor {
    const f = this.asFloat();
    const out = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const v = Math.sign(f[i]) * Math.pow(Math.abs(f[i]) + 1e-20, order);
      out[i] = toInt16(Math.round(v * -RING_MIN));
    }
    return new RingTensor(out, [...this.shape]);
  }

  poly_sigmoid(order: number, slope: number): RingTensor {
    const f = this.asFloat();
    const out = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const v = (1 + slope) * f[i] - slope * Math.sign(f[i]) * Math.pow(Math.abs(f[i]) + 1e-20, order);
      out[i] = toInt16(Math.round(v * -RING_MIN));
    }
    return new RingTensor(out, [...this.shape]);
  }

  sin2(): RingTensor {
    const f = this.asFloat();
    const out = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const v = (Math.sin(f[i] * Math.PI - Math.PI / 2) + 1) * Math.sign(f[i]) * 0.5;
      out[i] = toInt16(Math.round(v * -RING_MIN));
    }
    return new RingTensor(out, [...this.shape]);
  }

  cos(): RingTensor {
    const f = this.asFloat();
    const out = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const v = Math.cos(f[i] * Math.PI);
      out[i] = toInt16(Math.round(v * -RING_MIN));
    }
    return new RingTensor(out, [...this.shape]);
  }

  cos2(): RingTensor {
    const f = this.asFloat();
    const out = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const t = f[i] * Math.PI;
      const v = Math.cos(t) * 0.8 + Math.cos(t * 3) * 0.2;
      out[i] = toInt16(Math.round(v * -RING_MIN));
    }
    return new RingTensor(out, [...this.shape]);
  }

  protected constructorFromArray(data: Float32Array, shape: number[]): this {
    const raw = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) {
      // Quantize: map [-1, 1] to [RING_MIN, RING_MAX]
      // Don't clamp - let integer overflow/wraparound happen naturally (ring semantics)
      const quantized = Math.round(data[i] * -RING_MIN);
      raw[i] = toInt16(quantized);
    }
    return new RingTensor(raw, shape) as this;
  }
  protected constructorFromNumber(value: number, shape: number[]): this {
    const raw = new Int16Array(sizeOf(shape));
    const v = toInt16(Math.round(value * -RING_MIN));
    raw.fill(v);
    return new RingTensor(raw, shape) as this;
  }
  protected unaryOp(f: (a: number) => number): this {
    const inF = this.asFloat();
    const out = new Float32Array(inF.length);
    for (let i = 0; i < inF.length; i++) out[i] = f(inF[i]);
    return this.constructorFromArray(out, [...this.shape]);
  }
  protected binaryOp(other: Tensor<Int16Array>, f: (a: number, b: number) => number): this {
    const a = this.asFloat(); const b = other.asFloat();
    
    // If shapes match exactly, use simple element-wise operation
    if (a.length === b.length && JSON.stringify(this.shape) === JSON.stringify(other.shape)) {
      const out = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) out[i] = f(a[i], b[i]);
      return this.constructorFromArray(out, [...this.shape]);
    }
    
    // Otherwise, use broadcasting
    const outShape = broadcastShapes(this.shape, other.shape);
    const outSize = sizeOf(outShape);
    const out = new Float32Array(outSize);
    const aStrides = computeStrides(this.shape);
    const bStrides = computeStrides(other.shape);
    const outStrides = computeStrides(outShape);
    
    // Iterate over output indices
    const indices = new Array(outShape.length).fill(0);
    for (let outIdx = 0; outIdx < outSize; outIdx++) {
      // Decode output index to coordinates
      let rem = outIdx;
      for (let i = 0; i < outShape.length; i++) {
        indices[i] = Math.floor(rem / outStrides[i]);
        rem %= outStrides[i];
      }
      
      // Map to input indices (align from right for broadcasting)
      let aIdx = 0;
      let bIdx = 0;
      const rankA = this.shape.length;
      const rankB = other.shape.length;
      const rankOut = outShape.length;
      
      for (let i = 0; i < rankOut; i++) {
        const coord = indices[i];
        
        // Output dimension i maps to input dimensions counting from the right
        // Position from right: rankOut - 1 - i
        const posFromRight = rankOut - 1 - i;
        
        // Map to input A dimension from right
        if (posFromRight < rankA) {
          const aInputDimIdx = rankA - 1 - posFromRight;
          const dimA = this.shape[aInputDimIdx];
          // If dimension size is 1, broadcast (use coordinate 0)
          const aCoord = dimA === 1 ? 0 : coord;
          aIdx += aCoord * aStrides[aInputDimIdx];
        } else {
          // A doesn't have this dimension (it's broadcast), use coordinate 0
          // No need to add anything since it's broadcast
        }
        
        // Map to input B dimension from right
        if (posFromRight < rankB) {
          const bInputDimIdx = rankB - 1 - posFromRight;
          const dimB = other.shape[bInputDimIdx];
          // If dimension size is 1, broadcast (use coordinate 0)
          const bCoord = dimB === 1 ? 0 : coord;
          bIdx += bCoord * bStrides[bInputDimIdx];
        } else {
          // B doesn't have this dimension (it's broadcast), use coordinate 0
          // No need to add anything since it's broadcast
        }
      }
      
      out[outIdx] = f(a[aIdx] || 0, b[bIdx] || 0);
    }
    
    return this.constructorFromArray(out, [...outShape]);
  }
}

export class RealTensor extends Tensor<Float32Array> {
  readonly dtype: DType = 'float32';

  constructor(data: Float32Array | number[], shape: number[]) {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    super(arr, shape);
  }

  static full(shape: number[], value: number): RealTensor {
    const arr = new Float32Array(sizeOf(shape)); arr.fill(f32(value));
    return new RealTensor(arr, shape);
  }

  asFloat(): Float32Array { return this.data; }

  mul(other: RealTensor | number): RealTensor {
    const o = other instanceof RealTensor ? other : new RealTensor(new Float32Array(this.size).fill(f32(other)), [...this.shape]);
    const a = this.asFloat(); const b = o.asFloat();
    const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) out[i] = f32(a[i] * b[i]);
    return new RealTensor(out, [...this.shape]);
  }

  div(other: RealTensor | number): RealTensor {
    const o = other instanceof RealTensor ? other : new RealTensor(new Float32Array(this.size).fill(f32(other)), [...this.shape]);
    const a = this.asFloat(); const b = o.asFloat();
    const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) out[i] = f32(a[i] / b[i]);
    return new RealTensor(out, [...this.shape]);
  }

  pow(other: RealTensor | number): RealTensor {
    const o = other instanceof RealTensor ? other : new RealTensor(new Float32Array(this.size).fill(f32(other)), [...this.shape]);
    const a = this.asFloat(); const b = o.asFloat();
    const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) out[i] = f32(Math.pow(a[i], b[i]));
    return new RealTensor(out, [...this.shape]);
  }

  abs(): RealTensor {
    const a = this.asFloat();
    const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.abs(a[i]);
    return new RealTensor(out, [...this.shape]);
  }

  cross_entropy(other: RealTensor): RealTensor {
    const logits = this.asFloat();
    const tgt = other.asFloat();
    // reshape to (N, -1)
    const N = this.shape[0];
    const K = logits.length / N;
    const out = new Float32Array(1);
    let loss = 0;
    for (let n = 0; n < N; n++) {
      // stable softmax
      let maxv = -Infinity;
      for (let k = 0; k < K; k++) maxv = Math.max(maxv, logits[n * K + k]);
      let sum = 0;
      const exps = new Float32Array(K);
      for (let k = 0; k < K; k++) { const v = Math.exp(logits[n * K + k] - maxv); exps[k] = v; sum += v; }
      for (let k = 0; k < K; k++) exps[k] = exps[k] / sum;
      for (let k = 0; k < K; k++) loss += -tgt[n * K + k] * Math.log(exps[k] + 1e-20);
    }
    out[0] = f32(loss / N);
    return new RealTensor(out, []);
  }

  protected constructorFromArray(data: Float32Array, shape: number[]): this {
    return new RealTensor(data, shape) as this;
  }
  protected constructorFromNumber(value: number, shape: number[]): this {
    const out = new Float32Array(sizeOf(shape)).fill(f32(value));
    return new RealTensor(out, shape) as this;
  }
  protected unaryOp(f: (a: number) => number): this {
    const a = this.asFloat(); const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) out[i] = f(a[i]);
    return new RealTensor(out, [...this.shape]) as this;
  }
  protected binaryOp(other: Tensor<Float32Array>, f: (a: number, b: number) => number): this {
    const a = this.asFloat(); const b = other.asFloat();
    
    // If shapes match exactly, use simple element-wise operation
    if (a.length === b.length && JSON.stringify(this.shape) === JSON.stringify(other.shape)) {
      const out = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) out[i] = f(a[i], b[i]);
      return new RealTensor(out, [...this.shape]) as this;
    }
    
    // Otherwise, use broadcasting
    const outShape = broadcastShapes(this.shape, other.shape);
    const outSize = sizeOf(outShape);
    const out = new Float32Array(outSize);
    const aStrides = computeStrides(this.shape);
    const bStrides = computeStrides(other.shape);
    const outStrides = computeStrides(outShape);
    
    // Iterate over output indices
    const indices = new Array(outShape.length).fill(0);
    for (let outIdx = 0; outIdx < outSize; outIdx++) {
      // Decode output index to coordinates
      let rem = outIdx;
      for (let i = 0; i < outShape.length; i++) {
        indices[i] = Math.floor(rem / outStrides[i]);
        rem %= outStrides[i];
      }
      
      // Map to input indices (align from right for broadcasting)
      let aIdx = 0;
      let bIdx = 0;
      const rankA = this.shape.length;
      const rankB = other.shape.length;
      const rankOut = outShape.length;
      
      for (let i = 0; i < rankOut; i++) {
        const coord = indices[i];
        
        // Output dimension i maps to input dimensions counting from the right
        // Position from right: rankOut - 1 - i
        const posFromRight = rankOut - 1 - i;
        
        // Map to input A dimension from right
        if (posFromRight < rankA) {
          const aInputDimIdx = rankA - 1 - posFromRight;
          const dimA = this.shape[aInputDimIdx];
          // If dimension size is 1, broadcast (use coordinate 0)
          const aCoord = dimA === 1 ? 0 : coord;
          aIdx += aCoord * aStrides[aInputDimIdx];
        } else {
          // A doesn't have this dimension (it's broadcast), use coordinate 0
          // No need to add anything since it's broadcast
        }
        
        // Map to input B dimension from right
        if (posFromRight < rankB) {
          const bInputDimIdx = rankB - 1 - posFromRight;
          const dimB = other.shape[bInputDimIdx];
          // If dimension size is 1, broadcast (use coordinate 0)
          const bCoord = dimB === 1 ? 0 : coord;
          bIdx += bCoord * bStrides[bInputDimIdx];
        } else {
          // B doesn't have this dimension (it's broadcast), use coordinate 0
          // No need to add anything since it's broadcast
        }
      }
      
      out[outIdx] = f(a[aIdx] || 0, b[bIdx] || 0);
    }
    
    return new RealTensor(out, [...outShape]) as this;
  }
}

function toInt16(x: number): number {
  let v = x | 0;
  v = ((v + 32768) & 0xFFFF) - 32768; // wrap to int16
  return v;
}

export { Tensor };


