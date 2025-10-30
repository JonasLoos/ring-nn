import { RingTensor, RealTensor } from './tensor.js';
import type { SerializedHeaderV1, SerializedTensorMeta, AnyTensor } from './types.js';

// Binary format:
// [Uint32 headerLengthBytes][UTF-8 JSON header][raw payloads...]

function encodeUTF8(str: string): Uint8Array {
  return new TextEncoder().encode(str);
}
function decodeUTF8(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

export function collectWeights(module: { weights: AnyTensor[] }): { meta: SerializedTensorMeta[]; payload: ArrayBuffer } {
  const metas: SerializedTensorMeta[] = [];
  const payloads: Uint8Array[] = [];
  let offset = 0;
  for (const w of module.weights) {
    let buf: ArrayBuffer;
    if (w instanceof RingTensor) buf = new Int16Array(w.data).buffer;
    else if (w instanceof RealTensor) buf = new Float32Array(w.data).buffer;
    else throw new Error('Unknown tensor kind');
    const bytes = new Uint8Array(buf.slice(0));
    payloads.push(bytes);
    metas.push({ dtype: w.dtype, shape: [...w.shape], byteOffset: offset, byteLength: bytes.byteLength });
    offset += bytes.byteLength;
  }
  const joined = new Uint8Array(offset);
  let ptr = 0;
  for (const p of payloads) { joined.set(p, ptr); ptr += p.byteLength; }
  return { meta: metas, payload: joined.buffer };
}

export function saveToBlob(module: { weights: AnyTensor[] }): Blob {
  const { meta, payload } = collectWeights(module);
  const header: SerializedHeaderV1 = { version: 1, tensors: meta };
  const headerBytes = encodeUTF8(JSON.stringify(header));
  const len = new Uint32Array([headerBytes.byteLength]);
  const headerArrayBuffer = headerBytes.buffer instanceof ArrayBuffer
    ? headerBytes.buffer.slice(headerBytes.byteOffset, headerBytes.byteOffset + headerBytes.byteLength)
    : new Uint8Array(headerBytes).buffer;
  return new Blob([len.buffer, headerArrayBuffer, payload], { type: 'application/octet-stream' });
}

export async function loadFromArrayBuffer<T extends { weights: AnyTensor[] }>(module: T, ab: ArrayBuffer): Promise<T> {
  const u8 = new Uint8Array(ab);
  const len = new DataView(ab, 0, 4).getUint32(0, true);
  const headerBytes = u8.slice(4, 4 + len);
  const header: SerializedHeaderV1 = JSON.parse(decodeUTF8(headerBytes));
  const payload = u8.slice(4 + len).buffer;
  const view = new DataView(payload);
  for (let i = 0; i < module.weights.length; i++) {
    const w = module.weights[i];
    const m = header.tensors[i];
    const slice = payload.slice(m.byteOffset, m.byteOffset + m.byteLength);
    if (w instanceof RingTensor) {
      const arr = new Int16Array(slice);
      const wData = (w as any).data as Int16Array;
      if (arr.length !== wData.length) {
        throw new Error(`Weight ${i} size mismatch: expected ${wData.length}, got ${arr.length}`);
      }
      wData.set(arr);
    } else if (w instanceof RealTensor) {
      const arr = new Float32Array(slice);
      const wData = (w as any).data as Float32Array;
      if (arr.length !== wData.length) {
        throw new Error(`Weight ${i} size mismatch: expected ${wData.length}, got ${arr.length}`);
      }
      wData.set(arr);
    } else {
      throw new Error('Unknown tensor kind');
    }
  }
  return module;
}


