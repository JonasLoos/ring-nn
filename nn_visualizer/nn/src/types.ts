export type Shape = readonly number[];

export type DType = 'int16' | 'float32';

export type TensorLike = number | number[] | Float32Array | Int16Array | TensorLike[];

export interface TensorBase<TArray extends ArrayLike<number>> {
  readonly dtype: DType;
  readonly shape: Shape;
  readonly size: number;
  readonly data: TArray;
  asFloat(): Float32Array;
}

export interface SerializedTensorMeta {
  dtype: DType;
  shape: number[];
  byteOffset: number;
  byteLength: number;
}

export interface SerializedHeaderV1 {
  version: 1;
  tensors: SerializedTensorMeta[];
}

// Union type for RingTensor | RealTensor
export type AnyTensor = import('./tensor.js').RingTensor | import('./tensor.js').RealTensor;

