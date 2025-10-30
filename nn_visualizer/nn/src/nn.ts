import { RingTensor, RealTensor, Tensor } from './tensor.js';
import type { AnyTensor } from './types.js';

export abstract class Model {
  abstract get weights(): AnyTensor[];
  abstract get nparams(): number;
  abstract forward(x: AnyTensor): AnyTensor;
  call(x: AnyTensor): AnyTensor { return this.forward(x); }
}

export abstract class Module extends Model {
  protected _weights: AnyTensor[] = [];

  get weights(): AnyTensor[] {
    const own = this._weights ?? [];
    const nested: AnyTensor[] = [];
    for (const v of Object.values(this as any)) {
      if (v instanceof Module) nested.push(...v.weights);
    }
    return [...own, ...nested];
  }
  get nparams(): number { return this.weights.reduce((a, w) => a + w.size, 0); }
  forward(_x: AnyTensor): AnyTensor { throw new Error('Not implemented'); }
}

export class RingFF extends Module {
  constructor(inputSize: number, outputSize: number) {
    super();
    this._weights = [RingTensor.rand([inputSize, outputSize])];
  }
  forward(x: RingTensor): RingTensor {
    // (x.unsqueeze(-1) - W).cos().mean(axis=-2)
    const expanded = x.unsqueeze(-1) as RingTensor;
    const diff = expanded.sub(this._weights[0] as RingTensor) as RingTensor;
    const act = diff.cos();
    return act.mean(-2) as RingTensor;
  }
  toString(): string {
    const w = this._weights[0];
    return `RingFF(input_size=${w.shape[0]}, output_size=${w.shape[1]})`;
  }
}

export class RingConv extends Module {
  private window: number;
  private padding: number;
  private stride: number;
  constructor(inputChannels: number, outputChannels: number, windowSize = 3, padding = 1, stride = 1) {
    super();
    this.window = windowSize; this.padding = padding; this.stride = stride;
    this._weights = [RingTensor.rand([inputChannels, windowSize, windowSize, outputChannels])];
  }
  forward(x: RingTensor): RingTensor {
    // (x.sliding_window_2d(ws,pad,stride).unsqueeze(-1) - W).cos().mean(axis=(-4,-3,-2))
    const sw = x.sliding_window_2d(this.window, this.padding, this.stride) as RingTensor;
    const expanded = sw.unsqueeze(-1) as RingTensor;
    const diff = expanded.sub(this._weights[0] as RingTensor) as RingTensor;
    const act = diff.cos();
    // mean over last 3 window dims: (-4,-3,-2) - use array to mean over multiple axes at once
    // After unsqueeze(-1), shape is (B, H', W', C, ws, ws, 1)
    // After subtraction with weights (C, ws, ws, O), shape is (B, H', W', C, ws, ws, O)
    // So -4,-3,-2 refer to dimensions C, ws, ws (indices 3, 4, 5 in rank-7 tensor)
    const rank = act.shape.length;
    const axes = [rank - 4, rank - 3, rank - 2]; // Map negative indices to positive
    return act.mean(axes, false) as RingTensor;
  }
  toString(): string {
    const w = this._weights[0];
    return `RingConv(input_size=${w.shape[0]}, output_size=${w.shape[3]}, window_size=${this.window}, padding=${this.padding}, stride=${this.stride})`;
  }
}

export class Sequential extends Module {
  constructor(readonly modules: (Module | ((x: any) => any))[]) { super(); }
  get weights(): AnyTensor[] { return this.modules.flatMap(m => m instanceof Module ? m.weights : []); }
  forward(x: any, onLayerOutput?: (layerIdx: number, name: string, output: any) => void): any {
    let value = x;
    for (let i = 0; i < this.modules.length; i++) {
      const m = this.modules[i];
      value = m instanceof Module ? m.forward(value) : m(value);
      if (onLayerOutput) {
        const name = m instanceof Module ? m.toString() : String(m);
        onLayerOutput(i, name, value);
      }
    }
    return value;
  }
  toString(): string { return this.modules.map(m => (m instanceof Module ? m.toString() : String(m))).join('\n-> '); }
}

export class Partial<F extends (x: any, ...args: any[]) => any> {
  constructor(public fn: F, public args: any[] = [], public kwargs: Record<string, any> = {}) {}
  call = (x: any) => this.fn(x, ...this.args);
  toString(): string {
    const argsRepr = [...this.args.map(a => JSON.stringify(a)), ...Object.entries(this.kwargs).map(([k, v]) => `${k}=${JSON.stringify(v)}`)].join(', ');
    return `${this.fn.name}(${argsRepr})`;
  }
}

export class Input extends Model {
  private _inputShape: number[];
  private _shape: number[];
  private _network: Sequential;

  constructor(shape: number[]) {
    super();
    this._inputShape = [...shape];
    this._shape = [...shape];
    this._network = new Sequential([]);
  }
  get weights(): AnyTensor[] { return this._network.weights; }
  get nparams(): number { return this._network.nparams; }
  forward(x: any, onLayerOutput?: (layerIdx: number, name: string, output: any) => void): any {
    return this._network.forward(x, onLayerOutput);
  }

  private addFn<T extends AnyTensor>(fn: (x: any) => T): this {
    // Infer output shape by running on a dummy tensor
    const dummy = RingTensor.rand(this._shape);
    const out = fn(dummy);
    this._shape = [...out.shape];
    this._network.modules.push(fn as any);
    return this;
  }

  ff(outputSize: number): this {
    const inputSize = this._shape[this._shape.length - 1];
    const module = new RingFF(inputSize, outputSize);
    this._network.modules.push(module);
    const dummy = RingTensor.rand(this._shape);
    const out = module.forward(dummy);
    this._shape = [...out.shape];
    return this;
  }

  conv(outputSize: number, windowSize = 3, padding = 1, stride = 1): this {
    const inputChannels = this._shape[this._shape.length - 1];
    const module = new RingConv(inputChannels, outputSize, windowSize, padding, stride);
    this._network.modules.push(module);
    const dummy = RingTensor.rand(this._shape);
    const out = module.forward(dummy);
    this._shape = [...out.shape];
    return this;
  }

  apply(fn: (x: RingTensor) => AnyTensor): this { return this.addFn(fn); }
  flatten(...axes: number[]): this { return this.addFn((x: RingTensor) => x.flatten(...axes)); }

  toString(): string {
    return `Input(shape=${JSON.stringify(this._inputShape)})\n-> ${this._network.toString()}\n-> Output(shape=${JSON.stringify(this._shape)})`;
  }
}

export { RingTensor, RealTensor, Tensor };
export type { AnyTensor } from './types.js';


