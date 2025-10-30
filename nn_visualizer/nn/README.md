# nn (Browser ESM)

Forward-only TypeScript reimplementation of tensor.py and nn.py with Int16 ring semantics and binary .bin weight serialization. No backward pass.

## Install / Build

```bash
npm run build
```

## Usage (Browser)

```html
<script type="module">
  import { RingTensor, Input, saveToBlob, loadFromArrayBuffer } from './dist/index.js';

  // Build a tiny model
  const model = new Input([1, 28, 28, 1])
    .conv(8, 3, 1, 1)
    .flatten(1, 2)
    .ff(10);

  // Forward
  const x = RingTensor.rand([1, 28, 28, 1]);
  const y = model.forward(x);
  console.log('out shape', y.shape);

  // Save weights
  const blob = saveToBlob(model);
  // ...persist blob or download...

  // Load weights
  const ab = await blob.arrayBuffer();
  await loadFromArrayBuffer(model, ab);
</script>
```

## API Highlights
- RingTensor (Int16) with cos, cos2, poly, poly_sigmoid, sin2, real()
- RealTensor (Float32) with mul, div, pow, abs, cross_entropy
- Layers: RingFF, RingConv, Sequential, Input
- Serialization: saveToBlob, loadFromArrayBuffer

## Notes
- All ops are forward-only, no gradients.
- Shapes are NHWC for images.
