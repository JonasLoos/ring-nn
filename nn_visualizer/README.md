# NN Visualizer

This directory contains the TypeScript Ring NN library and an interactive web visualizer for neural networks.

## Components

### TypeScript Library (`nn/`)

A browser-compatible ESM implementation of the Python `tensor.py` and `nn.py` APIs.

### Interactive Visualizer (`visualizer.html`)

A web application for visualizing neural network activations and outputs. Allows you to:
- Draw/edit 2D input images on a canvas
- Load trained models (`.bin` format)
- Visualize intermediate layer activations
- View classification outputs

## Setup

### 1. Build the TypeScript Package

```bash
cd nn
npm run build
```

### 2. Compile Visualizer TypeScript

```bash
cd ..
npx tsc src/visualizer.ts --outDir src --target ES2020 --module ES2020 --moduleResolution node --esModuleInterop --resolveJsonModule --skipLibCheck
```

### 3. Start the Flask Backend (for model conversion)

From the project root:

```bash
python app.py
```

This starts a Flask server on `http://localhost:5000` that provides:
- `/list-models` - List available `.pkl` model files
- `/convert-model?model_file=<name>` - Convert `.pkl` to `.bin` format

### 4. Serve the Visualizer

You need a local web server (required for ES modules):

**Using Python:**
```bash
python3 -m http.server 8000
```

**Using Node.js (http-server):**
```bash
npx http-server -p 8000
```

**Using Node.js (serve):**
```bash
npx serve
```

### 5. Open the Visualizer

Navigate to `http://localhost:8000/visualizer.html` (or `http://localhost:8000/visualizer.html` if using the Flask server)

## Using the Visualizer

### Loading a Model

1. **From Server**: Select a model from the dropdown (requires Flask backend running)
2. **From File**: Click "Choose File" and select a `.bin` model file

### Drawing Input Images

- Click and drag on the canvas to draw
- Adjust brush size with the slider
- Click "Clear" to reset the canvas
- The input preview shows the 28×28 grayscale image that will be fed to the network

### Viewing Activations

- Activations update automatically as you draw (debounced by 200ms)
- Each layer shows:
  - **Conv layers**: Grid of feature maps (one per channel)
  - **FF layers**: Bar chart of activations
- Final output shows classification probabilities with predicted class highlighted

### Model Architecture

The visualizer currently expects MNIST-like models with this structure:
```typescript
Input([1, 28, 28, 1])
  .conv(4, 2, 0, 2)
  .conv(8, 4, 0, 2)
  .flatten(1, 2)
  .ff(10)
  .apply((x) => 0.5 + x.cos().real()/2)
```

To use a different architecture, modify the `loadModelFromFile` function in `visualizer.html`.

## Converting Python Models

### Using the Flask Endpoint

1. Place your trained `.pkl` model file in the `logs/` directory
2. Access: `http://localhost:5000/convert-model?model_file=<model_name>.pkl`
3. Download the `.bin` file
4. Load it in the visualizer

### Manual Conversion

You can also convert models programmatically using Python:

```python
from nn import Input
from nn_visualizer.nn import saveToBlob  # Hypothetical - you'd need to implement this

# Load your model
model = Input(...)
model.load('path/to/model.pkl')

# Convert to .bin format
blob = saveToBlob(model)
with open('model.bin', 'wb') as f:
    f.write(blob)
```

## Test Page (`test.html`)

The test page automatically runs comprehensive tests covering:

- **Basic Tensor Operations**: shape, size, add, neg, sum, mean, reshape, unsqueeze
- **Ring Tensor Operations**: cos, cos2, poly, sin2, real conversion
- **Model Building & Forward Pass**: RingFF, RingConv, Sequential models
- **Serialization**: save/load operations with Blob and ArrayBuffer
- **Full Model Example**: Complete MNIST-like model with multiple layers

Open `http://localhost:8000/test.html` to run tests.

## File Structure

```
nn_visualizer/
├── nn/                  # TypeScript package
│   ├── src/             # Source files
│   ├── dist/             # Built ESM output (generated)
│   └── package.json
├── src/
│   ├── visualizer.ts     # Visualization utilities source
│   └── visualizer.js     # Compiled visualization utilities
├── visualizer.html       # Main visualizer webapp
├── test.html             # Test page
└── README.md             # This file
```

## Troubleshooting

### Model loading fails

- Ensure the model architecture matches what's expected in `visualizer.html`
- Check that weights are in the correct order
- Verify the `.bin` file was created correctly

### Visualizations not updating

- Check browser console for errors
- Ensure the model is loaded successfully
- Verify the canvas has content (not all black)

### CORS errors

- Make sure you're using a local web server (not `file://` protocol)
- If using Flask backend, ensure both servers are running
