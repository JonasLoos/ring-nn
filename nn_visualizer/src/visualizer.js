export class Visualizer {
    constructor(model) {
        this.model = model;
    }
    /**
     * Format layer name for display - handles anonymous functions and long names
     */
    static formatLayerName(name) {
        console.log(name);
        const result = name.split('\n').map(line => line.trim()).filter(line => line.length > 0 && !line.startsWith('//')).join(' ');
        console.log(result);
        return result;
    }
    /**
     * Convert tensor to ImageData for canvas rendering
     * For shape [1, H, W, C], extracts a single channel or shows mean
     */
    static tensorToImageData(tensor, channel, scale = 1) {
        const floatData = tensor.asFloat();
        const shape = tensor.shape;
        const isRingTensor = tensor.dtype === 'int16';
        // Handle different tensor shapes
        let H, W, C;
        let dataOffset = 0;
        if (shape.length === 4 && shape[0] === 1) {
            // NHWC format: [1, H, W, C]
            [H, W, C] = [shape[1], shape[2], shape[3]];
            dataOffset = 0;
        }
        else if (shape.length === 2 && shape[0] === 1) {
            // Flattened 2D: [1, N] - reshape to square if possible
            const N = shape[1];
            const side = Math.ceil(Math.sqrt(N));
            H = W = side;
            C = 1;
            dataOffset = 0;
        }
        else if (shape.length === 3 && shape[0] === 1) {
            // [1, H, W]
            H = shape[1];
            W = shape[2];
            C = 1;
            dataOffset = 0;
        }
        else {
            throw new Error(`Unsupported tensor shape for visualization: ${shape.join(',')}`);
        }
        const canvas = document.createElement('canvas');
        canvas.width = W * scale;
        canvas.height = H * scale;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        // Normalize values to [0, 1] for display
        const values = [];
        for (let i = 0; i < floatData.length; i++) {
            values.push(floatData[i]);
        }
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal || 1;
        // Extract channel or compute mean
        let pixelValues = [];
        if (C === 1) {
            // Single channel
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const idx = (h * W + w);
                    pixelValues.push(floatData[dataOffset + idx]);
                }
            }
        }
        else if (channel !== undefined) {
            // Specific channel
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const idx = (h * W + w) * C + channel;
                    pixelValues.push(floatData[dataOffset + idx]);
                }
            }
        }
        else {
            // Mean across channels
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    let sum = 0;
                    for (let c = 0; c < C; c++) {
                        const idx = (h * W + w) * C + c;
                        sum += floatData[dataOffset + idx];
                    }
                    pixelValues.push(sum / C);
                }
            }
        }
        // Fill ImageData - cyclic color for RingTensor, grayscale for RealTensor
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const val = pixelValues[h * W + w];
                const normalized = (val - minVal) / range;
                let r, g, b;
                if (isRingTensor) {
                    // Cyclic color scheme: map normalized value to hue (0-1)
                    // Use full saturation and value for vibrant colors
                    const hue = normalized; // Value maps directly to hue position on color wheel
                    [r, g, b] = this.hsvToRgb(hue, 1.0, 1.0);
                }
                else {
                    // Grayscale for RealTensor
                    const gray = Math.max(0, Math.min(255, Math.round(normalized * 255)));
                    r = g = b = gray;
                }
                // Scale up pixels
                for (let sh = 0; sh < scale; sh++) {
                    for (let sw = 0; sw < scale; sw++) {
                        const x = w * scale + sw;
                        const y = h * scale + sh;
                        const idx = (y * canvas.width + x) * 4;
                        imageData.data[idx] = r; // R
                        imageData.data[idx + 1] = g; // G
                        imageData.data[idx + 2] = b; // B
                        imageData.data[idx + 3] = 255; // A
                    }
                }
            }
        }
        return imageData;
    }
    /**
     * Convert HSV to RGB
     */
    static hsvToRgb(h, s, v) {
        const c = v * s;
        const x = c * (1 - Math.abs((h * 6) % 2 - 1));
        const m = v - c;
        let r = 0, g = 0, b = 0;
        if (h * 6 < 1) {
            r = c;
            g = x;
            b = 0;
        }
        else if (h * 6 < 2) {
            r = x;
            g = c;
            b = 0;
        }
        else if (h * 6 < 3) {
            r = 0;
            g = c;
            b = x;
        }
        else if (h * 6 < 4) {
            r = 0;
            g = x;
            b = c;
        }
        else if (h * 6 < 5) {
            r = x;
            g = 0;
            b = c;
        }
        else {
            r = c;
            g = 0;
            b = x;
        }
        return [
            Math.round((r + m) * 255),
            Math.round((g + m) * 255),
            Math.round((b + m) * 255)
        ];
    }
    /**
     * Draw activation grids for multi-channel tensors
     */
    static drawActivationsGrid(activation, container) {
        const tensor = activation.output;
        const shape = tensor.shape;
        // Clear container
        container.innerHTML = '';
        const header = document.createElement('div');
        header.style.marginBottom = '10px';
        header.style.fontWeight = 'bold';
        const formattedName = this.formatLayerName(activation.name);
        header.textContent = `Layer ${activation.layerIdx}: ${formattedName} (shape: [${shape.join(', ')}])`;
        container.appendChild(header);
        // Handle different tensor shapes
        if (shape.length === 4 && shape[0] === 1) {
            // NHWC: [1, H, W, C] - show all channels
            const C = shape[3];
            const H = shape[1];
            const W = shape[2];
            const grid = document.createElement('div');
            grid.style.display = 'grid';
            grid.style.gap = '5px';
            grid.style.gridTemplateColumns = `repeat(${Math.ceil(Math.sqrt(C))}, auto)`;
            grid.style.padding = '10px';
            grid.style.border = '1px solid #ccc';
            grid.style.borderRadius = '4px';
            for (let c = 0; c < C; c++) {
                const canvas = document.createElement('canvas');
                const imageData = this.tensorToImageData(tensor, c, 6);
                canvas.width = imageData.width;
                canvas.height = imageData.height;
                const ctx = canvas.getContext('2d');
                ctx.putImageData(imageData, 0, 0);
                const wrapper = document.createElement('div');
                wrapper.style.textAlign = 'center';
                wrapper.style.fontSize = '10px';
                wrapper.style.color = '#666';
                const label = document.createElement('div');
                label.textContent = `Ch ${c}`;
                wrapper.appendChild(label);
                wrapper.appendChild(canvas);
                grid.appendChild(wrapper);
            }
            container.appendChild(grid);
        }
        else if (shape.length === 2 && shape[0] === 1) {
            // Flattened output: [1, N] - show as bar chart
            const formattedName = this.formatLayerName(activation.name);
            this.drawBarChart(tensor, container, `Layer ${activation.layerIdx}: ${formattedName}`);
        }
        else {
            // Single image
            const canvas = document.createElement('canvas');
            const imageData = this.tensorToImageData(tensor, undefined, 6);
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            const ctx = canvas.getContext('2d');
            ctx.putImageData(imageData, 0, 0);
            container.appendChild(canvas);
        }
    }
    /**
     * Draw bar chart for 1D/2D outputs (e.g., classification probabilities)
     */
    static drawBarChart(tensor, container, title) {
        const floatData = tensor.asFloat();
        const shape = tensor.shape;
        // Extract values
        let values;
        if (shape.length === 2 && shape[0] === 1) {
            // [1, N]
            values = Array.from(floatData);
        }
        else if (shape.length === 1) {
            // [N]
            values = Array.from(floatData);
        }
        else if (shape.length === 0) {
            // Scalar
            values = [floatData[0]];
        }
        else {
            throw new Error(`Cannot draw bar chart for shape: [${shape.join(', ')}]`);
        }
        const maxVal = Math.max(...values.map(Math.abs));
        const maxIdx = values.indexOf(Math.max(...values));
        const wrapper = document.createElement('div');
        wrapper.style.padding = '10px';
        wrapper.style.border = '1px solid #ccc';
        wrapper.style.borderRadius = '4px';
        if (title) {
            const header = document.createElement('div');
            header.style.marginBottom = '10px';
            header.style.fontWeight = 'bold';
            header.textContent = title;
            wrapper.appendChild(header);
        }
        const barsContainer = document.createElement('div');
        barsContainer.style.display = 'flex';
        barsContainer.style.alignItems = 'flex-end';
        barsContainer.style.gap = '5px';
        barsContainer.style.height = '200px';
        values.forEach((val, idx) => {
            const barWrapper = document.createElement('div');
            barWrapper.style.display = 'flex';
            barWrapper.style.flexDirection = 'column';
            barWrapper.style.alignItems = 'center';
            barWrapper.style.flex = '1';
            const bar = document.createElement('div');
            const height = maxVal > 0 ? (Math.abs(val) / maxVal) * 100 : 0;
            bar.style.width = '100%';
            bar.style.height = `${height}%`;
            bar.style.backgroundColor = idx === maxIdx ? '#4CAF50' : (val >= 0 ? '#2196F3' : '#f44336');
            bar.style.minHeight = '2px';
            bar.style.borderRadius = '2px 2px 0 0';
            bar.style.transition = 'background-color 0.2s';
            const label = document.createElement('div');
            label.style.fontSize = '10px';
            label.style.marginTop = '5px';
            label.style.color = '#666';
            label.textContent = idx.toString();
            barWrapper.appendChild(bar);
            barWrapper.appendChild(label);
            barsContainer.appendChild(barWrapper);
        });
        wrapper.appendChild(barsContainer);
        // Add value labels
        const valuesLabel = document.createElement('div');
        valuesLabel.style.marginTop = '10px';
        valuesLabel.style.fontSize = '12px';
        valuesLabel.style.fontFamily = 'monospace';
        valuesLabel.style.color = '#666';
        valuesLabel.textContent = `Values: [${values.map(v => v.toFixed(3)).join(', ')}]`;
        wrapper.appendChild(valuesLabel);
        const predictionLabel = document.createElement('div');
        predictionLabel.style.marginTop = '5px';
        predictionLabel.style.fontWeight = 'bold';
        predictionLabel.style.color = '#4CAF50';
        const maxValue = values[maxIdx];
        // Don't assume values are probabilities - just show the raw value
        predictionLabel.textContent = `Prediction: ${maxIdx} (value: ${maxValue.toFixed(3)})`;
        wrapper.appendChild(predictionLabel);
        container.appendChild(wrapper);
    }
    /**
     * Run forward pass and collect activations
     */
    runForward(input, onActivation) {
        const activations = [];
        const output = this.model.forward(input, (layerIdx, name, output) => {
            const activation = { layerIdx, name, output };
            activations.push(activation);
            if (onActivation) {
                onActivation(activation);
            }
        });
        return output;
    }
}
