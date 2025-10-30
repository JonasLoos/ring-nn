export class Visualizer {
    constructor(model) {
        this.model = model;
    }
    /**
     * Convert tensor to ImageData for canvas rendering
     * For shape [1, H, W, C], extracts a single channel or shows mean
     */
    static tensorToImageData(tensor, channel, scale = 1) {
        const floatData = tensor.asFloat();
        const shape = tensor.shape;
        let H, W, C;

        if (shape.length === 4 && shape[0] === 1) {
            [H, W, C] = [shape[1], shape[2], shape[3]];
        } else if (shape.length === 2 && shape[0] === 1) {
            const side = Math.ceil(Math.sqrt(shape[1]));
            H = W = side;
            C = 1;
        } else if (shape.length === 3 && shape[0] === 1) {
            [H, W, C] = [shape[1], shape[2], 1];
        } else {
            throw new Error(`Unsupported tensor shape for visualization: ${shape.join(',')}`);
        }

        const canvas = document.createElement('canvas');
        canvas.width = W * scale;
        canvas.height = H * scale;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);

        const minVal = Math.min(...floatData);
        const maxVal = Math.max(...floatData);
        const range = maxVal - minVal || 1;
        const pixelValues = [];
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                if (C === 1) {
                    pixelValues.push(floatData[h * W + w]);
                } else if (channel !== undefined) {
                    pixelValues.push(floatData[(h * W + w) * C + channel]);
                } else {
                    let sum = 0;
                    for (let c = 0; c < C; c++) {
                        sum += floatData[(h * W + w) * C + c];
                    }
                    pixelValues.push(sum / C);
                }
            }
        }
        // Fill ImageData (grayscale -> RGBA)
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const val = pixelValues[h * W + w];
                const normalized = (val - minVal) / range;
                const gray = Math.max(0, Math.min(255, Math.round(normalized * 255)));
                // Scale up pixels
                for (let sh = 0; sh < scale; sh++) {
                    for (let sw = 0; sw < scale; sw++) {
                        const x = w * scale + sw;
                        const y = h * scale + sh;
                        const idx = (y * canvas.width + x) * 4;
                        imageData.data[idx] = gray; // R
                        imageData.data[idx + 1] = gray; // G
                        imageData.data[idx + 2] = gray; // B
                        imageData.data[idx + 3] = 255; // A
                    }
                }
            }
        }
        return imageData;
    }
    /**
     * Draw activation grids for multi-channel tensors
     */
    static drawActivationsGrid(activation, container) {
        const tensor = activation.output;
        const shape = tensor.shape;
        container.innerHTML = '';

        const header = document.createElement('div');
        header.style.marginBottom = '10px';
        header.style.fontWeight = 'bold';
        header.textContent = `Layer ${activation.layerIdx}: ${activation.name} (shape: [${shape.join(', ')}])`;
        container.appendChild(header);

        if (shape.length === 4 && shape[0] === 1) {
            const [H, W, C] = [shape[1], shape[2], shape[3]];
            const grid = document.createElement('div');
            Object.assign(grid.style, {
                display: 'flex',
                flexWrap: 'wrap',
                gap: '5px',
                padding: '10px',
                border: '1px solid #ccc',
                borderRadius: '4px'
            });

            const FIXED_CANVAS_SIZE = 200;
            for (let c = 0; c < C; c++) {
                const canvas = document.createElement('canvas');
                const imageData = this.tensorToImageData(tensor, c, 1);

                // Create temporary canvas for the native resolution image
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = imageData.width;
                tempCanvas.height = imageData.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.putImageData(imageData, 0, 0);

                // Set fixed size for display canvas
                canvas.width = FIXED_CANVAS_SIZE;
                canvas.height = FIXED_CANVAS_SIZE;
                const ctx = canvas.getContext('2d');
                ctx.imageSmoothingEnabled = true;
                ctx.drawImage(tempCanvas, 0, 0, FIXED_CANVAS_SIZE, FIXED_CANVAS_SIZE);

                const wrapper = document.createElement('div');
                Object.assign(wrapper.style, {
                    textAlign: 'center',
                    fontSize: '10px',
                    color: '#666'
                });

                const label = document.createElement('div');
                label.textContent = `Ch ${c}`;
                wrapper.appendChild(label);
                wrapper.appendChild(canvas);
                grid.appendChild(wrapper);
            }
            container.appendChild(grid);
        } else if (shape.length === 2 && shape[0] === 1) {
            this.drawBarChart(tensor, container, `Layer ${activation.layerIdx}: ${activation.name}`);
        } else {
            const FIXED_CANVAS_SIZE = 200;
            const canvas = document.createElement('canvas');
            const imageData = this.tensorToImageData(tensor, undefined, 1);

            // Create temporary canvas for the native resolution image
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = imageData.width;
            tempCanvas.height = imageData.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);

            // Set fixed size for display canvas
            canvas.width = FIXED_CANVAS_SIZE;
            canvas.height = FIXED_CANVAS_SIZE;
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(tempCanvas, 0, 0, FIXED_CANVAS_SIZE, FIXED_CANVAS_SIZE);
            container.appendChild(canvas);
        }
    }
    /**
     * Draw bar chart for 1D/2D outputs (e.g., classification probabilities)
     */
    static drawBarChart(tensor, container, title) {
        const floatData = tensor.asFloat();
        const shape = tensor.shape;

        let values;
        if (shape.length === 2 && shape[0] === 1 || shape.length === 1) {
            values = Array.from(floatData);
        } else if (shape.length === 0) {
            values = [floatData[0]];
        } else {
            throw new Error(`Cannot draw bar chart for shape: [${shape.join(', ')}]`);
        }
        const maxVal = Math.max(...values.map(Math.abs));
        const maxIdx = values.indexOf(Math.max(...values));

        const wrapper = document.createElement('div');
        Object.assign(wrapper.style, {
            padding: '10px',
            border: '1px solid #ccc',
            borderRadius: '4px'
        });

        if (title) {
            const header = document.createElement('div');
            header.style.marginBottom = '10px';
            header.style.fontWeight = 'bold';
            header.textContent = title;
            wrapper.appendChild(header);
        }

        const barsContainer = document.createElement('div');
        Object.assign(barsContainer.style, {
            display: 'flex',
            alignItems: 'flex-end',
            gap: '5px',
            height: '200px'
        });
        values.forEach((val, idx) => {
            const barWrapper = document.createElement('div');
            Object.assign(barWrapper.style, {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                flex: '1'
            });

            const bar = document.createElement('div');
            const height = maxVal > 0 ? (Math.abs(val) / maxVal) * 100 : 0;
            Object.assign(bar.style, {
                width: '100%',
                height: `${height}%`,
                backgroundColor: idx === maxIdx ? '#4CAF50' : (val >= 0 ? '#2196F3' : '#f44336'),
                minHeight: '2px',
                borderRadius: '2px 2px 0 0',
                transition: 'background-color 0.2s'
            });

            const label = document.createElement('div');
            Object.assign(label.style, {
                fontSize: '10px',
                marginTop: '5px',
                color: '#666'
            });
            label.textContent = idx.toString();

            barWrapper.appendChild(bar);
            barWrapper.appendChild(label);
            barsContainer.appendChild(barWrapper);
        });
        wrapper.appendChild(barsContainer);

        const valuesLabel = document.createElement('div');
        Object.assign(valuesLabel.style, {
            marginTop: '10px',
            fontSize: '12px',
            fontFamily: 'monospace',
            color: '#666'
        });
        valuesLabel.textContent = `Values: [${values.map(v => v.toFixed(3)).join(', ')}]`;
        wrapper.appendChild(valuesLabel);

        const predictionLabel = document.createElement('div');
        Object.assign(predictionLabel.style, {
            marginTop: '5px',
            fontWeight: 'bold',
            color: '#4CAF50'
        });
        predictionLabel.textContent = `Prediction: ${maxIdx} (${(values[maxIdx] * 100).toFixed(1)}%)`;
        wrapper.appendChild(predictionLabel);
        container.appendChild(wrapper);
    }
    /**
     * Run forward pass and collect activations
     */
    runForward(input, onActivation) {
        return this.model.forward(input, (layerIdx, name, output) => {
            if (onActivation) {
                onActivation({ layerIdx, name, output });
            }
        });
    }
}
