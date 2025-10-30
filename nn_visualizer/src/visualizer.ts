import type { AnyTensor } from '../nn/dist/types.js';
import { RingTensor, RealTensor } from '../nn/dist/tensor.js';

export interface LayerActivation {
  layerIdx: number;
  name: string;
  output: AnyTensor;
}

export class Visualizer {
  constructor(private model: any) {}

  /**
   * Format layer name for display - handles anonymous functions and long names
   */
  static formatLayerName(name: string): string {
    return name.split('\n').map(line => line.trim()).filter(line => line.length > 0 && !line.startsWith('//')).join(' ');
  }

  /**
   * Convert HSV to RGB
   */
  static hsvToRgb(h: number, s: number, v: number): [number, number, number] {
    const c = v * s;
    const x = c * (1 - Math.abs((h * 6) % 2 - 1));
    const m = v - c;
    
    let r = 0, g = 0, b = 0;
    
    if (h * 6 < 1) {
      r = c; g = x; b = 0;
    } else if (h * 6 < 2) {
      r = x; g = c; b = 0;
    } else if (h * 6 < 3) {
      r = 0; g = c; b = x;
    } else if (h * 6 < 4) {
      r = 0; g = x; b = c;
    } else if (h * 6 < 5) {
      r = x; g = 0; b = c;
    } else {
      r = c; g = 0; b = x;
    }
    
    return [
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255)
    ];
  }

  /**
   * Convert tensor to ImageData for canvas rendering
   * For shape [1, H, W, C], extracts a single channel or shows mean
   * Uses cyclic color scheme for RingTensor, grayscale for RealTensor
   */
  static tensorToImageData(tensor: AnyTensor, channel?: number, scale: number = 1): ImageData {
    const floatData = tensor.asFloat();
    const shape = tensor.shape;
    const isRingTensor = tensor.dtype === 'int16';
    
    let H: number, W: number, C: number;
    
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
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    
    const minVal = Math.min(...floatData);
    const maxVal = Math.max(...floatData);
    const range = maxVal - minVal || 1;
    
    const pixelValues: number[] = [];
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
    
    // Fill ImageData - cyclic color for RingTensor, grayscale for RealTensor
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const val = pixelValues[h * W + w];
        const normalized = (val - minVal) / range;
        
        let r: number, g: number, b: number;
        
        if (isRingTensor) {
          // Cyclic color scheme: map normalized value to hue (0-1)
          // Use full saturation and value for vibrant colors
          const hue = normalized; // Value maps directly to hue position on color wheel
          [r, g, b] = this.hsvToRgb(hue, 1.0, 1.0);
        } else {
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
            imageData.data[idx] = r;     // R
            imageData.data[idx + 1] = g; // G
            imageData.data[idx + 2] = b;  // B
            imageData.data[idx + 3] = 255;   // A
          }
        }
      }
    }
    
    return imageData;
  }

  /**
   * Draw activation grids for multi-channel tensors
   */
  static drawActivationsGrid(activation: LayerActivation, container: HTMLElement): void {
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
        const ctx = canvas.getContext('2d')!;
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
    } else if (shape.length === 2 && shape[0] === 1) {
      // Flattened output: [1, N] - show as bar chart
      const formattedName = this.formatLayerName(activation.name);
      this.drawBarChart(tensor, container, `Layer ${activation.layerIdx}: ${formattedName}`);
    } else {
      // Single image
      const canvas = document.createElement('canvas');
      const imageData = this.tensorToImageData(tensor, undefined, 6);
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      container.appendChild(canvas);
    }
  }

  /**
   * Draw bar chart for 1D/2D outputs (e.g., classification probabilities)
   * Enhanced with better visuals, hover tooltips, statistics, and zero line
   */
  static drawBarChart(tensor: AnyTensor, container: HTMLElement, title?: string): void {
    const floatData = tensor.asFloat();
    const shape = tensor.shape;
    
    let values: number[];
    if (shape.length === 2 && shape[0] === 1 || shape.length === 1) {
      values = Array.from(floatData);
    } else if (shape.length === 0) {
      values = [floatData[0]];
    } else {
      throw new Error(`Cannot draw bar chart for shape: [${shape.join(', ')}]`);
    }
    
    // Calculate statistics
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const meanVal = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((acc, val) => acc + Math.pow(val - meanVal, 2), 0) / values.length;
    const stdVal = Math.sqrt(variance);
    const maxAbsVal = Math.max(...values.map(Math.abs));
    const maxIdx = values.indexOf(maxVal);
    
    // Check if we have negative values
    const hasNegative = minVal < 0;
    
    const wrapper = document.createElement('div');
    wrapper.style.padding = '15px';
    wrapper.style.border = '1px solid #ddd';
    wrapper.style.borderRadius = '8px';
    wrapper.style.backgroundColor = '#fafafa';
    wrapper.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';
    
    if (title) {
      const header = document.createElement('div');
      header.style.marginBottom = '15px';
      header.style.fontWeight = 'bold';
      header.style.fontSize = '16px';
      header.style.color = '#333';
      header.textContent = title;
      wrapper.appendChild(header);
    }
    
    // Statistics panel
    const statsPanel = document.createElement('div');
    statsPanel.style.display = 'grid';
    statsPanel.style.gridTemplateColumns = 'repeat(4, 1fr)';
    statsPanel.style.gap = '10px';
    statsPanel.style.marginBottom = '15px';
    statsPanel.style.padding = '10px';
    statsPanel.style.backgroundColor = '#f0f0f0';
    statsPanel.style.borderRadius = '4px';
    statsPanel.style.fontSize = '12px';
    
    const statItem = (label: string, value: number, color: string = '#666') => {
      const div = document.createElement('div');
      div.style.textAlign = 'center';
      const labelEl = document.createElement('div');
      labelEl.style.color = '#888';
      labelEl.style.fontSize = '10px';
      labelEl.textContent = label;
      const valueEl = document.createElement('div');
      valueEl.style.color = color;
      valueEl.style.fontWeight = 'bold';
      valueEl.style.fontFamily = 'monospace';
      valueEl.textContent = value.toFixed(3);
      div.appendChild(labelEl);
      div.appendChild(valueEl);
      return div;
    };
    
    statsPanel.appendChild(statItem('Min', minVal, '#f44336'));
    statsPanel.appendChild(statItem('Max', maxVal, '#4CAF50'));
    statsPanel.appendChild(statItem('Mean', meanVal, '#2196F3'));
    statsPanel.appendChild(statItem('Std', stdVal, '#FF9800'));
    wrapper.appendChild(statsPanel);
    
    // Chart container with zero line support
    const chartContainer = document.createElement('div');
    chartContainer.style.position = 'relative';
    chartContainer.style.paddingTop = hasNegative ? '20px' : '0';
    chartContainer.style.paddingBottom = '30px';
    
    // Zero line indicator
    if (hasNegative) {
      const zeroLine = document.createElement('div');
      zeroLine.style.position = 'absolute';
      zeroLine.style.left = '0';
      zeroLine.style.right = '0';
      zeroLine.style.height = '2px';
      zeroLine.style.backgroundColor = '#999';
      zeroLine.style.opacity = '0.5';
      zeroLine.style.zIndex = '1';
      const zeroLinePos = maxAbsVal > 0 ? (maxAbsVal / (maxAbsVal + Math.abs(minVal))) * 100 : 50;
      zeroLine.style.top = `${zeroLinePos}%`;
      chartContainer.appendChild(zeroLine);
      
      const zeroLabel = document.createElement('div');
      zeroLabel.style.position = 'absolute';
      zeroLabel.style.right = '5px';
      zeroLabel.style.top = `${zeroLinePos}%`;
      zeroLabel.style.transform = 'translateY(-50%)';
      zeroLabel.style.fontSize = '10px';
      zeroLabel.style.color = '#999';
      zeroLabel.textContent = '0';
      chartContainer.appendChild(zeroLabel);
    }
    
    const barsContainer = document.createElement('div');
    barsContainer.style.display = 'flex';
    barsContainer.style.alignItems = hasNegative ? 'center' : 'flex-end';
    barsContainer.style.gap = '4px';
    barsContainer.style.height = hasNegative ? '250px' : '220px';
    barsContainer.style.position = 'relative';
    barsContainer.style.zIndex = '2';
    
    // Tooltip element (fixed positioning for viewport-relative placement)
    const tooltip = document.createElement('div');
    tooltip.style.position = 'fixed';
    tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
    tooltip.style.color = 'white';
    tooltip.style.padding = '6px 10px';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '12px';
    tooltip.style.fontFamily = 'monospace';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.zIndex = '10000';
    tooltip.style.display = 'none';
    tooltip.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
    tooltip.style.whiteSpace = 'pre-line';
    // Append to wrapper so it gets cleaned up with the container
    wrapper.appendChild(tooltip);
    
    values.forEach((val, idx) => {
      const barWrapper = document.createElement('div');
      barWrapper.style.display = 'flex';
      barWrapper.style.flexDirection = 'column';
      barWrapper.style.alignItems = 'center';
      barWrapper.style.flex = '1';
      barWrapper.style.position = 'relative';
      barWrapper.style.justifyContent = hasNegative ? (val >= 0 ? 'flex-end' : 'flex-start') : 'flex-end';
      
      // Calculate bar height and position
      let heightPercent: number;
      let barTop: string = 'auto';
      
      if (hasNegative) {
        const totalRange = maxAbsVal + Math.abs(minVal);
        heightPercent = (Math.abs(val) / totalRange) * 100;
        // Position relative to zero line
        if (val >= 0) {
          barTop = `${(Math.abs(minVal) / totalRange) * 100}%`;
        } else {
          barTop = `${(maxAbsVal / totalRange) * 100}%`;
        }
      } else {
        heightPercent = maxAbsVal > 0 ? (Math.abs(val) / maxAbsVal) * 100 : 0;
      }
      
      const bar = document.createElement('div');
      bar.style.width = '100%';
      bar.style.height = `${heightPercent}%`;
      bar.style.position = hasNegative ? 'absolute' : 'relative';
      if (hasNegative) {
        bar.style.bottom = val >= 0 ? '50%' : 'auto';
        bar.style.top = val < 0 ? '50%' : 'auto';
      }
      
      // Enhanced color scheme with gradients
      let bgColor: string;
      if (idx === maxIdx) {
        bgColor = '#4CAF50'; // Green for maximum
      } else if (val >= 0) {
        // Positive values: blue gradient
        const intensity = val / maxAbsVal;
        const r = Math.round(33 + (33 - 33) * (1 - intensity));
        const g = Math.round(150 + (150 - 150) * (1 - intensity));
        const b = Math.round(243 + (243 - 243) * (1 - intensity));
        bgColor = `rgb(${r}, ${g}, ${b})`;
      } else {
        // Negative values: red gradient
        const intensity = Math.abs(val) / maxAbsVal;
        const r = Math.round(244 + (200 - 244) * (1 - intensity));
        const g = Math.round(67 + (67 - 67) * (1 - intensity));
        const b = Math.round(54 + (54 - 54) * (1 - intensity));
        bgColor = `rgb(${r}, ${g}, ${b})`;
      }
      
      bar.style.backgroundColor = bgColor;
      bar.style.minHeight = '2px';
      bar.style.borderRadius = hasNegative ? '2px' : '2px 2px 0 0';
      bar.style.transition = 'all 0.2s ease';
      bar.style.cursor = 'pointer';
      bar.style.boxShadow = idx === maxIdx ? '0 2px 4px rgba(0,0,0,0.2)' : '0 1px 2px rgba(0,0,0,0.1)';
      
      // Value label on top of bar
      const valueLabel = document.createElement('div');
      valueLabel.style.position = hasNegative ? 'absolute' : 'relative';
      valueLabel.style.fontSize = '9px';
      valueLabel.style.fontWeight = 'bold';
      valueLabel.style.color = idx === maxIdx ? '#2E7D32' : '#555';
      valueLabel.style.fontFamily = 'monospace';
      valueLabel.style.textAlign = 'center';
      valueLabel.style.marginBottom = '2px';
      valueLabel.style.whiteSpace = 'nowrap';
      valueLabel.textContent = val.toFixed(2);
      if (hasNegative) {
        valueLabel.style.top = val >= 0 ? '-15px' : 'calc(100% + 2px)';
        valueLabel.style.width = '100%';
      }
      
      // Hover effects
      bar.addEventListener('mouseenter', (e) => {
        bar.style.transform = 'scaleY(1.05)';
        bar.style.opacity = '0.9';
        tooltip.style.display = 'block';
        tooltip.textContent = `Index: ${idx}\nValue: ${val.toFixed(6)}\nPercent: ${((val / maxAbsVal) * 100).toFixed(2)}%`;
        const rect = bar.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        // Center tooltip above bar, ensuring it stays within viewport
        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
        let top = rect.top - tooltipRect.height - 10;
        // Keep tooltip within viewport bounds
        if (left < 10) left = 10;
        if (left + tooltipRect.width > window.innerWidth - 10) {
          left = window.innerWidth - tooltipRect.width - 10;
        }
        if (top < 10) {
          top = rect.bottom + 10; // Show below if no room above
        }
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
      });
      
      bar.addEventListener('mouseleave', () => {
        bar.style.transform = 'scaleY(1)';
        bar.style.opacity = '1';
        tooltip.style.display = 'none';
      });
      
      // Index label
      const label = document.createElement('div');
      label.style.fontSize = '10px';
      label.style.marginTop = '5px';
      label.style.color = '#666';
      label.style.fontWeight = idx === maxIdx ? 'bold' : 'normal';
      label.textContent = idx.toString();
      
      barWrapper.appendChild(valueLabel);
      barWrapper.appendChild(bar);
      barWrapper.appendChild(label);
      barsContainer.appendChild(barWrapper);
    });
    
    chartContainer.appendChild(barsContainer);
    wrapper.appendChild(chartContainer);
    
    // Summary info
    const summaryDiv = document.createElement('div');
    summaryDiv.style.marginTop = '15px';
    summaryDiv.style.padding = '10px';
    summaryDiv.style.backgroundColor = '#f9f9f9';
    summaryDiv.style.borderRadius = '4px';
    summaryDiv.style.fontSize = '12px';
    
    const predictionLabel = document.createElement('div');
    predictionLabel.style.fontWeight = 'bold';
    predictionLabel.style.color = '#4CAF50';
    predictionLabel.style.marginBottom = '5px';
    const maxValue = values[maxIdx];
    predictionLabel.textContent = `Prediction: Index ${maxIdx} (value: ${maxValue.toFixed(6)})`;
    summaryDiv.appendChild(predictionLabel);
    
    // Compact values display (truncated if too long)
    const valuesLabel = document.createElement('div');
    valuesLabel.style.fontFamily = 'monospace';
    valuesLabel.style.color = '#666';
    valuesLabel.style.fontSize = '11px';
    if (values.length <= 20) {
      valuesLabel.textContent = `Values: [${values.map(v => v.toFixed(3)).join(', ')}]`;
    } else {
      const firstFew = values.slice(0, 5).map(v => v.toFixed(3)).join(', ');
      const lastFew = values.slice(-5).map(v => v.toFixed(3)).join(', ');
      valuesLabel.textContent = `Values: [${firstFew}, ..., ${lastFew}] (${values.length} total)`;
    }
    summaryDiv.appendChild(valuesLabel);
    
    wrapper.appendChild(summaryDiv);
    container.appendChild(wrapper);
  }

  /**
   * Run forward pass and collect activations
   */
  runForward(input: RingTensor, onActivation?: (activation: LayerActivation) => void): AnyTensor {
    return this.model.forward(input, (layerIdx: number, name: string, output: AnyTensor) => {
      if (onActivation) {
        onActivation({ layerIdx, name, output });
      }
    });
  }
}

