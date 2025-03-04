// Global variables
let currentLogDir = '';
let currentEpoch = 0;
let lossData = [];
let networkData = {
    initial: null,
    epochs: {}
};
let performanceData = {
    expected: [],
    predictions: []
};

// DOM elements
const logDirectorySelect = document.getElementById('log-directory');
const epochSlider = document.getElementById('epoch-slider');
const epochDisplay = document.getElementById('epoch-display');
const epochSelector = document.getElementById('epoch-selector');
const tabButtons = document.querySelectorAll('.tab-button');
const tabPanes = document.querySelectorAll('.tab-pane');
const compareButton = document.getElementById('compare-button');
const comparisonView = document.getElementById('comparison-view');

// Charts
let lossChart = null;
let expectedPlot = null;
let predictionPlot = null;
let errorPlot = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Fetch available log directories
    fetchLogDirectories();
    
    // Set up event listeners
    setupEventListeners();
});

// Fetch available log directories
function fetchLogDirectories() {
    // In a real application, this would be an API call
    // For this demo, we'll use a simulated response
    const logDirs = [
        'training_logs_2d_run_1741095157',
        'training_logs_2d_run_1741097911',
        'training_logs_2d_run_1741098063'
    ];
    
    // Populate the select dropdown
    logDirs.forEach(dir => {
        const option = document.createElement('option');
        option.value = dir;
        option.textContent = dir;
        logDirectorySelect.appendChild(option);
    });
}

// Set up event listeners
function setupEventListeners() {
    // Log directory selection
    logDirectorySelect.addEventListener('change', (e) => {
        currentLogDir = e.target.value;
        if (currentLogDir) {
            loadLogData(currentLogDir);
            epochSelector.classList.remove('hidden');
        } else {
            epochSelector.classList.add('hidden');
        }
    });
    
    // Epoch slider
    epochSlider.addEventListener('input', (e) => {
        currentEpoch = parseInt(e.target.value);
        epochDisplay.textContent = `Epoch: ${currentEpoch}`;
        updateNetworkVisualization();
    });
    
    // Tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button and corresponding pane
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
            
            // Update visualizations when tab is switched
            if (tabId === 'loss-tab') {
                updateLossChart();
            } else if (tabId === 'weights-tab') {
                updateNetworkVisualization();
            } else if (tabId === 'performance-tab') {
                updatePerformanceVisualization();
            }
        });
    });
    
    // Compare button
    compareButton.addEventListener('click', () => {
        if (comparisonView.classList.contains('hidden')) {
            comparisonView.classList.remove('hidden');
            compareNetworkWeights();
        } else {
            comparisonView.classList.add('hidden');
        }
    });
}

// Load log data for the selected directory
function loadLogData(logDir) {
    // Reset current data
    lossData = [];
    networkData = {
        initial: null,
        epochs: {}
    };
    performanceData = {
        expected: [],
        predictions: []
    };
    
    // Load loss data
    fetch(`../${logDir}/loss.csv`)
        .then(response => response.text())
        .then(data => {
            parseLossData(data);
            updateLossChart();
        })
        .catch(error => console.error('Error loading loss data:', error));
    
    // Load network initial state
    fetch(`../${logDir}/network_initial.txt`)
        .then(response => response.text())
        .then(data => {
            networkData.initial = parseNetworkState(data);
        })
        .catch(error => console.error('Error loading initial network state:', error));
    
    // Load network states for epochs
    // In a real application, this would be more dynamic
    // For this demo, we'll load epochs 0, 100, 200, ..., 1000
    for (let epoch = 0; epoch <= 1000; epoch += 100) {
        fetch(`../${logDir}/epoch_${epoch.toString().padStart(4, '0')}/network_state.txt`)
            .then(response => response.text())
            .then(data => {
                networkData.epochs[epoch] = parseNetworkState(data);
                // Update visualization if this is the current epoch
                if (epoch === currentEpoch) {
                    updateNetworkVisualization();
                }
            })
            .catch(error => console.error(`Error loading network state for epoch ${epoch}:`, error));
        
        // Load predictions for the last epoch to show performance
        if (epoch === 1000) {
            fetch(`../${logDir}/epoch_${epoch.toString().padStart(4, '0')}/predictions.csv`)
                .then(response => response.text())
                .then(data => {
                    parsePerformanceData(data);
                    updatePerformanceVisualization();
                })
                .catch(error => console.error('Error loading performance data:', error));
        }
    }
    
    // Update the epoch slider max value based on available epochs
    epochSlider.max = 1000;
    epochSlider.value = 0;
    currentEpoch = 0;
    epochDisplay.textContent = `Epoch: ${currentEpoch}`;
}

// Parse loss data from CSV
function parseLossData(csvData) {
    const lines = csvData.trim().split('\n');
    // Skip header
    for (let i = 1; i < lines.length; i++) {
        const [epoch, loss] = lines[i].split(',');
        lossData.push({
            epoch: parseInt(epoch),
            loss: parseFloat(loss)
        });
    }
}

// Parse network state from text file
function parseNetworkState(textData) {
    const lines = textData.split('\n');
    const networkState = {
        numLayers: 0,
        layers: []
    };
    
    // Extract number of layers
    const layerMatch = lines[1].match(/Number of layers: (\d+)/);
    if (layerMatch) {
        networkState.numLayers = parseInt(layerMatch[1]);
    }
    
    // Extract layer information
    let currentLayer = null;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // New layer
        if (line.startsWith('Layer')) {
            const layerMatch = line.match(/Layer (\d+):/);
            if (layerMatch) {
                currentLayer = {
                    index: parseInt(layerMatch[1]),
                    size: 0,
                    weights: [],
                    alpha: []
                };
                networkState.layers.push(currentLayer);
            }
        }
        // Layer size
        else if (line.startsWith('Size:') && currentLayer) {
            const sizeMatch = line.match(/Size: (\d+)/);
            if (sizeMatch) {
                currentLayer.size = parseInt(sizeMatch[1]);
            }
        }
        // Neuron weights
        else if (line.startsWith('Neuron') && currentLayer) {
            const weightsMatch = line.match(/Neuron \d+: \[(.*)\]/);
            if (weightsMatch) {
                const weights = weightsMatch[1].split(',').map(w => parseFloat(w.trim()));
                currentLayer.weights.push(weights);
            }
        }
        // Alpha values
        else if (line.startsWith('[') && line.endsWith(']') && currentLayer) {
            const alphaMatch = line.match(/\[(.*)\]/);
            if (alphaMatch) {
                currentLayer.alpha = alphaMatch[1].split(',').map(a => parseFloat(a.trim()));
            }
        }
    }
    
    return networkState;
}

// Parse performance data from CSV
function parsePerformanceData(csvData) {
    const lines = csvData.trim().split('\n');
    // Skip header
    const header = lines[0].split(',');
    const numOutputs = header.length - 3; // x, y, expected, then outputs
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const x = parseFloat(values[0]);
        const y = parseFloat(values[1]);
        const expected = parseFloat(values[2]);
        const predictions = [];
        
        for (let j = 0; j < numOutputs; j++) {
            predictions.push(parseFloat(values[3 + j]));
        }
        
        performanceData.expected.push({ x, y, z: expected });
        performanceData.predictions.push({ x, y, outputs: predictions });
    }
}

// Update the loss chart
function updateLossChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (lossChart) {
        lossChart.destroy();
    }
    
    // Create new chart
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: lossData.map(d => d.epoch),
            datasets: [{
                label: 'Loss',
                data: lossData.map(d => d.loss),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Update the network visualization
function updateNetworkVisualization() {
    const container = document.getElementById('network-visualization');
    container.innerHTML = '';
    
    // Get network state for current epoch
    const networkState = networkData.epochs[currentEpoch];
    if (!networkState) {
        container.innerHTML = '<p>No data available for this epoch.</p>';
        return;
    }
    
    // Create SVG for network visualization
    const width = container.clientWidth;
    const height = container.clientHeight;
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Calculate positions
    const numLayers = networkState.numLayers;
    const layerWidth = width / (numLayers + 1);
    const maxNeurons = Math.max(...networkState.layers.map(l => l.size));
    const neuronHeight = height / (maxNeurons + 1);
    
    // Draw connections and neurons
    for (let i = 0; i < networkState.layers.length; i++) {
        const layer = networkState.layers[i];
        const layerX = (i + 1) * layerWidth;
        
        // Draw neurons
        for (let j = 0; j < layer.size; j++) {
            const neuronY = (j + 1) * neuronHeight;
            
            // Draw neuron
            svg.append('circle')
                .attr('cx', layerX)
                .attr('cy', neuronY)
                .attr('r', 10)
                .attr('fill', `hsl(${layer.alpha[j] * 360}, 70%, 50%)`)
                .attr('stroke', '#333')
                .attr('stroke-width', 1);
            
            // Draw connections to next layer if not the last layer
            if (i < networkState.layers.length - 1) {
                const nextLayer = networkState.layers[i + 1];
                const nextLayerX = (i + 2) * layerWidth;
                
                for (let k = 0; k < nextLayer.size; k++) {
                    const nextNeuronY = (k + 1) * neuronHeight;
                    const weight = layer.weights[j][k];
                    
                    // Draw connection
                    svg.append('line')
                        .attr('x1', layerX)
                        .attr('y1', neuronY)
                        .attr('x2', nextLayerX)
                        .attr('y2', nextNeuronY)
                        .attr('stroke', `hsl(${weight * 360}, 70%, 50%)`)
                        .attr('stroke-width', Math.abs(weight) * 3)
                        .attr('opacity', 0.6);
                }
            }
        }
    }
}

// Compare network weights between current epoch and next epoch
function compareNetworkWeights() {
    const container = document.getElementById('weight-changes');
    container.innerHTML = '';
    
    // Get network states
    const currentState = networkData.epochs[currentEpoch];
    const nextEpoch = currentEpoch + 100;
    const nextState = networkData.epochs[nextEpoch];
    
    if (!currentState || !nextState) {
        container.innerHTML = '<p>Cannot compare: data missing for one of the epochs.</p>';
        return;
    }
    
    // Create a table to show weight changes
    const table = document.createElement('table');
    table.className = 'weight-changes-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = `
        <th>Layer</th>
        <th>Neuron</th>
        <th>Weight</th>
        <th>Current Value</th>
        <th>Next Value</th>
        <th>Change</th>
    `;
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    
    // Add rows for each weight
    for (let i = 0; i < currentState.layers.length; i++) {
        const currentLayer = currentState.layers[i];
        const nextLayer = nextState.layers[i];
        
        for (let j = 0; j < currentLayer.weights.length; j++) {
            for (let k = 0; k < currentLayer.weights[j].length; k++) {
                const currentWeight = currentLayer.weights[j][k];
                const nextWeight = nextLayer.weights[j][k];
                const change = nextWeight - currentWeight;
                const percentChange = (change / Math.abs(currentWeight)) * 100;
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${i}</td>
                    <td>${j}</td>
                    <td>${k}</td>
                    <td>${currentWeight.toFixed(4)}</td>
                    <td>${nextWeight.toFixed(4)}</td>
                    <td class="${change > 0 ? 'positive' : change < 0 ? 'negative' : ''}">
                        ${change.toFixed(4)} (${percentChange.toFixed(2)}%)
                    </td>
                `;
                tbody.appendChild(row);
            }
        }
    }
    
    table.appendChild(tbody);
    container.appendChild(table);
    
    // Add some styling for the table
    const style = document.createElement('style');
    style.textContent = `
        .weight-changes-table {
            width: 100%;
            border-collapse: collapse;
        }
        .weight-changes-table th, .weight-changes-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }
        .weight-changes-table th {
            background-color: #f2f2f2;
            text-align: center;
        }
        .weight-changes-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
    `;
    document.head.appendChild(style);
}

// Update the performance visualization
function updatePerformanceVisualization() {
    // Create 3D surface plots for expected function, prediction, and error
    createExpectedPlot();
    createPredictionPlot();
    createErrorPlot();
}

// Create 3D surface plot for expected function
function createExpectedPlot() {
    const ctx = document.getElementById('expected-plot').getContext('2d');
    
    // Prepare data for the plot
    const xValues = [...new Set(performanceData.expected.map(d => d.x))].sort();
    const yValues = [...new Set(performanceData.expected.map(d => d.y))].sort();
    
    // Create a 2D grid of z values
    const zValues = [];
    for (let i = 0; i < yValues.length; i++) {
        const row = [];
        for (let j = 0; j < xValues.length; j++) {
            const point = performanceData.expected.find(d => d.x === xValues[j] && d.y === yValues[i]);
            row.push(point ? point.z : 0);
        }
        zValues.push(row);
    }
    
    // Destroy existing chart if it exists
    if (expectedPlot) {
        expectedPlot.destroy();
    }
    
    // Create new chart using Chart.js
    expectedPlot = new Chart(ctx, {
        type: 'surface',
        data: {
            labels: xValues.map(x => x.toFixed(2)),
            datasets: [{
                label: 'Expected Function',
                data: zValues,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'X'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y'
                    }
                }
            }
        }
    });
}

// Create 3D surface plot for prediction
function createPredictionPlot() {
    const ctx = document.getElementById('prediction-plot').getContext('2d');
    
    // Prepare data for the plot
    const xValues = [...new Set(performanceData.predictions.map(d => d.x))].sort();
    const yValues = [...new Set(performanceData.predictions.map(d => d.y))].sort();
    
    // Create a 2D grid of z values (using the 3rd output - index 2)
    const zValues = [];
    for (let i = 0; i < yValues.length; i++) {
        const row = [];
        for (let j = 0; j < xValues.length; j++) {
            const point = performanceData.predictions.find(d => d.x === xValues[j] && d.y === yValues[i]);
            row.push(point ? point.outputs[2] : 0); // Using the 3rd output (index 2)
        }
        zValues.push(row);
    }
    
    // Destroy existing chart if it exists
    if (predictionPlot) {
        predictionPlot.destroy();
    }
    
    // Create new chart using Chart.js
    predictionPlot = new Chart(ctx, {
        type: 'surface',
        data: {
            labels: xValues.map(x => x.toFixed(2)),
            datasets: [{
                label: 'Network Prediction',
                data: zValues,
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'X'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y'
                    }
                }
            }
        }
    });
}

// Create 3D surface plot for error
function createErrorPlot() {
    const ctx = document.getElementById('error-plot').getContext('2d');
    
    // Prepare data for the plot
    const xValues = [...new Set(performanceData.expected.map(d => d.x))].sort();
    const yValues = [...new Set(performanceData.expected.map(d => d.y))].sort();
    
    // Create a 2D grid of error values
    const errorValues = [];
    for (let i = 0; i < yValues.length; i++) {
        const row = [];
        for (let j = 0; j < xValues.length; j++) {
            const expected = performanceData.expected.find(d => d.x === xValues[j] && d.y === yValues[i]);
            const predicted = performanceData.predictions.find(d => d.x === xValues[j] && d.y === yValues[i]);
            
            if (expected && predicted) {
                const error = Math.abs(expected.z - predicted.outputs[2]); // Using the 3rd output (index 2)
                row.push(error);
            } else {
                row.push(0);
            }
        }
        errorValues.push(row);
    }
    
    // Destroy existing chart if it exists
    if (errorPlot) {
        errorPlot.destroy();
    }
    
    // Create new chart using Chart.js
    errorPlot = new Chart(ctx, {
        type: 'surface',
        data: {
            labels: xValues.map(x => x.toFixed(2)),
            datasets: [{
                label: 'Absolute Error',
                data: errorValues,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'X'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y'
                    }
                }
            }
        }
    });
} 