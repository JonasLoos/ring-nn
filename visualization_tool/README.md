# Ring Neural Network - Training Logs Viewer

This is a web application for viewing and inspecting the training logs of the Ring Neural Network. It allows you to:

1. Select training logs to inspect
2. View the loss curve during training
3. Visualize network weights at specific epochs
4. Compare weight changes between epochs
5. Investigate the final performance of the network

## Requirements

- Node.js (for running the server)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Getting Started

1. Navigate to the webapp directory:
   ```
   cd webapp
   ```

2. Start the server:
   ```
   node server.js
   ```

3. Open your browser and go to:
   ```
   http://localhost:3000
   ```

## Features

### Loss Curve

The Loss Curve tab shows how the loss value changes during training. This helps you understand if the network is learning effectively.

### Network Weights

The Network Weights tab visualizes the network structure and weights at a specific epoch. You can:

- Use the epoch slider to select different epochs
- Click "Compare with Next Epoch" to see how weights change between epochs

### Performance

The Performance tab shows how well the network approximates the target function:

- Expected Function: The true function being approximated (x² + y²)
- Network Prediction: The network's output
- Absolute Error: The difference between expected and predicted values

## Implementation Details

The webapp is built using:

- HTML, CSS, and JavaScript for the frontend
- Chart.js for plotting loss curves and performance visualizations
- D3.js for network weight visualizations
- Node.js for the simple server

The application reads training log files directly from the filesystem, including:

- `loss.csv`: Contains loss values for each epoch
- `network_initial.txt`: Initial network state
- `network_state.txt`: Network state at each logged epoch
- `predictions.csv`: Network predictions for evaluation

## Troubleshooting

If you encounter issues:

1. Make sure the server is running
2. Check that training log directories exist in the expected location
3. Verify that log files have the expected format
4. Check the browser console for any JavaScript errors 