# Ring Neural Network

A neural network implementation using ring topology in Rust.


## Overview

Ring Neural Network is a novel neural network architecture that represents weights and inputs as positions on a circular ring, with similarity determined by their circular distance. Unlike traditional neural networks that use floating-point weights, this architecture offers unique properties for certain types of problems, particularly those with circular or periodic features.


## Features

- Fixed-point arithmetic for high precision
- Ring topology for weights and inputs
- Product-based activation function
- Multiple optimizer implementations (SGD, Adam)
- Loss functions (MSE, Cross Entropy)
- Data handling with batch processing
- Visualization utilities for network structure and training progress


## Usage

```rust
use ring_nn::{Fixed32, RingNetwork, optimizer, loss, visualization, data};
use ring_nn::optimizer::Optimizer;
use ring_nn::loss::Loss;

fn main() {
    // Create a network with ring size 256
    let mut network = RingNetwork::new(256);
    
    // Add layers
    network.add_layer(3, 5);  // 3 inputs, 5 hidden neurons
    network.add_layer(5, 1);  // 5 hidden neurons, 1 output
    
    // Create training data
    let data = vec![vec![64, 128, 192], vec![32, 96, 160]];
    let targets = vec![
        vec![Fixed32::from_float(0.8)],
        vec![Fixed32::from_float(0.2)]
    ];
    
    // Create data loader with batching
    let data_loader = data::DataLoader::new(
        data.clone(),
        targets.clone(),
        2,  // batch_size
        true // shuffle
    );
    
    // Create optimizer
    let mut optimizer = optimizer::Adam::new(0.01, 0.9, 0.999, 1e-8);
    
    // Train network (simplified)
    for epoch in 0..50 {
        // Training code...
        // optimizer.update(&mut network, &loss_gradient);
    }
    
    // Visualize results
    visualization::plot_loss(&losses);
    visualization::visualize_ring_weights(&network.layers[0].weights, 256);
    
    // Make predictions
    let input = vec![50, 100, 150];
    let prediction = network.forward(&input);
    println!("Prediction: {}", prediction[0].to_float());
}
```


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ring-nn = "0.1.0"
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
