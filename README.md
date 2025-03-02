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
- Visualization utilities

## Usage

```rust
use ring_nn::{Fixed32, RingNetwork, optimizer, loss, visualization};

fn main() {
    // Create a network with ring size 256
    let mut network = RingNetwork::new(256);
    
    // Add layers
    network.add_layer(3, 5);  // 3 inputs, 5 hidden neurons
    network.add_layer(5, 1);  // 5 hidden neurons, 1 output
    
    // Train the network
    // ...
    
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