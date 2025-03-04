# Ring Neural Network

Ring Neural Network is a novel neural network architecture that represents weights and inputs as positions on the circular ring naturally provided by u32 with overflow. Activations are computed using the product of similarities between weights and inputs.


Current State:

* [x] First implementation
* [x] Make sure the network learns at all
* [ ] Make sure the network can learn something meaningful (simple functions)
* [ ] Extend the architecture so more complex tasks are possible
* [ ] Test the network on more complex tasks (e.g. MNIST)
* [ ] Optimize/parallelize the implementation


## Usage

```rust
use ring_nn::{Fixed32, RingNetwork, visualization};
use ring_nn::loss::{MSELoss, Loss};
use ring_nn::optimizer::{Adam, Optimizer};

fn main() {
    // Create a network
    let mut network = RingNetwork::new();
    
    // Add layers (each with 3 neurons)
    network.add_layer(3);  // Input layer
    network.add_layer(3);  // Output layer
    
    // Visualize network structure
    visualization::visualize_network_structure(&network);
    
    // Create training data (using Fixed32 values)
    let data = vec![
        vec![Fixed32::from_float(0.1).unwrap(), Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.3).unwrap()],
        vec![Fixed32::from_float(0.4).unwrap(), Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.6).unwrap()]
    ];
    
    let targets = vec![
        vec![Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.3).unwrap(), Fixed32::from_float(0.4).unwrap()],
        vec![Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.6).unwrap(), Fixed32::from_float(0.7).unwrap()]
    ];
    
    // Create an optimizer
    let mut optimizer = Adam::new(0.005, 0.9, 0.999, 1e-8);
    
    // Train network
    let mut losses = Vec::new();
    
    for epoch in 0..50 {
        let mut epoch_loss = 0.0;
        
        for i in 0..data.len() {
            // Forward pass with caching
            let (predictions, caches) = network.forward_with_cache(&data[i]);
            
            // Calculate loss
            let loss = MSELoss::forward(&predictions, &targets[i]);
            epoch_loss += loss;
            
            // Calculate loss gradients
            let loss_grad = MSELoss::backward(&predictions, &targets[i]);
            
            // Backward pass
            network.backward(&loss_grad, &caches);
            
            // Apply gradients
            optimizer.step(&mut network);
        }
        
        // Record average loss
        epoch_loss /= data.len() as f32;
        losses.push(epoch_loss);
        
        println!("Epoch {}: Loss = {}", epoch, epoch_loss);
    }
    
    // Visualize results
    visualization::plot_loss(&losses);
    visualization::visualize_ring_weights(&network.layers[0].weights);
    
    // Make predictions with raw u32 inputs
    let input = vec![50, 100, 150];
    let prediction = network.forward(&input);
    
    println!("Prediction: {:?}", prediction.iter().map(|p| p.to_float()).collect::<Vec<f32>>());
}
```


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ring-nn = { git = "https://github.com/JonasLoos/ring-nn.git" }
```

Alternatively, you can clone the repository and build from source:

```bash
git clone https://github.com/JonasLoos/ring-nn.git
cd ring-nn
cargo build --release
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
