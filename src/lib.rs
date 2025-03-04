//! # Ring Neural Network
//!
//! A neural network implementation using ring topology in Rust.
//!
//! Ring Neural Network is a novel neural network architecture that represents weights and inputs
//! as positions on a circular ring, with similarity determined by their circular distance.

mod fixed;
pub mod layer;
mod network;
pub mod data;
pub mod loss;
pub mod optimizer;
pub mod visualization;

// Re-export main types
pub use fixed::Fixed32;
pub use layer::RingLayer;
pub use layer::ForwardCache;
pub use network::RingNetwork;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use crate::{Fixed32, RingNetwork};
    use crate::loss::{MSELoss, Loss};
    use crate::optimizer::{Adam, Optimizer};

    #[test]
    fn test_readme_example() {
        // This test verifies that the example in the README works correctly
        
        // Create a network
        let mut network = RingNetwork::new();
        
        // Add layers (each with 3 neurons)
        network.add_layer(3);  // Input layer
        network.add_layer(3);  // Output layer
        
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
        
        // Reduce epochs for testing
        let epochs = 5;
        
        for _epoch in 0..epochs {
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
        }
        
        // Verify that loss decreased
        assert!(losses[0] > losses[losses.len() - 1], 
                "Training did not reduce loss: initial={}, final={}", 
                losses[0], losses[losses.len() - 1]);
        
        // Make predictions with raw u32 inputs
        let input = vec![50, 100, 150];
        let prediction = network.forward(&input);
        
        // Verify prediction shape
        assert_eq!(prediction.len(), 3, "Prediction should have 3 values");
        
        // Verify all prediction values are within valid range
        for p in &prediction {
            let p_float = p.to_float();
            assert!(p_float >= 0.0 && p_float <= 1.0, 
                    "Prediction value {} is outside valid range [0, 1]", p_float);
        }
    }
}
