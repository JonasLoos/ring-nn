use crate::RingNetwork;
use super::Optimizer;

/// Basic Stochastic Gradient Descent optimizer
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    /// Create a new SGD optimizer with the specified learning rate
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    /// Apply a single optimization step using SGD
    fn step(&mut self, network: &mut RingNetwork) {
        network.apply_gradients(self.learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fixed32, RingLayer};

    #[test]
    fn test_sgd_step() {
        // Create a simple network
        let mut network = RingNetwork::new(10);
        network.add_layer(2, 1);
        
        // Set some gradients manually
        network.layers[0].weight_gradients[0][0] = 1.0;
        network.layers[0].weight_gradients[0][1] = -1.0;
        
        // Store original weights
        let original_w1 = network.layers[0].weights[0][0];
        let original_w2 = network.layers[0].weights[0][1];
        
        // Apply SGD step
        let mut sgd = SGD::new(0.1);
        sgd.step(&mut network);
        
        // Check that weights were updated correctly
        let new_w1 = network.layers[0].weights[0][0];
        let new_w2 = network.layers[0].weights[0][1];
        
        // w1 should increase by 0.1 (or wrap around)
        assert_eq!(new_w1, (original_w1 + 1) % 10);
        
        // w2 should decrease by 0.1 (or wrap around)
        assert_eq!(new_w2, (original_w2 + 10 - 1) % 10);
        
        // Gradients should be reset
        assert_eq!(network.layers[0].weight_gradients[0][0], 0.0);
        assert_eq!(network.layers[0].weight_gradients[0][1], 0.0);
    }
} 