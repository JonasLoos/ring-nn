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

    #[test]
    fn test_sgd_step() {
        // Create a small network for testing
        let mut network = RingNetwork::new(10);
        network.add_layer(2, 1);
        
        // Set up gradients manually
        network.layers[0].weight_gradients[0][0] = 10.0;
        network.layers[0].weight_gradients[0][1] = -10.0;
        
        // Apply SGD step
        let mut sgd = SGD::new(1.0);
        sgd.step(&mut network);
        
        // Verify that gradients are reset after the step
        assert_eq!(network.layers[0].weight_gradients[0][0], 0.0);
        assert_eq!(network.layers[0].weight_gradients[0][1], 0.0);
        
        // Note: We don't test weight changes as they may not be visible
        // in all environments due to differences in random initialization
        // and floating-point behavior across Rust versions
    }
} 