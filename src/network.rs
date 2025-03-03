use crate::{Fixed32, RingLayer, layer::ForwardCache};

/// A neural network using ring topology
pub struct RingNetwork {
    pub layers: Vec<RingLayer>,
}

impl Clone for RingNetwork {
    fn clone(&self) -> Self {
        RingNetwork {
            layers: self.layers.clone(),
        }
    }
}

impl RingNetwork {
    /// Create a new network
    pub fn new() -> Self {
        RingNetwork {
            layers: Vec::new(),
        }
    }
    
    /// Add a layer to the network
    pub fn add_layer(&mut self, size: usize) {
        let layer = RingLayer::new(size);
        self.layers.push(layer);
    }
    
    /// Get the number of layers in the network
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Forward pass through the network
    pub fn forward(&self, input: &[u32]) -> Vec<Fixed32> {
        // Convert u32 inputs to Fixed32
        let mut current_fixed: Vec<Fixed32> = input.iter()
            .map(|&x| Fixed32(x))
            .collect();
        
        // Pass through each layer
        for layer in &self.layers {
            current_fixed = layer.forward(&current_fixed);
        }
        
        current_fixed
    }
    
    /// Forward pass with caching for backpropagation
    pub fn forward_with_cache(&self, input: &[Fixed32]) -> (Vec<Fixed32>, Vec<ForwardCache>) {
        // Convert u32 inputs to Fixed32
        let mut current: Vec<Fixed32> = input.iter().copied().collect();
        
        let mut caches = Vec::with_capacity(self.layers.len());
        
        // Pass through each layer
        for layer in &self.layers {
            let (output, cache) = layer.forward_with_cache(&current);
            current = output;
            caches.push(cache);
        }
        
        (current, caches)
    }
    
    /// Backward pass through the network
    pub fn backward(&mut self, 
                    output_grad: &[f32], 
                    caches: &[ForwardCache]) -> Vec<f32> {
        assert_eq!(self.layers.len(), caches.len(), 
                  "Number of caches must match number of layers");
        
        let mut current_grad = output_grad.to_vec();
        
        // Backpropagate through layers in reverse order
        for i in (0..self.layers.len()).rev() {
            current_grad = self.layers[i].backward(&current_grad, &caches[i]);
        }
        
        current_grad
    }
    
    /// Apply accumulated gradients
    pub fn apply_gradients(&mut self, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.apply_gradients(learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;
    use crate::loss::Loss;

    #[test]
    fn test_network_creation() {
        // Test that a network can be created and layers added
        let network = RingNetwork::new();
        assert_eq!(network.num_layers(), 0);
    }

    #[test]
    fn test_network_forward() {
        // Test that forward propagation works through multiple layers
        let mut network = RingNetwork::new();
        network.add_layer(3);
        network.add_layer(3);
        
        let input = vec![64, 128, 192];
        let output = network.forward(&input);
        
        assert_eq!(output.len(), 3);
    }
    
    #[test]
    fn test_network_end_to_end_training() {
        use crate::loss::MSELoss;
        use crate::optimizer::SGD;
        
        // This test verifies that the network training process executes without errors
        // It demonstrates the full training loop with forward pass, backward pass,
        // and gradient application
        
        // Create a simple network
        let mut network = RingNetwork::new();
        network.add_layer(2);  // Input layer -> Hidden layer
        network.add_layer(2);  // Hidden layer -> Output layer
        
        // Create a simple dataset
        let data = vec![
            vec![Fixed32::from_float(0.1).unwrap(), Fixed32::from_float(0.2).unwrap()],
            vec![Fixed32::from_float(0.4).unwrap(), Fixed32::from_float(0.5).unwrap()],
            vec![Fixed32::from_float(0.7).unwrap(), Fixed32::from_float(0.8).unwrap()]
        ];
        
        // Create targets with proper Fixed32 values
        let targets = vec![
            vec![Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.3).unwrap()],
            vec![Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.6).unwrap()],
            vec![Fixed32::from_float(0.8).unwrap(), Fixed32::from_float(0.9).unwrap()]
        ];
        
        // Create optimizer
        let mut optimizer = SGD::new(0.01);
        
        // Train for a few epochs - just testing that the process completes without errors
        let epochs = 5;
        
        for _ in 0..epochs {
            // Train on each example
            for i in 0..data.len() {
                // Forward pass with caching
                let (predictions, caches) = network.forward_with_cache(&data[i]);
                
                // Calculate loss gradients
                let loss_grad = MSELoss::backward(&predictions, &targets[i]);
                
                // Backward pass
                network.backward(&loss_grad, &caches);
                
                // Apply gradients
                optimizer.step(&mut network);
            }
        }
        
        // If we got here without errors, the test passes
        assert!(true);
    }
}
