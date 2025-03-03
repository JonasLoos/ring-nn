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
    pub fn add_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = RingLayer::new(input_size, output_size);
        self.layers.push(layer);
    }
    
    /// Get the number of layers in the network
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Forward pass through the network
    pub fn forward(&self, input: &[u32]) -> Vec<Fixed32> {
        let mut current = input.to_vec();
        let mut current_fixed: Vec<Fixed32> = Vec::new();
        
        // Pass through each layer
        for (i, layer) in self.layers.iter().enumerate() {
            // For all but the last layer, we need to convert Fixed32 back to u32
            if i > 0 {
                current = current_fixed.iter()
                    .map(|&x| (x.to_float() * u32::MAX as f32) as u32)
                    .collect();
            }
            
            current_fixed = layer.forward(&current);
        }
        
        current_fixed
    }
    
    /// Forward pass with caching for backpropagation
    pub fn forward_with_cache(&self, input: &[u32]) -> (Vec<Fixed32>, Vec<ForwardCache>) {
        let mut current = input.to_vec();
        let mut current_fixed: Vec<Fixed32> = Vec::new();
        let mut caches = Vec::with_capacity(self.layers.len());
        
        // Pass through each layer
        for (i, layer) in self.layers.iter().enumerate() {
            // For all but the last layer, we need to convert Fixed32 back to u32
            if i > 0 {
                current = current_fixed.iter()
                    .map(|&x| (x.to_float() * u32::MAX as f32) as u32)
                    .collect();
            }
            
            let (output, cache) = layer.forward_with_cache(&current);
            current_fixed = output;
            caches.push(cache);
        }
        
        (current_fixed, caches)
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

    #[test]
    fn test_network_creation() {
        let network = RingNetwork::new();
        assert_eq!(network.num_layers(), 0);
    }

    #[test]
    fn test_network_forward() {
        let mut network = RingNetwork::new();
        network.add_layer(3, 2);
        network.add_layer(2, 1);
        
        let input = vec![64, 128, 192];
        let output = network.forward(&input);
        
        assert_eq!(output.len(), 1);
    }
} 