use crate::{Fixed32, RingLayer, layer::ForwardCache};

/// A neural network using ring topology
pub struct RingNetwork {
    pub layers: Vec<RingLayer>,
    pub(crate) ring_size: u32,
}

impl Clone for RingNetwork {
    fn clone(&self) -> Self {
        RingNetwork {
            layers: self.layers.clone(),
            ring_size: self.ring_size,
        }
    }
}

impl RingNetwork {
    /// Create a new network with the specified ring size
    pub fn new(ring_size: u32) -> Self {
        RingNetwork {
            layers: Vec::new(),
            ring_size,
        }
    }
    
    /// Add a layer to the network
    pub fn add_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = RingLayer::new(input_size, output_size, self.ring_size);
        self.layers.push(layer);
    }
    
    /// Get the number of layers in the network
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Get the ring size used by the network
    pub fn ring_size(&self) -> u32 {
        self.ring_size
    }
    
    /// Forward pass through the network
    pub fn forward(&self, input: &[u32]) -> Vec<Fixed32> {
        // Ensure input uses the correct ring size
        let input_on_ring: Vec<u32> = input.iter()
            .map(|&x| x % self.ring_size)
            .collect();
        
        let mut current = input_on_ring;
        let mut current_fixed: Vec<Fixed32> = Vec::new();
        
        // Pass through each layer
        for (i, layer) in self.layers.iter().enumerate() {
            // For all but the last layer, we need to convert Fixed32 back to u32
            if i > 0 {
                current = current_fixed.iter()
                    .map(|&x| (x.to_float() * self.ring_size as f32) as u32 % self.ring_size)
                    .collect();
            }
            
            current_fixed = layer.forward(&current);
        }
        
        current_fixed
    }
    
    /// Forward pass with caching for backpropagation
    pub fn forward_with_cache(&self, input: &[u32]) -> (Vec<Fixed32>, Vec<ForwardCache>) {
        // Ensure input uses the correct ring size
        let input_on_ring: Vec<u32> = input.iter()
            .map(|&x| x % self.ring_size)
            .collect();
        
        let mut current = input_on_ring;
        let mut current_fixed: Vec<Fixed32> = Vec::new();
        let mut caches = Vec::with_capacity(self.layers.len());
        
        // Pass through each layer
        for (i, layer) in self.layers.iter().enumerate() {
            // For all but the last layer, we need to convert Fixed32 back to u32
            if i > 0 {
                current = current_fixed.iter()
                    .map(|&x| (x.to_float() * self.ring_size as f32) as u32 % self.ring_size)
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
        let network = RingNetwork::new(256);
        assert_eq!(network.num_layers(), 0);
        assert_eq!(network.ring_size(), 256);
    }

    #[test]
    fn test_network_forward() {
        let mut network = RingNetwork::new(256);
        network.add_layer(3, 2);
        network.add_layer(2, 1);
        
        let input = vec![64, 128, 192];
        let output = network.forward(&input);
        
        assert_eq!(output.len(), 1);
    }
} 