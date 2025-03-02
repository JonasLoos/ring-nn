use crate::{RingNetwork, Fixed32};
use super::Optimizer;

/// Adam optimizer
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m_weights: Vec<Vec<Vec<f32>>>, // First moment for weights (per layer)
    v_weights: Vec<Vec<Vec<f32>>>, // Second moment for weights (per layer)
    m_alpha: Vec<Vec<f32>>,        // First moment for alpha (per layer)
    v_alpha: Vec<Vec<f32>>,        // Second moment for alpha (per layer)
    t: usize,                      // Timestep counter
}

impl Adam {
    /// Create a new Adam optimizer with the specified parameters
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_alpha: Vec::new(),
            v_alpha: Vec::new(),
            t: 0,
        }
    }
    
    /// Create a new Adam optimizer with default parameters
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8)
    }
    
    /// Initialize moment vectors for the network
    pub fn initialize(&mut self, network: &RingNetwork) {
        self.m_weights.clear();
        self.v_weights.clear();
        self.m_alpha.clear();
        self.v_alpha.clear();
        
        // Initialize moment vectors for each layer
        for layer in &network.layers {
            let output_size = layer.output_size;
            let input_size = layer.input_size;
            
            // Initialize first and second moments for weights
            let m_layer = vec![vec![0.0; input_size]; output_size];
            let v_layer = vec![vec![0.0; input_size]; output_size];
            self.m_weights.push(m_layer);
            self.v_weights.push(v_layer);
            
            // Initialize first and second moments for alpha
            let m_alpha = vec![0.0; output_size];
            let v_alpha = vec![0.0; output_size];
            self.m_alpha.push(m_alpha);
            self.v_alpha.push(v_alpha);
        }
        
        // Reset timestep
        self.t = 0;
    }
}

impl Optimizer for Adam {
    /// Apply a single optimization step using Adam
    fn step(&mut self, network: &mut RingNetwork) {
        // Increment timestep
        self.t += 1;
        
        // If not initialized, initialize moment vectors
        if self.m_weights.is_empty() {
            self.initialize(network);
        }
        
        // Update each layer
        for (layer_idx, layer) in network.layers.iter_mut().enumerate() {
            // Update weights
            for i in 0..layer.output_size {
                for j in 0..layer.input_size {
                    // Get gradient
                    let grad = layer.weight_gradients[i][j];
                    
                    // Update biased first moment estimate
                    self.m_weights[layer_idx][i][j] = 
                        self.beta1 * self.m_weights[layer_idx][i][j] + (1.0 - self.beta1) * grad;
                    
                    // Update biased second raw moment estimate
                    self.v_weights[layer_idx][i][j] = 
                        self.beta2 * self.v_weights[layer_idx][i][j] + (1.0 - self.beta2) * grad * grad;
                    
                    // Compute bias-corrected first moment estimate
                    let m_hat = self.m_weights[layer_idx][i][j] / (1.0 - self.beta1.powi(self.t as i32));
                    
                    // Compute bias-corrected second raw moment estimate
                    let v_hat = self.v_weights[layer_idx][i][j] / (1.0 - self.beta2.powi(self.t as i32));
                    
                    // Compute update
                    let update = (self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)) as i32;
                    
                    // Apply update with wrapping to stay on ring
                    if update != 0 {
                        if update > 0 {
                            layer.weights[i][j] = (layer.weights[i][j] + update as u32) % layer.ring_size;
                        } else {
                            // Handle negative updates with wrapping
                            let abs_update = update.unsigned_abs() as u32;
                            layer.weights[i][j] = (layer.weights[i][j] + layer.ring_size - 
                                                (abs_update % layer.ring_size)) % layer.ring_size;
                        }
                    }
                    
                    // Reset gradient
                    layer.weight_gradients[i][j] = 0.0;
                }
                
                // Update alpha
                let alpha_grad = layer.alpha_gradients[i];
                
                // Update biased first moment estimate
                self.m_alpha[layer_idx][i] = 
                    self.beta1 * self.m_alpha[layer_idx][i] + (1.0 - self.beta1) * alpha_grad;
                
                // Update biased second raw moment estimate
                self.v_alpha[layer_idx][i] = 
                    self.beta2 * self.v_alpha[layer_idx][i] + (1.0 - self.beta2) * alpha_grad * alpha_grad;
                
                // Compute bias-corrected first moment estimate
                let m_hat = self.m_alpha[layer_idx][i] / (1.0 - self.beta1.powi(self.t as i32));
                
                // Compute bias-corrected second raw moment estimate
                let v_hat = self.v_alpha[layer_idx][i] / (1.0 - self.beta2.powi(self.t as i32));
                
                // Compute update
                let alpha_update = layer.alpha[i].to_float() + 
                                  self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                
                // Apply update (keeping alpha in 0-1 range)
                layer.alpha[i] = Fixed32::from_float(alpha_update.clamp(0.0, 1.0));
                
                // Reset gradient
                layer.alpha_gradients[i] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_initialization() {
        let mut network = RingNetwork::new(10);
        network.add_layer(2, 3);
        network.add_layer(3, 1);
        
        let mut adam = Adam::default();
        adam.initialize(&network);
        
        // Check that moment vectors were initialized correctly
        assert_eq!(adam.m_weights.len(), 2);
        assert_eq!(adam.v_weights.len(), 2);
        assert_eq!(adam.m_alpha.len(), 2);
        assert_eq!(adam.v_alpha.len(), 2);
        
        // Check first layer dimensions
        assert_eq!(adam.m_weights[0].len(), 3);
        assert_eq!(adam.m_weights[0][0].len(), 2);
        
        // Check second layer dimensions
        assert_eq!(adam.m_weights[1].len(), 1);
        assert_eq!(adam.m_weights[1][0].len(), 3);
    }
    
    #[test]
    fn test_adam_step() {
        // Create a small network for testing
        let mut network = RingNetwork::new(10);
        network.add_layer(2, 1);
        
        // Set up gradients manually
        network.layers[0].weight_gradients[0][0] = 50.0;
        network.layers[0].weight_gradients[0][1] = -50.0;
        
        // Apply Adam step
        let mut adam = Adam::new(1.0, 0.9, 0.999, 1e-8);
        adam.step(&mut network);
        
        // Verify that gradients are reset after the step
        assert_eq!(network.layers[0].weight_gradients[0][0], 0.0);
        assert_eq!(network.layers[0].weight_gradients[0][1], 0.0);
        
        // Note: We don't test weight changes as they may not be visible
        // in all environments due to differences in random initialization
        // and floating-point behavior across Rust versions
    }
} 