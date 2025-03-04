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
            let output_size = layer.size;
            let input_size = layer.size;
            
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
        // Initialize if not already done
        if self.m_weights.is_empty() {
            self.initialize(network);
        }
        
        // Increment timestep
        self.t += 1;
        
        // For each layer in the network
        for layer_idx in 0..network.layers.len() {
            let layer = &mut network.layers[layer_idx];
            
            // Update weights
            for i in 0..layer.size { // output
                for j in 0..layer.size {  // input
                    // Update first and second moments for weights
                    self.m_weights[layer_idx][i][j] = self.beta1 * self.m_weights[layer_idx][i][j] + 
                                                    (1.0 - self.beta1) * layer.weight_gradients[i][j];
                    
                    self.v_weights[layer_idx][i][j] = self.beta2 * self.v_weights[layer_idx][i][j] + 
                                                    (1.0 - self.beta2) * layer.weight_gradients[i][j].powi(2);
                    
                    // Bias correction
                    let m_hat = self.m_weights[layer_idx][i][j] / (1.0 - self.beta1.powi(self.t as i32));
                    let v_hat = self.v_weights[layer_idx][i][j] / (1.0 - self.beta2.powi(self.t as i32));
                    
                    // Calculate update
                    let update = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                    if update.abs() > 0.5 {
                        println!("Warning: Update value {} is too large for Fixed32", update);
                    }
                    let update_fixed = Fixed32::from_float(update.abs().clamp(0.0, 0.5)).unwrap();

                    // Apply update with Fixed32 methods instead of wrapping
                    if update > 0.0 {
                        layer.weights[i][j] -= update_fixed;
                    } else {
                        layer.weights[i][j] += update_fixed;
                    }
                    
                    // Reset gradient
                    layer.weight_gradients[i][j] = 0.0;
                }
                
                // Update alpha
                // Update first and second moments for alpha
                self.m_alpha[layer_idx][i] = self.beta1 * self.m_alpha[layer_idx][i] + 
                                           (1.0 - self.beta1) * layer.alpha_gradients[i];
                
                self.v_alpha[layer_idx][i] = self.beta2 * self.v_alpha[layer_idx][i] + 
                                           (1.0 - self.beta2) * layer.alpha_gradients[i].powi(2);
                
                // Bias correction
                let m_hat = self.m_alpha[layer_idx][i] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v_alpha[layer_idx][i] / (1.0 - self.beta2.powi(self.t as i32));
                
                // Calculate update
                let update = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                
                // Apply update to alpha (clamping to 0-1 range)
                layer.alpha[i] -= update;
                
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
        let mut network = RingNetwork::new();
        network.add_layer(2);
        network.add_layer(3);
        
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
        
        // Initialize Adam optimizer
        adam.initialize(&network);
        
        // Check that moments are initialized correctly
        assert_eq!(adam.m_weights.len(), network.layers.len());
        assert_eq!(adam.v_weights.len(), network.layers.len());
        assert_eq!(adam.m_alpha.len(), network.layers.len());
        assert_eq!(adam.v_alpha.len(), network.layers.len());
        
        // Check first layer dimensions
        assert_eq!(adam.m_weights[0].len(), network.layers[0].size);
        assert_eq!(adam.m_weights[0][0].len(), network.layers[0].size);
        assert_eq!(adam.v_weights[0].len(), network.layers[0].size);
        assert_eq!(adam.v_weights[0][0].len(), network.layers[0].size);
        assert_eq!(adam.m_alpha[0].len(), network.layers[0].size);
        assert_eq!(adam.v_alpha[0].len(), network.layers[0].size);
        
        // Check second layer dimensions
        assert_eq!(adam.m_weights[1].len(), network.layers[1].size);
        assert_eq!(adam.m_weights[1][0].len(), network.layers[1].size);
        assert_eq!(adam.v_weights[1].len(), network.layers[1].size);
        assert_eq!(adam.v_weights[1][0].len(), network.layers[1].size);
        assert_eq!(adam.m_alpha[1].len(), network.layers[1].size);
        assert_eq!(adam.v_alpha[1].len(), network.layers[1].size);
    }
    
    #[test]
    fn test_adam_momentum() {
        // Create a small network for testing
        let mut network = RingNetwork::new();
        network.add_layer(2);
        
        // Set up gradients manually
        network.layers[0].weight_gradients[0][0] = 0.1;
        network.layers[0].weight_gradients[0][1] = -0.1;
        
        // Create Adam optimizer with high learning rate for visible effect
        let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8);
        adam.initialize(&network);
        
        // Apply step
        adam.step(&mut network);
        
        // Verify that gradients are reset
        assert_eq!(network.layers[0].weight_gradients[0][0], 0.0);
        assert_eq!(network.layers[0].weight_gradients[0][1], 0.0);
        
        // Verify that moments were updated (non-zero)
        assert!(adam.m_weights[0][0][0] != 0.0);
        assert!(adam.m_weights[0][0][1] != 0.0);
        assert!(adam.v_weights[0][0][0] != 0.0);
        assert!(adam.v_weights[0][0][1] != 0.0);
    }
} 