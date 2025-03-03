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
            for i in 0..layer.output_size {
                for j in 0..layer.input_size {
                    // Update first and second moments for weights
                    self.m_weights[layer_idx][i][j] = self.beta1 * self.m_weights[layer_idx][i][j] + 
                                                    (1.0 - self.beta1) * layer.weight_gradients[i][j];
                    
                    self.v_weights[layer_idx][i][j] = self.beta2 * self.v_weights[layer_idx][i][j] + 
                                                    (1.0 - self.beta2) * layer.weight_gradients[i][j].powi(2);
                    
                    // Bias correction
                    let m_hat = self.m_weights[layer_idx][i][j] / (1.0 - self.beta1.powi(self.t as i32));
                    let v_hat = self.v_weights[layer_idx][i][j] / (1.0 - self.beta2.powi(self.t as i32));
                    
                    // Calculate update
                    let update = (self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)) as i32;
                    
                    // Apply update with wrapping to stay on ring
                    if update > 0 {
                        layer.weights[i][j] = layer.weights[i][j].wrapping_add(update as u32);
                    } else {
                        let abs_update = update.unsigned_abs() as u32;
                        layer.weights[i][j] = layer.weights[i][j].wrapping_sub(abs_update);
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
                let new_alpha = (layer.alpha[i].to_float() - update).clamp(0.0, 1.0);
                layer.alpha[i] = Fixed32::from_float(new_alpha);
                
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
        network.add_layer(2, 3);
        network.add_layer(3, 1);
        
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
        
        // Initialize the optimizer
        adam.initialize(&network);
        
        // Check that Adam is initialized with the correct dimensions
        assert_eq!(adam.m_weights.len(), network.layers.len());
        assert_eq!(adam.v_weights.len(), network.layers.len());
        
        // Check that m and v are initialized to zeros
        for layer_idx in 0..network.layers.len() {
            assert_eq!(adam.m_weights[layer_idx].len(), network.layers[layer_idx].weights.len());
            assert_eq!(adam.v_weights[layer_idx].len(), network.layers[layer_idx].weights.len());
            
            for neuron_idx in 0..network.layers[layer_idx].weights.len() {
                assert_eq!(adam.m_weights[layer_idx][neuron_idx].len(), network.layers[layer_idx].weights[neuron_idx].len());
                assert_eq!(adam.v_weights[layer_idx][neuron_idx].len(), network.layers[layer_idx].weights[neuron_idx].len());
                
                for weight_idx in 0..network.layers[layer_idx].weights[neuron_idx].len() {
                    assert_eq!(adam.m_weights[layer_idx][neuron_idx][weight_idx], 0.0);
                    assert_eq!(adam.v_weights[layer_idx][neuron_idx][weight_idx], 0.0);
                }
            }
        }
    }
    
    #[test]
    fn test_adam_step() {
        // Create a small network for testing
        let mut network = RingNetwork::new();
        network.add_layer(2, 1);
        
        // Set up gradients manually
        network.layers[0].weight_gradients[0][0] = 10.0;
        network.layers[0].weight_gradients[0][1] = -10.0;
        
        // Apply Adam step
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
        adam.step(&mut network);
        
        // Verify that gradients are reset after the step
        assert_eq!(network.layers[0].weight_gradients[0][0], 0.0);
        assert_eq!(network.layers[0].weight_gradients[0][1], 0.0);
    }
    
    #[test]
    fn test_adam_momentum() {
        // Create a small network for testing
        let mut network = RingNetwork::new();
        network.add_layer(2, 1);
        
        // Set up gradients manually
        network.layers[0].weight_gradients[0][0] = 10.0;
        network.layers[0].weight_gradients[0][1] = -10.0;
        
        // Apply Adam step
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
        
        // First step
        adam.step(&mut network);
        
        // Check that m and v are updated
        assert!(adam.m_weights[0][0][0] > 0.0);
        assert!(adam.m_weights[0][0][1] < 0.0);
        assert!(adam.v_weights[0][0][0] > 0.0);
        assert!(adam.v_weights[0][0][1] > 0.0);
        
        // Set new gradients
        network.layers[0].weight_gradients[0][0] = 5.0;
        network.layers[0].weight_gradients[0][1] = -5.0;
        
        // Second step
        adam.step(&mut network);
        
        // Check that t is incremented
        assert_eq!(adam.t, 2);
    }
} 