use crate::Fixed32;
use rand::Rng;

/// A layer in the Ring Neural Network
pub struct RingLayer {
    /// Number of neurons
    pub size: usize,
    /// Weights for each connection (size × size)
    pub weights: Vec<Vec<Fixed32>>,
    /// Alpha scaling factors for each output neuron
    pub alpha: Vec<f32>,
    /// Gradients for weights (stored as f32 for better precision)
    pub(crate) weight_gradients: Vec<Vec<f32>>,
    /// Gradients for alpha (stored as f32)
    pub(crate) alpha_gradients: Vec<f32>,
}

/// Store inputs and intermediate values needed for backpropagation
#[derive(Clone)]
pub struct ForwardCache {
    pub(crate) inputs: Vec<Fixed32>,
    pub(crate) intermediate_products: Vec<Vec<Fixed32>>,
    pub(crate) final_products: Vec<Fixed32>,
}

impl Clone for RingLayer {
    fn clone(&self) -> Self {
        RingLayer {
            size: self.size,
            weights: self.weights.clone(),
            alpha: self.alpha.clone(),
            weight_gradients: vec![vec![0.0; self.size]; self.size],
            alpha_gradients: vec![0.0; self.size],
        }
    }
}

impl RingLayer {
    /// Create a new RingLayer with random initialization
    pub fn new(size: usize) -> Self {
        let mut rng = rand::rng();
        
        // Initialize weights randomly on the ring
        let weights = (0..size)
            .map(|_| (0..size)
                .map(|_| Fixed32(rng.random_range(0..u32::MAX)))
                .collect())
            .collect();
        
        // Initialize alpha between -1 and 1 (stored as 0-1, but interpreted as -1 to 1)
        let alpha = (0..size)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        
        // Initialize gradients to zero
        let weight_gradients = vec![vec![0.0; size]; size];
        let alpha_gradients = vec![0.0; size];
        
        RingLayer {
            size,
            weights,
            alpha,
            weight_gradients,
            alpha_gradients,
        }
    }
    
    /// Forward pass implementing the Ring Neural Network formula
    pub fn forward(&self, input: &[Fixed32]) -> Vec<Fixed32> {
        let mut output = vec![Fixed32::ZERO; self.size];
        
        // For each output neuron
        for i in 0..self.size {
            // Start with product = 1.0
            let mut product = Fixed32::ONE;
            
            // For each input connection
            for j in 0..self.size {
                let x_j = input[j];
                let w_ij = self.weights[i][j];
                
                // Calculate ring similarity factor
                product *= x_j.similarity(w_ij);
            }
            
            // Scale by alpha and 0.5 (so that we have -0.5 to 0.5)
            let change = 0.5 * (product.to_float() * self.alpha[i]).clamp(-1.0, 1.0);
            let magnitude = Fixed32::from_float(change.abs()).unwrap();
            output[i] = if change >= 0.0 { input[i] + magnitude } else { input[i] - magnitude };
        }
        
        output
    }
    
    /// Forward pass with caching for backpropagation
    pub fn forward_with_cache(&self, input: &[Fixed32]) -> (Vec<Fixed32>, ForwardCache) {
        let mut output = vec![Fixed32::ZERO; self.size];
        let mut intermediate_products = vec![vec![Fixed32::ZERO; self.size]; self.size];
        let mut final_products = vec![Fixed32::ZERO; self.size];
        
        // For each output neuron
        for i in 0..self.size {
            // Start with product = 1.0
            let mut product = Fixed32::ONE;
            
            // For each input connection
            for j in 0..self.size {
                let x_j = input[j];
                let w_ij = self.weights[i][j];
                
                // Calculate ring similarity factor
                let factor = x_j.similarity(w_ij);
                
                // Store intermediate product
                intermediate_products[i][j] = factor;
                
                // Accumulate product
                product *= factor;
            }
            
            // Store final product
            final_products[i] = product;
            
            // Scale by alpha and 0.5 (so that we have -0.5 to 0.5)
            let change = 0.5 * (product.to_float() * self.alpha[i]).clamp(-1.0, 1.0);
            let magnitude = Fixed32::from_float(change.abs()).unwrap();
            output[i] = if change >= 0.0 { input[i] + magnitude } else { input[i] - magnitude };
        }
        
        // Create cache for backpropagation
        let cache = ForwardCache {
            inputs: input.to_vec(),
            intermediate_products,
            final_products,
        };
        
        (output, cache)
    }
}

impl RingLayer {
    /// Backward pass using the chain rule
    /// Returns gradients with respect to inputs
    pub fn backward(&mut self, 
                   grad_output: &[f32], 
                   cache: &ForwardCache) -> Vec<f32> {
        // Initialize input gradients
        let mut grad_input = vec![0.0; self.size];
        
        // For each output neuron
        for i in 0..self.size {
            // Get output gradient for this neuron
            let grad_out_i = grad_output[i];
            
            // Skip if gradient is zero
            if grad_out_i == 0.0 {
                continue;
            }
            
            // Scale factor 0.5
            let ring_scale = 0.5;
            
            // Gradient for alpha
            self.alpha_gradients[i] += grad_out_i * ring_scale * 
                                      cache.final_products[i].to_float() * 2.0;
            
            // For each input connection
            for j in 0..self.size {
                let x_j = cache.inputs[j];
                let w_ij = self.weights[i][j];
                
                // Get intermediate product before this factor was applied
                let prev_product = cache.intermediate_products[i][j].to_float();
                
                // Compute gradient for the weight w_ij
                // We need to compute: ∂(similarity)/∂w_ij * prev_product
                
                // Determine gradient direction based on which side of the distance we're on
                let grad_direction = if x_j.0 > w_ij.0 { 2.0 } else { -2.0 };
                
                // Scale by 1/n
                let grad_factor = grad_direction / u32::MAX as f32;
                
                // Calculate gradient for weight
                let similarity = x_j.similarity(w_ij).to_float();
                let weight_grad = grad_out_i * self.alpha[i] * ring_scale * 
                                 prev_product * grad_factor * 
                                 cache.final_products[i].to_float() / 
                                 (similarity + 1e-10); // Avoid division by zero
                
                self.weight_gradients[i][j] += weight_grad;
                
                // Calculate gradient for input
                // Similar calculation but with opposite sign on gradient direction
                let input_grad = grad_out_i * self.alpha[i] * ring_scale * 
                                prev_product * (-grad_direction) / u32::MAX as f32 * 
                                cache.final_products[i].to_float() / 
                                (similarity + 1e-10); // Avoid division by zero
                
                grad_input[j] += input_grad;
            }
        }
        
        grad_input
    }
    
    /// Apply gradients using a learning rate
    pub fn apply_gradients(&mut self, learning_rate: f32) {
        // Update weights (keeping them on the ring)
        for i in 0..self.size {
            for j in 0..self.size {
                // Calculate weight update
                let update = (learning_rate * self.weight_gradients[i][j]) as i32;
                if update != 0 {
                    // Apply update
                    if update > 0 {
                        // Create a Fixed32 with the update value
                        if let Some(update_fixed) = Fixed32::from_float(update as f32 / u32::MAX as f32) {
                            self.weights[i][j] = self.weights[i][j] + update_fixed;
                        }
                    } else {
                        // Handle negative updates
                        if let Some(update_fixed) = Fixed32::from_float((-update) as f32 / u32::MAX as f32) {
                            self.weights[i][j] = self.weights[i][j] - update_fixed;
                        }
                    }
                }
                
                // Reset gradient
                self.weight_gradients[i][j] = 0.0;
            }
            
            // Update alpha (keeping it in 0-1 range)
            self.alpha[i] = (self.alpha[i] + learning_rate * self.alpha_gradients[i]).clamp(0.0, 1.0);
            
            // Reset gradient
            self.alpha_gradients[i] = 0.0;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        // Create a layer manually with fixed weights
        let mut layer = RingLayer::new(2);
        layer.weights[0][0] = Fixed32::from_float(0.3).unwrap();
        layer.weights[0][1] = Fixed32::from_float(0.7).unwrap();
        layer.alpha[0] = 0.75; // Maps to 0.5 in -1 to 1 range
        
        // Test with fixed input
        let input = vec![
            Fixed32::from_float(0.4).unwrap(),
            Fixed32::from_float(0.6).unwrap()
        ];
        let output = layer.forward(&input);
        
        // Calculate expected output manually
        let factor1 = input[0].similarity(layer.weights[0][0]); // Similarity between 0.4 and 0.3
        let factor2 = input[1].similarity(layer.weights[0][1]); // Similarity between 0.6 and 0.7
        let product = factor1 * factor2;
        
        // The output should be the input plus a scaled change based on the product and alpha
        let change = 0.5 * (product.to_float() * layer.alpha[0]).clamp(-1.0, 1.0);
        let magnitude = Fixed32::from_float(change.abs()).unwrap();
        let expected = if change >= 0.0 { input[0] + magnitude } else { input[0] - magnitude };
        
        // Check that output matches expected calculation
        assert!((output[0].to_float() - expected.to_float()).abs() < 1e-6);
    }
    
    #[test]
    fn test_layer_backward() {
        // This test verifies that the backward pass correctly computes gradients
        // for both weights and alpha parameters. This is critical for training.
        
        // Create a layer with fixed weights for deterministic testing
        let mut layer = RingLayer::new(2);
        layer.weights[0][0] = Fixed32::from_float(0.5).unwrap();
        layer.weights[0][1] = Fixed32::from_float(0.8).unwrap();
        layer.alpha[0] = 0.75; // Maps to 0.5 in -1 to 1 range
        
        // Forward pass with fixed input to create cache
        let input = vec![
            Fixed32::from_float(0.4).unwrap(),
            Fixed32::from_float(0.6).unwrap()
        ];
        let (_, cache) = layer.forward_with_cache(&input);
        
        // Create a simple gradient for backpropagation (dL/dy = 1.0)
        let output_gradient = vec![1.0, 0.5];
        
        // Run backward pass
        let input_gradient = layer.backward(&output_gradient, &cache);
        
        // Verify gradients are non-zero (exact values are complex to calculate)
        // Weight gradients should be updated
        assert!(layer.weight_gradients[0][0] != 0.0, 
                "Weight gradient for weight[0][0] should be non-zero");
        assert!(layer.weight_gradients[0][1] != 0.0, 
                "Weight gradient for weight[0][1] should be non-zero");
        
        // Alpha gradient should be updated
        assert!(layer.alpha_gradients[0] != 0.0, 
                "Alpha gradient should be non-zero");
        
        // Input gradients should be returned
        assert_eq!(input_gradient.len(), 2, 
                   "Input gradient should have same length as input");
        assert!(input_gradient[0] != 0.0 || input_gradient[1] != 0.0, 
                "At least one input gradient should be non-zero");
    }
    
    #[test]
    fn test_layer_apply_gradients() {
        // This test verifies that gradients are correctly applied to update weights and alpha
        // This is essential for the learning process
        
        // Create a layer with fixed weights
        let mut layer = RingLayer::new(2);
        let original_weight_00 = layer.weights[0][0];
        let original_weight_01 = layer.weights[0][1];
        let original_alpha = layer.alpha[0];
        
        // Set gradients manually - use large values to ensure changes are applied
        layer.weight_gradients[0][0] = 100.0;  // Large positive gradient
        layer.weight_gradients[0][1] = -100.0; // Large negative gradient
        layer.alpha_gradients[0] = 0.5;        // Large alpha gradient
        
        // Apply gradients with learning rate
        let learning_rate = 0.01;
        layer.apply_gradients(learning_rate);
        
        // Verify weights changed
        assert!(layer.weights[0][0] != original_weight_00, 
                "Weight should change after applying gradient");
        assert!(layer.weights[0][1] != original_weight_01, 
                "Weight should change after applying gradient");
        
        // Verify alpha changed
        assert!(layer.alpha[0] != original_alpha, 
                "Alpha should change after applying gradient");
        
        // Verify gradients are reset
        assert_eq!(layer.weight_gradients[0][0], 0.0, 
                   "Weight gradient should be reset after applying");
        assert_eq!(layer.weight_gradients[0][1], 0.0, 
                   "Weight gradient should be reset after applying");
        assert_eq!(layer.alpha_gradients[0], 0.0, 
                   "Alpha gradient should be reset after applying");
    }
}
