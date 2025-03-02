use crate::{Fixed32, ring};
use rand::Rng;

/// A layer in the Ring Neural Network
pub struct RingLayer {
    /// Number of input neurons
    pub input_size: usize,
    /// Number of output neurons
    pub output_size: usize,
    /// Size of the ring
    pub ring_size: u32,
    /// Weights for each connection (output_size × input_size)
    pub weights: Vec<Vec<u32>>,
    /// Alpha scaling factors for each output neuron
    pub alpha: Vec<Fixed32>,
    /// Gradients for weights (stored as f32 for better precision)
    pub(crate) weight_gradients: Vec<Vec<f32>>,
    /// Gradients for alpha (stored as f32)
    pub(crate) alpha_gradients: Vec<f32>,
}

/// Store inputs and intermediate values needed for backpropagation
#[derive(Clone)]
pub struct ForwardCache {
    pub(crate) inputs: Vec<u32>,
    pub(crate) intermediate_products: Vec<Vec<Fixed32>>,
    pub(crate) final_products: Vec<Fixed32>,
}

impl Clone for RingLayer {
    fn clone(&self) -> Self {
        RingLayer {
            input_size: self.input_size,
            output_size: self.output_size,
            ring_size: self.ring_size,
            weights: self.weights.clone(),
            alpha: self.alpha.clone(),
            weight_gradients: vec![vec![0.0; self.input_size]; self.output_size],
            alpha_gradients: vec![0.0; self.output_size],
        }
    }
}

impl RingLayer {
    /// Create a new RingLayer with random initialization
    pub fn new(input_size: usize, output_size: usize, ring_size: u32) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize weights randomly on the ring
        let weights = (0..output_size)
            .map(|_| (0..input_size)
                .map(|_| rng.gen_range(0..ring_size))
                .collect())
            .collect();
        
        // Initialize alpha between -1 and 1 (stored as 0-1, but interpreted as -1 to 1)
        let alpha = (0..output_size)
            .map(|_| Fixed32::from_float(rng.gen_range(0.0..1.0)))
            .collect();
        
        // Initialize gradients to zero
        let weight_gradients = vec![vec![0.0; input_size]; output_size];
        let alpha_gradients = vec![0.0; output_size];
        
        RingLayer {
            input_size,
            output_size,
            ring_size,
            weights,
            alpha,
            weight_gradients,
            alpha_gradients,
        }
    }
    
    /// Forward pass implementing the Ring Neural Network formula
    pub fn forward(&self, input: &[u32]) -> Vec<Fixed32> {
        let mut output = vec![Fixed32::ZERO; self.output_size];
        
        // For each output neuron
        for i in 0..self.output_size {
            // Start with product = 1.0
            let mut product = Fixed32::ONE;
            
            // For each input connection
            for j in 0..self.input_size {
                let x_j = input[j] % self.ring_size;
                let w_ij = self.weights[i][j];
                
                // Calculate ring similarity factor
                let factor = ring::ring_similarity_factor(x_j, w_ij, self.ring_size);
                
                // Accumulate product
                product = product.mul(factor);
            }
            
            // Scale by alpha and n/2
            let ring_scale = Fixed32::from_float(self.ring_size as f32 / 2.0);
            let alpha_mapped = Fixed32::from_float(self.alpha[i].to_float() * 2.0 - 1.0); // Map from 0-1 to -1-1
            output[i] = alpha_mapped.mul(ring_scale).mul(product);
        }
        
        output
    }
    
    /// Forward pass with caching for backpropagation
    pub fn forward_with_cache(&self, input: &[u32]) -> (Vec<Fixed32>, ForwardCache) {
        let mut output = vec![Fixed32::ZERO; self.output_size];
        let mut intermediate_products = vec![vec![Fixed32::ONE; self.input_size]; self.output_size];
        let mut final_products = vec![Fixed32::ONE; self.output_size];
        
        // For each output neuron
        for i in 0..self.output_size {
            // Start with product = 1.0
            let mut product = Fixed32::ONE;
            
            // For each input connection
            for j in 0..self.input_size {
                let x_j = input[j] % self.ring_size;
                let w_ij = self.weights[i][j];
                
                // Calculate ring similarity factor
                let factor = ring::ring_similarity_factor(x_j, w_ij, self.ring_size);
                
                // Record intermediate product
                intermediate_products[i][j] = product;
                
                // Accumulate product
                product = product.mul(factor);
            }
            
            // Record final product
            final_products[i] = product;
            
            // Scale by alpha and n/2
            let ring_scale = Fixed32::from_float(self.ring_size as f32 / 2.0);
            let alpha_mapped = Fixed32::from_float(self.alpha[i].to_float() * 2.0 - 1.0); // Map from 0-1 to -1-1
            output[i] = alpha_mapped.mul(ring_scale).mul(product);
        }
        
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
        let mut grad_input = vec![0.0; self.input_size];
        
        // For each output neuron
        for i in 0..self.output_size {
            // Get output gradient for this neuron
            let grad_out_i = grad_output[i];
            
            // Skip if gradient is zero
            if grad_out_i == 0.0 {
                continue;
            }
            
            // Scale factor n/2
            let ring_scale = self.ring_size as f32 / 2.0;
            
            // Gradient for alpha (map from -1-1 to 0-1 scale)
            let alpha_mapped = self.alpha[i].to_float() * 2.0 - 1.0;
            self.alpha_gradients[i] += grad_out_i * ring_scale * 
                                      cache.final_products[i].to_float() * 2.0;
            
            // For each input connection
            for j in 0..self.input_size {
                let x_j = cache.inputs[j] % self.ring_size;
                let w_ij = self.weights[i][j];
                
                // Get intermediate product before this factor was applied
                let prev_product = cache.intermediate_products[i][j].to_float();
                
                // Compute gradient for the weight w_ij
                // We need to compute: ∂(factor)/∂w_ij * prev_product
                // where factor = (n-2*min_dist)/n
                
                // Calculate both distances to determine which one is minimum
                let dist1 = (x_j.wrapping_sub(w_ij)) % self.ring_size;
                let dist2 = (w_ij.wrapping_sub(x_j)) % self.ring_size;
                
                // Determine gradient direction based on which distance is smaller
                let grad_direction = if dist1 <= dist2 { 2.0 } else { -2.0 };
                
                // Scale by 1/n
                let grad_factor = grad_direction / self.ring_size as f32;
                
                // Calculate gradient for weight
                let weight_grad = grad_out_i * alpha_mapped * ring_scale * 
                                 prev_product * grad_factor * 
                                 cache.final_products[i].to_float() / 
                                 ring::ring_similarity_factor(x_j, w_ij, self.ring_size).to_float();
                
                self.weight_gradients[i][j] += weight_grad;
                
                // Calculate gradient for input
                // Similar calculation but with opposite sign on gradient direction
                let input_grad = grad_out_i * alpha_mapped * ring_scale * 
                                prev_product * (-grad_direction) / self.ring_size as f32 * 
                                cache.final_products[i].to_float() / 
                                ring::ring_similarity_factor(x_j, w_ij, self.ring_size).to_float();
                
                grad_input[j] += input_grad;
            }
        }
        
        grad_input
    }
    
    /// Apply gradients using a learning rate
    pub fn apply_gradients(&mut self, learning_rate: f32) {
        // Update weights (keeping them on the ring)
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                // Calculate weight update
                let update = (learning_rate * self.weight_gradients[i][j]) as i32;
                if update != 0 {
                    // Apply update with wrapping to stay on ring
                    if update > 0 {
                        self.weights[i][j] = (self.weights[i][j] + update as u32) % self.ring_size;
                    } else {
                        // Handle negative updates with wrapping
                        let abs_update = update.unsigned_abs() as u32;
                        self.weights[i][j] = (self.weights[i][j] + self.ring_size - 
                                            (abs_update % self.ring_size)) % self.ring_size;
                    }
                }
                
                // Reset gradient
                self.weight_gradients[i][j] = 0.0;
            }
            
            // Update alpha (keeping it in 0-1 range)
            let alpha_update = self.alpha[i].to_float() + learning_rate * self.alpha_gradients[i];
            self.alpha[i] = Fixed32::from_float(alpha_update.clamp(0.0, 1.0));
            
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
        let layer = RingLayer::new(2, 1, 10);
        let input = vec![3, 7];
        let output = layer.forward(&input);
        
        // Just check that the output is in the expected range
        assert!(output[0].to_float() >= -5.0);
        assert!(output[0].to_float() <= 5.0);
    }
} 