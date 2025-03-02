use crate::Fixed32;
use super::Loss;

/// Cross Entropy loss (for classification tasks)
pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    /// Calculate cross entropy loss for binary or multi-class classification
    /// For binary classification, each target should be a single value
    /// For multi-class, targets should be one-hot encoded
    fn forward(predictions: &[Fixed32], targets: &[Fixed32]) -> f32 {
        let n = predictions.len();
        assert_eq!(n, targets.len(), "Predictions and targets must have same length");
        
        let mut loss = 0.0;
        
        // Check if this is binary classification (single output)
        if n == 1 {
            // Binary cross entropy: -t*log(p) - (1-t)*log(1-p)
            let p = predictions[0].to_float().clamp(1e-7, 1.0 - 1e-7); // Avoid log(0)
            let t = targets[0].to_float();
            loss = -t * p.ln() - (1.0 - t) * (1.0 - p).ln();
        } else {
            // Multi-class cross entropy: -sum(t_i * log(p_i))
            for i in 0..n {
                let p = predictions[i].to_float().clamp(1e-7, 1.0); // Avoid log(0)
                let t = targets[i].to_float();
                if t > 0.0 {
                    loss -= t * p.ln();
                }
            }
        }
        
        loss
    }
    
    /// Calculate gradients for cross entropy loss
    fn backward(predictions: &[Fixed32], targets: &[Fixed32]) -> Vec<f32> {
        let n = predictions.len();
        assert_eq!(n, targets.len(), "Predictions and targets must have same length");
        
        let mut gradients = Vec::with_capacity(n);
        
        // Check if this is binary classification (single output)
        if n == 1 {
            // Gradient of binary cross entropy: -t/p + (1-t)/(1-p)
            let p = predictions[0].to_float().clamp(1e-7, 1.0 - 1e-7);
            let t = targets[0].to_float();
            gradients.push(-t / p + (1.0 - t) / (1.0 - p));
        } else {
            // Gradient of multi-class cross entropy: -t_i/p_i
            for i in 0..n {
                let p = predictions[i].to_float().clamp(1e-7, 1.0);
                let t = targets[i].to_float();
                gradients.push(-t / p);
            }
        }
        
        gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_cross_entropy() {
        let predictions = vec![Fixed32::from_float(0.7)];
        let targets = vec![Fixed32::from_float(1.0)];
        
        let loss = CrossEntropyLoss::forward(&predictions, &targets);
        let gradients = CrossEntropyLoss::backward(&predictions, &targets);
        
        // Expected loss: -1.0*ln(0.7) - 0.0*ln(0.3) ≈ 0.357
        assert!((loss - 0.357).abs() < 0.01);
        
        // Expected gradient: -1.0/0.7 + 0.0/0.3 ≈ -1.429
        assert!((gradients[0] + 1.429).abs() < 0.01);
    }

    #[test]
    fn test_multiclass_cross_entropy() {
        let predictions = vec![
            Fixed32::from_float(0.1), 
            Fixed32::from_float(0.7), 
            Fixed32::from_float(0.2)
        ];
        let targets = vec![
            Fixed32::from_float(0.0), 
            Fixed32::from_float(1.0), 
            Fixed32::from_float(0.0)
        ];
        
        let loss = CrossEntropyLoss::forward(&predictions, &targets);
        let gradients = CrossEntropyLoss::backward(&predictions, &targets);
        
        // Expected loss: -0.0*ln(0.1) - 1.0*ln(0.7) - 0.0*ln(0.2) ≈ 0.357
        assert!((loss - 0.357).abs() < 0.01);
        
        // Expected gradients: [-0.0/0.1, -1.0/0.7, -0.0/0.2] ≈ [0, -1.429, 0]
        assert!((gradients[0] - 0.0).abs() < 0.01);
        assert!((gradients[1] + 1.429).abs() < 0.01);
        assert!((gradients[2] - 0.0).abs() < 0.01);
    }
} 