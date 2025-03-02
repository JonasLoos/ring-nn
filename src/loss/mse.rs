use crate::Fixed32;
use super::Loss;

/// Mean Squared Error loss
pub struct MSELoss;

impl Loss for MSELoss {
    /// Calculate MSE loss between predictions and targets
    fn forward(predictions: &[Fixed32], targets: &[Fixed32]) -> f32 {
        let n = predictions.len();
        assert_eq!(n, targets.len(), "Predictions and targets must have same length");
        
        let mut sum_squared_error = 0.0;
        for i in 0..n {
            let diff = predictions[i].to_float() - targets[i].to_float();
            sum_squared_error += diff * diff;
        }
        
        sum_squared_error / n as f32
    }
    
    /// Calculate gradients for MSE loss
    fn backward(predictions: &[Fixed32], targets: &[Fixed32]) -> Vec<f32> {
        let n = predictions.len();
        assert_eq!(n, targets.len(), "Predictions and targets must have same length");
        
        let mut gradients = Vec::with_capacity(n);
        for i in 0..n {
            let diff = predictions[i].to_float() - targets[i].to_float();
            gradients.push(2.0 * diff / n as f32);
        }
        
        gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let predictions = vec![Fixed32::from_float(0.5), Fixed32::from_float(0.8)];
        let targets = vec![Fixed32::from_float(0.4), Fixed32::from_float(0.9)];
        
        let loss = MSELoss::forward(&predictions, &targets);
        let gradients = MSELoss::backward(&predictions, &targets);
        
        // Expected loss: ((0.5-0.4)^2 + (0.8-0.9)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
        assert!((loss - 0.01).abs() < 1e-6);
        
        // Expected gradients: [2*(0.5-0.4)/2, 2*(0.8-0.9)/2] = [0.1, -0.1]
        assert!((gradients[0] - 0.1).abs() < 1e-6);
        assert!((gradients[1] + 0.1).abs() < 1e-6);
    }
} 