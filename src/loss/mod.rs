//! Loss functions for training neural networks

mod mse;
mod cross_entropy;

pub use mse::MSELoss;
pub use cross_entropy::CrossEntropyLoss;

/// Common trait for loss functions
pub trait Loss {
    /// Calculate the loss between predictions and targets
    fn forward(predictions: &[crate::Fixed32], targets: &[crate::Fixed32]) -> f32;
    
    /// Calculate gradients for the loss
    fn backward(predictions: &[crate::Fixed32], targets: &[crate::Fixed32]) -> Vec<f32>;
} 