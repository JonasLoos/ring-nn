//! Optimizers for training neural networks

mod sgd;
mod adam;

pub use sgd::SGD;
pub use adam::Adam;

use crate::RingNetwork;

/// Common trait for optimizers
pub trait Optimizer {
    /// Apply a single optimization step
    fn step(&mut self, network: &mut RingNetwork);
} 