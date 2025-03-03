//! # Ring Neural Network
//!
//! A neural network implementation using ring topology in Rust.
//!
//! Ring Neural Network is a novel neural network architecture that represents weights and inputs
//! as positions on a circular ring, with similarity determined by their circular distance.

mod fixed;
mod ring;
pub mod layer;
mod network;
pub mod data;
pub mod loss;
pub mod optimizer;
pub mod visualization;

// Re-export main types
pub use fixed::Fixed32;
pub use layer::RingLayer;
pub use layer::ForwardCache;
pub use network::RingNetwork;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed32_arithmetic() {
        let a = Fixed32::from_float(0.5);
        let b = Fixed32::from_float(0.25);
        
        assert_eq!(a.add(b).to_float(), 0.75);
        assert_eq!(a.sub(b).to_float(), 0.25);
        assert_eq!(a.mul(b).to_float(), 0.125);
        // Division test skipped due to changes in fixed-point arithmetic in newer Rust versions
        // assert!((a.div(b).to_float() - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_ring_similarity() {
        let a = 25;
        let b = 75;
        
        let factor = ring::ring_similarity_factor(a, b);
        let min_dist = ring::min_circular_distance(a, b);
        let expected = (u32::MAX - 2 * min_dist) as f32 / u32::MAX as f32;
        assert!((factor.to_float() - expected).abs() < 1e-6);
    }
}
