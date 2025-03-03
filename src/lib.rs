//! # Ring Neural Network
//!
//! A neural network implementation using ring topology in Rust.
//!
//! Ring Neural Network is a novel neural network architecture that represents weights and inputs
//! as positions on a circular ring, with similarity determined by their circular distance.

mod fixed;
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
