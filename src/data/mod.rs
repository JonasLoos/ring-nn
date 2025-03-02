//! Data handling utilities for Ring Neural Networks

mod loader;

pub use loader::DataLoader;
use crate::Fixed32;

/// Convert float data to ring representation
pub fn float_to_ring(data: &[Vec<f32>], ring_size: u32) -> Vec<Vec<u32>> {
    data.iter()
        .map(|sample| sample.iter()
            .map(|&x| (x.clamp(0.0, 1.0) * ring_size as f32) as u32 % ring_size)
            .collect())
        .collect()
}

/// Convert float targets to Fixed32
pub fn float_to_fixed(targets: &[Vec<f32>]) -> Vec<Vec<Fixed32>> {
    targets.iter()
        .map(|sample| sample.iter()
            .map(|&x| Fixed32::from_float(x))
            .collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_to_ring() {
        let data = vec![
            vec![0.1, 0.5, 0.9],
            vec![0.0, 0.3, 1.0],
        ];
        
        let ring_data = float_to_ring(&data, 10);
        
        // Expected: [[1, 5, 9], [0, 3, 0]]
        assert_eq!(ring_data[0][0], 1);
        assert_eq!(ring_data[0][1], 5);
        assert_eq!(ring_data[0][2], 9);
        assert_eq!(ring_data[1][0], 0);
        assert_eq!(ring_data[1][1], 3);
        assert_eq!(ring_data[1][2], 0); // 1.0 wraps to 0 on a ring of size 10
    }

    #[test]
    fn test_float_to_fixed() {
        let targets = vec![
            vec![0.1, 0.5, 0.9],
        ];
        
        let fixed_targets = float_to_fixed(&targets);
        
        assert!((fixed_targets[0][0].to_float() - 0.1).abs() < 1e-6);
        assert!((fixed_targets[0][1].to_float() - 0.5).abs() < 1e-6);
        assert!((fixed_targets[0][2].to_float() - 0.9).abs() < 1e-6);
    }
} 