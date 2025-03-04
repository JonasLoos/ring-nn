//! Data handling utilities for Ring Neural Networks

mod loader;

pub use loader::DataLoader;
use crate::Fixed32;

/// Convert float data to ring representation
pub fn float_to_ring(data: &[Vec<f32>]) -> Vec<Vec<u32>> {
    data.iter()
        .map(|sample| sample.iter()
            .map(|&x| (x.clamp(0.0, 1.0) * u32::MAX as f32) as u32)
            .collect())
        .collect()
}

/// Convert float targets to Fixed32
pub fn float_to_fixed(targets: &[Vec<f32>]) -> Option<Vec<Vec<Fixed32>>> {
    targets.iter()
        .map(|sample| sample.iter()
            .map(|&x| Fixed32::from_float(x))
            .collect::<Option<Vec<_>>>())
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
        
        let ring_data = float_to_ring(&data);
        
        // Check that values are proportional to the input
        assert!(ring_data[0][0] < ring_data[0][1]);
        assert!(ring_data[0][1] < ring_data[0][2]);
        assert_eq!(ring_data[1][0], 0);
        assert!(ring_data[1][1] > 0);
        assert_eq!(ring_data[1][2], u32::MAX); // 1.0 maps to u32::MAX
        
        // Check approximate values
        let expected_0_1 = (0.1 * u32::MAX as f32) as u32;
        let expected_0_5 = (0.5 * u32::MAX as f32) as u32;
        let expected_0_9 = (0.9 * u32::MAX as f32) as u32;
        let expected_0_3 = (0.3 * u32::MAX as f32) as u32;
        
        assert_eq!(ring_data[0][0], expected_0_1);
        assert_eq!(ring_data[0][1], expected_0_5);
        assert_eq!(ring_data[0][2], expected_0_9);
        assert_eq!(ring_data[1][1], expected_0_3);
    }

    #[test]
    fn test_float_to_fixed() {
        let targets = vec![
            vec![0.1, 0.5, 0.9],
        ];
        
        let fixed_targets = float_to_fixed(&targets).unwrap();
        
        assert!((fixed_targets[0][0].to_float() - 0.1).abs() < 1e-6);
        assert!((fixed_targets[0][1].to_float() - 0.5).abs() < 1e-6);
        assert!((fixed_targets[0][2].to_float() - 0.9).abs() < 1e-6);
    }
} 