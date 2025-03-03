use crate::Fixed32;

/// Calculate minimum circular distance on the ring
pub fn min_circular_distance(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b).min(b.wrapping_sub(a))
}

/// Calculate the factor (n-2*min_dist)/n as in the formula
pub fn ring_similarity_factor(a: u32, b: u32) -> Fixed32 {
    let min_dist = min_circular_distance(a, b);
    
    // Calculate (u32::MAX - 2*min_dist)/u32::MAX as a Fixed32
    // Use wrapping operations to avoid overflow
    let numerator = u32::MAX.wrapping_sub(min_dist.wrapping_mul(2));
    
    // Convert to fixed-point format
    Fixed32::from_float(numerator as f32 / u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_circular_distance() {
        // Test on the full u32 ring
        assert_eq!(min_circular_distance(0, 1), 1);
        assert_eq!(min_circular_distance(0, u32::MAX), 1);
        assert_eq!(min_circular_distance(0, u32::MAX / 2), u32::MAX / 2);
        assert_eq!(min_circular_distance(1, u32::MAX - 1), 3);
    }

    #[test]
    fn test_ring_similarity_factor() {
        // Test on the full u32 ring
        // For small distance, factor should be close to 1.0
        let factor = ring_similarity_factor(1, 2);
        assert!((factor.to_float() - (1.0 - 2.0 * 1.0 / u32::MAX as f32)).abs() < 1e-6);
        
        // For distance u32::MAX/2, factor should be close to 0.0
        let factor = ring_similarity_factor(0, u32::MAX / 2);
        assert!(factor.to_float() < 1e-6);
    }
} 