use crate::Fixed32;

/// Calculate minimum circular distance on the ring
pub fn min_circular_distance(a: u32, b: u32, ring_size: u32) -> u32 {
    let a = a % ring_size;
    let b = b % ring_size;
    let dist1 = (a + ring_size - b) % ring_size;
    let dist2 = (b + ring_size - a) % ring_size;
    std::cmp::min(dist1, dist2)
}

/// Calculate the factor (n-2*min_dist)/n as in the formula
pub fn ring_similarity_factor(a: u32, b: u32, ring_size: u32) -> Fixed32 {
    let min_dist = min_circular_distance(a, b, ring_size);
    
    // Calculate (ring_size - 2*min_dist)/ring_size as a Fixed32
    // Need to be careful with the arithmetic to avoid overflow
    let numerator = ring_size.saturating_sub(2 * min_dist);
    
    // Convert to fixed-point format
    Fixed32::from_float(numerator as f32 / ring_size as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_circular_distance() {
        // Test on a ring of size 10
        assert_eq!(min_circular_distance(1, 9, 10), 2);
        assert_eq!(min_circular_distance(0, 5, 10), 5);
        assert_eq!(min_circular_distance(7, 2, 10), 5);
    }

    #[test]
    fn test_ring_similarity_factor() {
        // Test on a ring of size 10
        // For distance 2, factor should be (10-2*2)/10 = 0.6
        let factor = ring_similarity_factor(1, 9, 10);
        assert!((factor.to_float() - 0.6).abs() < 1e-6);
        
        // For distance 5, factor should be (10-2*5)/10 = 0.0
        let factor = ring_similarity_factor(0, 5, 10);
        assert!((factor.to_float() - 0.0).abs() < 1e-6);
    }
} 