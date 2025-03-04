/// Fixed-point number using all 32 bits for fraction (0.32 format)
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Fixed32(pub u32);

impl Fixed32 {
    /// Maximum value (0.99999...)
    pub const MAX: Self = Fixed32(u32::MAX);
    
    /// Zero value
    pub const ZERO: Self = Fixed32(0);
    
    /// One value (not representable perfectly)
    pub const ONE: Self = Fixed32(u32::MAX);
    
    /// Create from float
    pub fn from_float(value: f32) -> Option<Self> {
        if value < 0.0 || value > 1.0 {
            return None;
        }
        Some(Fixed32((value * u32::MAX as f32) as u32))
    }
    
    /// Convert to float
    pub fn to_float(self) -> f32 {
        self.0 as f32 / u32::MAX as f32
    }
}

impl Fixed32 {
    /// Multiply two fixed-point numbers
    pub fn mul(self, other: Self) -> Self {
        // Use u64 for intermediate calculation to avoid overflow
        let result = (self.0 as u64 * other.0 as u64) >> 32;
        Fixed32(result as u32)
    }
    
    /// Add two fixed-point numbers
    pub fn add(self, other: Self) -> Self {
        Fixed32(self.0.wrapping_add(other.0))
    }
    
    /// Subtract two fixed-point numbers
    pub fn sub(self, other: Self) -> Self {
        Fixed32(self.0.wrapping_sub(other.0))
    }

    /// Minimum distance between two fixed-point numbers, between 0 and 0.5
    pub fn dist(self, other: Self) -> Self {
        Fixed32((self - other).0.min((other - self).0))
    }

    /// Similarity between two fixed-point numbers, between 0 and 1
    pub fn similarity(self, other: Self) -> Self {
        Fixed32::ONE - Fixed32(self.dist(other).0 * 2)
    }
}


// Implement standard operators
impl std::ops::Mul for Fixed32 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}

impl std::ops::Add for Fixed32 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}

impl std::ops::Sub for Fixed32 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.sub(other)
    }
}


// Implement assignment operators
impl std::ops::MulAssign for Fixed32 {
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}

impl std::ops::AddAssign for Fixed32 {
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
}

impl std::ops::SubAssign for Fixed32 {
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(other);
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed32_creation_and_conversion() {
        // Test creation from various float values and conversion back to float
        // This tests the fundamental representation capability of our fixed-point format
        let test_cases = [
            (0.0, 0.0),       // Zero
            (1.0, 1.0),       // One (maximum value)
            (0.5, 0.5),       // Half
            (0.25, 0.25),     // Quarter
            (0.75, 0.75),     // Three quarters
            (0.999, 0.999),   // Near maximum
            (0.001, 0.001),   // Near zero
        ];

        for (input, expected) in test_cases {
            let fixed = Fixed32::from_float(input).unwrap();
            let output = fixed.to_float();
            assert!((output - expected).abs() < 1e-6, 
                    "Converting {input} to fixed and back gave {output}, expected {expected}");
        }
    }

    #[test]
    fn test_fixed32_addition() {
        // Test addition with various combinations including edge cases
        // Addition is critical for accumulating values in neural networks
        let test_cases = [
            // (a, b, expected)
            (0.0, 0.0, 0.0),          // Zero + Zero
            (0.5, 0.25, 0.75),        // Regular addition
            (0.75, 0.5, 0.25),        // Wrapping
        ];

        for (a, b, expected) in test_cases {
            let fixed_a = Fixed32::from_float(a).unwrap();
            let fixed_b = Fixed32::from_float(b).unwrap();
            let result = fixed_a.add(fixed_b).to_float();
            assert!((result - expected).abs() < 1e-6, 
                    "Adding {a} + {b} gave {result}, expected {expected}");
        }
    }

    #[test]
    fn test_fixed32_subtraction() {
        // Test subtraction with various combinations including edge cases
        // Subtraction is used in gradient calculations
        let test_cases = [
            // (a, b, expected)
            (0.5, 0.25, 0.25),        // Regular subtraction
            (0.25, 0.5, 0.75),        // Wrapping
            (0.0, 0.1, 0.9),          // Reverse wrapping
        ];

        for (a, b, expected) in test_cases {
            let fixed_a = Fixed32::from_float(a).unwrap();
            let fixed_b = Fixed32::from_float(b).unwrap();
            let result = fixed_a.sub(fixed_b).to_float();
            assert!((result - expected).abs() < 1e-6, 
                    "Subtracting {a} - {b} gave {result}, expected {expected}");
        }
    }

    #[test]
    fn test_fixed32_multiplication() {
        // Test multiplication with various combinations
        // Multiplication is core to the ring neural network's similarity calculation
        let test_cases = [
            // (a, b, expected)
            (0.0, 0.5, 0.0),          // Zero multiplication
            (1.0, 0.5, 0.5),          // Identity property
            (0.5, 0.5, 0.25),         // Regular multiplication
            (0.25, 0.25, 0.0625),     // Small values
            (0.999, 0.999, 0.998),    // Near maximum (approximate)
        ];

        for (a, b, expected) in test_cases {
            let fixed_a = Fixed32::from_float(a).unwrap();
            let fixed_b = Fixed32::from_float(b).unwrap();
            let result = fixed_a.mul(fixed_b).to_float();
            assert!((result - expected).abs() < 1e-3, 
                    "Multiplying {a} * {b} gave {result}, expected {expected}");
        }
    }

    #[test]
    fn test_fixed32_operator_overloads() {
        // Test that operator overloads work correctly
        // This ensures that the operators behave as expected in code
        let a = Fixed32::from_float(0.5).unwrap();
        let b = Fixed32::from_float(0.25).unwrap();
        
        assert!((a + b).to_float() - 0.75 < 1e-6);
        assert!((a - b).to_float() - 0.25 < 1e-6);
        assert!((a * b).to_float() - 0.125 < 1e-6);
    }
}
