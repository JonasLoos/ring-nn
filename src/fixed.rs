/// Fixed-point number using all 32 bits for fraction (0.32 format)
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Fixed32(pub u32);

impl Fixed32 {
    /// Maximum value (0.99999...)
    pub const MAX: Self = Fixed32(u32::MAX);
    
    /// Zero value
    pub const ZERO: Self = Fixed32(0);
    
    /// One value (not representable perfectly - use MAX instead)
    pub const ONE: Self = Fixed32(u32::MAX);
    
    /// Create from float (clamped to 0-1 range)
    pub fn from_float(value: f32) -> Self {
        let clamped = value.clamp(0.0, 1.0);
        Fixed32((clamped * u32::MAX as f32) as u32)
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
    
    /// Add two fixed-point numbers (with saturation)
    pub fn add(self, other: Self) -> Self {
        Fixed32(self.0.saturating_add(other.0))
    }
    
    /// Subtract two fixed-point numbers (with saturation)
    pub fn sub(self, other: Self) -> Self {
        Fixed32(self.0.saturating_sub(other.0))
    }
    
    /// Divide two fixed-point numbers
    pub fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            return Fixed32::MAX; // Handle division by zero
        }
        // Use u64 for intermediate calculation
        let result = ((self.0 as u64) << 32) / other.0 as u64;
        // Saturate at max value if overflow
        if result > u32::MAX as u64 {
            Fixed32::MAX
        } else {
            Fixed32(result as u32)
        }
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

impl std::ops::Div for Fixed32 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.div(other)
    }
} 