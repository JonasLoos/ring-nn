use crate::Fixed32;
use rand::seq::SliceRandom;

/// Data loader for batched training
pub struct DataLoader {
    data: Vec<Vec<u32>>,
    targets: Vec<Vec<Fixed32>>,
    batch_size: usize,
    shuffle: bool,
}

impl DataLoader {
    /// Create a new DataLoader
    pub fn new(
        data: Vec<Vec<u32>>, 
        targets: Vec<Vec<Fixed32>>, 
        batch_size: usize, 
        shuffle: bool
    ) -> Self {
        assert_eq!(data.len(), targets.len(), 
                  "Data and targets must have same number of samples");
        
        DataLoader {
            data,
            targets,
            batch_size,
            shuffle,
        }
    }
    
    /// Get the number of samples in the dataset
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    
    /// Create an iterator over the dataset
    pub fn iter(&self) -> DataLoaderIterator {
        DataLoaderIterator::new(&self.data, &self.targets, self.batch_size, self.shuffle)
    }
}

/// Iterator for DataLoader that yields batches
pub struct DataLoaderIterator<'a> {
    data: &'a [Vec<u32>],
    targets: &'a [Vec<Fixed32>],
    batch_size: usize,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<'a> DataLoaderIterator<'a> {
    /// Create a new DataLoaderIterator
    pub fn new(
        data: &'a [Vec<u32>],
        targets: &'a [Vec<Fixed32>],
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        let n_samples = data.len();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        // Shuffle indices if requested
        if shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }
        
        DataLoaderIterator {
            data,
            targets,
            batch_size,
            indices,
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = (Vec<&'a Vec<u32>>, Vec<&'a Vec<Fixed32>>);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None; // End of iteration
        }
        
        // Calculate end index for this batch
        let end_idx = std::cmp::min(
            self.current_idx + self.batch_size,
            self.indices.len()
        );
        
        // Collect batch data and targets
        let mut batch_data = Vec::with_capacity(end_idx - self.current_idx);
        let mut batch_targets = Vec::with_capacity(end_idx - self.current_idx);
        
        for idx in self.current_idx..end_idx {
            let sample_idx = self.indices[idx];
            batch_data.push(&self.data[sample_idx]);
            batch_targets.push(&self.targets[sample_idx]);
        }
        
        // Update current index for next batch
        self.current_idx = end_idx;
        
        Some((batch_data, batch_targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_creation() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        
        let targets = vec![
            vec![Fixed32::from_float(0.1).unwrap()],
            vec![Fixed32::from_float(0.2).unwrap()],
            vec![Fixed32::from_float(0.3).unwrap()],
            vec![Fixed32::from_float(0.4).unwrap()],
        ];
        
        let loader = DataLoader::new(data, targets, 2, false);
        
        assert_eq!(loader.len(), 4);
        assert_eq!(loader.batch_size(), 2);
    }

    #[test]
    fn test_dataloader_iteration() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        
        let targets = vec![
            vec![Fixed32::from_float(0.1).unwrap()],
            vec![Fixed32::from_float(0.2).unwrap()],
            vec![Fixed32::from_float(0.3).unwrap()],
            vec![Fixed32::from_float(0.4).unwrap()],
        ];
        
        let loader = DataLoader::new(data, targets, 2, false);
        
        let mut batches = Vec::new();
        for (batch_data, batch_targets) in loader.iter() {
            batches.push((batch_data.len(), batch_targets.len()));
        }
        
        // Should have 2 batches of size 2
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], (2, 2));
        assert_eq!(batches[1], (2, 2));
    }
}
