//! Visualization utilities for Ring Neural Networks

use crate::RingNetwork;

/// Plot training loss
pub fn plot_loss(losses: &[f32]) {
    // Simple text-based visualization of loss trend
    println!("Training Loss Curve:");
    println!("---------------------");
    
    if losses.is_empty() {
        println!("No loss data available.");
        return;
    }
    
    // Find min and max for scaling
    let min_loss = *losses.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_loss = *losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    // Avoid division by zero if all losses are the same
    let range = if (max_loss - min_loss).abs() < 1e-6 { 1.0 } else { max_loss - min_loss };
    
    // Number of rows in the plot
    let height = 15;
    let width = losses.len();
    
    // Create the plot
    let mut plot = vec![vec![' '; width]; height];
    
    // Fill in the plot
    for (i, &loss) in losses.iter().enumerate() {
        let normalized = if range == 0.0 { 0.5 } else { (loss - min_loss) / range };
        let row = ((1.0 - normalized) * (height - 1) as f32).round() as usize;
        let row = row.min(height - 1);
        plot[row][i] = '*';
    }
    
    // Print the plot
    for row in plot {
        print!("|");
        for col in row {
            print!("{}", col);
        }
        println!("|");
    }
    
    // Print x-axis
    print!("+");
    for _ in 0..width {
        print!("-");
    }
    println!("+");
    
    // Print scale
    println!("Min Loss: {:.6}, Max Loss: {:.6}", min_loss, max_loss);
    println!("Epochs: 0 to {}", losses.len() - 1);
    
    // Print final loss
    if let Some(&final_loss) = losses.last() {
        println!("Final Loss: {:.6}", final_loss);
    }
}

/// Visualize weight distribution on the ring
pub fn visualize_ring_weights(weights: &[Vec<u32>]) {
    println!("Ring Weights Distribution:");
    println!("--------------------------");
    
    // Flatten all weights into a single vector
    let all_weights: Vec<u32> = weights.iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    
    if all_weights.is_empty() {
        println!("No weights to visualize.");
        return;
    }
    
    // Define number of bins (16 bins for a cleaner visualization)
    let num_bins = 16usize;
    let bin_size = u32::MAX / (num_bins as u32);
    
    // Count weights in each bin
    let mut bins = vec![0usize; num_bins];
    for &w in &all_weights {
        let bin_idx = (w / bin_size) as usize;
        let bin_idx = bin_idx.min(num_bins - 1); // Handle edge case for max value
        bins[bin_idx] += 1;
    }
    
    // Find max count for scaling
    let max_count = *bins.iter().max().unwrap_or(&1);
    
    // Display histogram
    for i in 0..num_bins {
        let start = (i as u32) * bin_size;
        let end = if i == num_bins - 1 { u32::MAX } else { start + bin_size - 1 };
        
        let bar_width = 50usize;
        let bar_length = (bins[i] * bar_width) / max_count;
        
        print!("[{:3}-{:3}] ", start / (u32::MAX / 256), end / (u32::MAX / 256));
        for _ in 0..bar_length {
            print!("#");
        }
        println!(" ({})", bins[i]);
    }
    
    println!("Total weights: {}", all_weights.len());
}

/// Visualize weights for a specific layer in the network
pub fn visualize_ring_weights_for_network(network: &RingNetwork, layer_idx: usize) {
    if layer_idx >= network.layers.len() {
        println!("Layer index out of bounds.");
        return;
    }
    
    visualize_ring_weights(&network.layers[layer_idx].weights);
}

/// Visualize network structure
pub fn visualize_network_structure(network: &RingNetwork) {
    println!("Network Structure:");
    println!("-----------------");
    
    let mut prev_size = 0;
    
    for (i, layer) in network.layers.iter().enumerate() {
        if i == 0 {
            prev_size = layer.input_size;
        }
        
        println!("Layer {}: {} -> {} neurons", i, prev_size, layer.output_size);
        prev_size = layer.output_size;
    }
    
    println!("Total parameters: {}", network.layers.iter()
        .map(|layer| layer.input_size * layer.output_size + layer.output_size)
        .sum::<usize>());
} 