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

/// Visualize weights on the ring
pub fn visualize_ring_weights(weights: &[Vec<u32>], ring_size: u32) {
    println!("Ring Weights Distribution:");
    println!("--------------------------");
    
    // Count weights in each sector of the ring
    let num_sectors = 16; // Divide the ring into 16 sectors for visualization
    let sector_size = ring_size / num_sectors;
    
    // Initialize counters for each sector
    let mut sector_counts = vec![0; num_sectors as usize];
    
    // Count weights in each sector
    for row in weights {
        for &weight in row {
            let sector = (weight / sector_size) as usize % num_sectors as usize;
            sector_counts[sector] += 1;
        }
    }
    
    // Find the maximum count for scaling
    let max_count = *sector_counts.iter().max().unwrap_or(&1);
    
    // Print the distribution
    let max_bar_length = 50;
    for (i, &count) in sector_counts.iter().enumerate() {
        let start = i as u32 * sector_size;
        let end = start + sector_size - 1;
        
        // Scale the bar length
        let bar_length = ((count as f32 / max_count as f32) * max_bar_length as f32).round() as usize;
        
        // Print the bar
        print!("[{:3}-{:3}] ", start, end);
        for _ in 0..bar_length {
            print!("#");
        }
        println!(" ({})", count);
    }
    
    // Print total number of weights
    let total_weights: usize = sector_counts.iter().sum();
    println!("Total weights: {}", total_weights);
}

/// Visualize weights for a specific layer in the network
pub fn visualize_ring_weights_for_network(network: &RingNetwork, layer_idx: usize) {
    if layer_idx >= network.num_layers() {
        println!("Error: Layer index {} is out of bounds (network has {} layers)",
                layer_idx, network.num_layers());
        return;
    }
    
    // Now we can directly access the weights since layers is public
    visualize_ring_weights(&network.layers[layer_idx].weights, network.ring_size());
}

/// Visualize network structure
pub fn visualize_network_structure(network: &RingNetwork) {
    println!("Network Structure:");
    println!("-----------------");
    println!("Ring Size: {}", network.ring_size());
    
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