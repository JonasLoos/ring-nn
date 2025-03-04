use ring_nn::{Fixed32, RingNetwork};
use ring_nn::loss::{MSELoss, Loss};
use ring_nn::optimizer::{SGD, Optimizer};
use std::fs::{self, File, create_dir_all};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> std::io::Result<()> {
    test_2d_toy_function()
}

fn test_2d_toy_function() -> std::io::Result<()> {
    // Create a timestamped folder for logs
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let log_dir = format!("training_logs_2d_run_{}", timestamp);
    create_dir_all(&log_dir)?;
    
    println!("Saving logs to: {}", log_dir);
    
    // This test trains a network to approximate a simple 2D function
    // We'll use a simple quadratic function: f(x,y) = x^2 + y^2
    // Scaled to [0,1] range
    
    // Function to generate target values from inputs
    let simple_quadratic = |x: f32, y: f32| -> f32 {
        // Simple quadratic function: x^2 + y^2, scaled to [0,1]
        let result = (x * x + y * y) / 2.0;
        result.clamp(0.0, 1.0)
    };
    
    // Create a network with 3 layers
    // Note: In this implementation, all layers must have the same size
    let layer_size = 4;  // Using 4 neurons per layer
    let mut network = RingNetwork::new();
    network.add_layer(layer_size);  // Layer 1
    network.add_layer(layer_size);  // Layer 2
    network.add_layer(layer_size);  // Layer 3
    
    // Save initial network state
    save_network_state(&network, &format!("{}/network_initial.txt", log_dir))?;
    
    // Generate training data: grid of points in [0,1]×[0,1]
    let grid_size = 5;
    let mut data = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = i as f32 / (grid_size - 1) as f32;
            let y = j as f32 / (grid_size - 1) as f32;
            
            // Convert to Fixed32
            let x_fixed = Fixed32::from_float(x).unwrap();
            let y_fixed = Fixed32::from_float(y).unwrap();
            
            // Calculate target value
            let z = simple_quadratic(x, y);
            let z_fixed = Fixed32::from_float(z).unwrap();
            
            // Create input vector with padding to match layer size
            let mut input = vec![x_fixed, y_fixed];
            // Pad with zeros to match layer size
            while input.len() < layer_size {
                input.push(Fixed32::ZERO);
            }
            
            // Create target vector with the function output in the first position
            let target = vec![z_fixed, z_fixed, z_fixed, z_fixed];
            
            data.push(input);
            targets.push(target);
        }
    }
    
    // Save training data
    save_training_data(&data, &targets, &format!("{}/training_data.csv", log_dir))?;
    
    // Create an optimizer with a smaller learning rate
    // let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let mut optimizer = SGD::new(0.002);
    
    // Train network
    let mut losses = Vec::new();
    let epochs = 5000;  // More epochs for better convergence
    
    // Create loss log file
    let mut loss_file = File::create(format!("{}/loss.csv", log_dir))?;
    writeln!(loss_file, "epoch,loss")?;
    
    println!("Training network...");
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        // Create epoch directory for detailed logs
        let epoch_dir = format!("{}/epoch_{:04}", log_dir, epoch);
        if epoch % 100 == 0 {
            create_dir_all(&epoch_dir)?;
        }
        
        // Collect gradients for logging
        let mut all_gradients = Vec::new();
        
        for i in 0..data.len() {
            // Forward pass with caching
            let (mut predictions, caches) = network.forward_with_cache(&data[i]);

            // only loss for 3rd neuron
            for j in 0..predictions.len() {
                if j != 2 {
                    predictions[j] = targets[i][j];
                }
            }

            // Calculate loss
            let loss = MSELoss::forward(&predictions, &targets[i]);
            epoch_loss += loss;
            
            // Calculate loss gradients
            let loss_grad = MSELoss::backward(&predictions, &targets[i]);
            
            // Save predictions and gradients if it's a logging epoch
            if epoch % 100 == 0 {
                all_gradients.push(loss_grad.clone());
            }
            
            // Backward pass
            network.backward(&loss_grad, &caches);
            
            // Apply gradients
            optimizer.step(&mut network);
        }
        
        // Record average loss
        epoch_loss /= data.len() as f32;
        losses.push(epoch_loss);
        
        // Write loss to log file
        writeln!(loss_file, "{},{}", epoch, epoch_loss)?;
        
        // Save detailed logs every 100 epochs
        if epoch % 100 == 0 {
            // Save network state
            save_network_state(&network, &format!("{}/network_state.txt", epoch_dir))?;
            
            // Save gradients
            save_gradients(&all_gradients, &format!("{}/gradients.csv", epoch_dir))?;
            
            // Save predictions on test grid
            save_predictions(&network, simple_quadratic, layer_size, 10, &format!("{}/predictions.csv", epoch_dir))?;
        }
        
        // Print loss every 100 epochs
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, epoch_loss);
        }
    }
    
    // Print final loss
    println!("Initial loss: {}, Final loss: {}", losses[0], losses[losses.len() - 1]);
    
    // Save final network state
    save_network_state(&network, &format!("{}/network_final.txt", log_dir))?;
    
    // Generate results for visualization
    // Create a finer grid for visualization
    let vis_grid_size = 20;
    let mut results_file = File::create("results.csv")?;
    
    for i in 0..vis_grid_size {
        for j in 0..vis_grid_size {
            let x = i as f32 / (vis_grid_size - 1) as f32;
            let y = j as f32 / (vis_grid_size - 1) as f32;
            
            // Get expected output
            let expected = simple_quadratic(x, y);
            
            // Convert to u32 for raw input
            let x_u32 = (x * u32::MAX as f32) as u32;
            let y_u32 = (y * u32::MAX as f32) as u32;
            
            // Create input vector with padding
            let mut input = vec![x_u32, y_u32];
            // Pad with zeros to match layer size
            while input.len() < layer_size {
                input.push(0);
            }
            
            // Make prediction
            let prediction = network.forward(&input);
            
            // First output should estimate the function
            let pred = prediction[0].to_float();
            
            // Write to CSV: x, y, expected, predicted
            writeln!(results_file, "{},{},{},{}", x, y, expected, pred)?;
        }
    }
    
    println!("Results saved to results.csv");
    println!("All training logs saved to {}", log_dir);
    Ok(())
}

// Helper function to save network state
fn save_network_state(network: &RingNetwork, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "Network Structure:")?;
    writeln!(file, "Number of layers: {}", network.layers.len())?;
    
    for (i, layer) in network.layers.iter().enumerate() {
        writeln!(file, "\nLayer {}:", i)?;
        writeln!(file, "Size: {}", layer.size)?;
        
        writeln!(file, "\nWeights:")?;
        for (j, row) in layer.weights.iter().enumerate() {
            let weights_str: String = row.iter()
                .map(|w| w.to_float().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(file, "  Neuron {}: [{}]", j, weights_str)?;
        }
        
        writeln!(file, "\nAlpha values:")?;
        let alpha_str: String = layer.alpha.iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(file, "  [{}]", alpha_str)?;
    }
    
    Ok(())
}

// Helper function to save gradients
fn save_gradients(gradients: &[Vec<f32>], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write header
    let mut header = String::from("sample");
    for i in 0..gradients[0].len() {
        header.push_str(&format!(",grad_{}", i));
    }
    writeln!(file, "{}", header)?;
    
    // Write gradients
    for (i, grad) in gradients.iter().enumerate() {
        let mut line = format!("{}", i);
        for g in grad {
            line.push_str(&format!(",{}", g));
        }
        writeln!(file, "{}", line)?;
    }
    
    Ok(())
}

// Helper function to save training data
fn save_training_data(data: &[Vec<Fixed32>], targets: &[Vec<Fixed32>], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write header
    let mut header = String::from("sample");
    for i in 0..data[0].len() {
        header.push_str(&format!(",input_{}", i));
    }
    for i in 0..targets[0].len() {
        header.push_str(&format!(",target_{}", i));
    }
    writeln!(file, "{}", header)?;
    
    // Write data
    for i in 0..data.len() {
        let mut line = format!("{}", i);
        for d in &data[i] {
            line.push_str(&format!(",{}", d.to_float()));
        }
        for t in &targets[i] {
            line.push_str(&format!(",{}", t.to_float()));
        }
        writeln!(file, "{}", line)?;
    }
    
    Ok(())
}

// Helper function to save predictions on a test grid
fn save_predictions(network: &RingNetwork, func: impl Fn(f32, f32) -> f32, layer_size: usize, grid_size: usize, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write header
    writeln!(file, "x,y,expected,predicted_0,predicted_1,predicted_2,predicted_3")?;
    
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = i as f32 / (grid_size - 1) as f32;
            let y = j as f32 / (grid_size - 1) as f32;
            
            // Get expected output
            let expected = func(x, y);
            
            // Convert to u32 for raw input
            let x_u32 = (x * u32::MAX as f32) as u32;
            let y_u32 = (y * u32::MAX as f32) as u32;
            
            // Create input vector with padding
            let mut input = vec![x_u32, y_u32];
            // Pad with zeros to match layer size
            while input.len() < layer_size {
                input.push(0);
            }
            
            // Make prediction
            let prediction = network.forward(&input);
            
            // Write to CSV: x, y, expected, all predictions
            let pred_str: String = prediction.iter()
                .map(|p| p.to_float().to_string())
                .collect::<Vec<_>>()
                .join(",");
            
            writeln!(file, "{},{},{},{}", x, y, expected, pred_str)?;
        }
    }
    
    Ok(())
}