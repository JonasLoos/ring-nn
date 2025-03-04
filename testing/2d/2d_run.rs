use ring_nn::{Fixed32, RingNetwork};
use ring_nn::loss::{MSELoss, Loss};
use ring_nn::optimizer::{Adam, SGD, Optimizer};
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    test_2d_toy_function()
}

fn test_2d_toy_function() -> std::io::Result<()> {
    // This test trains a network to approximate a simple 2D function
    // We'll use a simple quadratic function: f(x,y) = x^2 + y^2
    // Scaled to [0,1] range
    
    // Function to generate target values from inputs
    let simple_quadratic = |x: f32, y: f32| -> f32 {
        let a = (2.0 * x - 1.0) * std::f32::consts::PI;
        let b = (2.0 * y - 1.0) * std::f32::consts::PI;
        let result = (a.cos() + 1.0) * (b.cos() + 1.0) / 4.0 / 2.0;
        result.clamp(0.0, 1.0)
    };
    
    // Create a network with 5 layers
    // Note: In this implementation, all layers must have the same size
    let layer_size = 4;  // Using 4 neurons per layer
    let mut network = RingNetwork::new();
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    network.add_layer(layer_size);
    
    // Generate training data: grid of points in [0,1]×[0,1]
    let grid_size = 10;
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
    
    // Create an optimizer with a smaller learning rate
    // let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let mut optimizer = SGD::new(0.002);
    
    // Train network
    let mut losses = Vec::new();
    let epochs = 5000;  // More epochs for better convergence
    
    println!("Training network...");
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
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
            
            // Backward pass
            network.backward(&loss_grad, &caches);
            
            // Apply gradients
            optimizer.step(&mut network);
        }
        
        // Record average loss
        epoch_loss /= data.len() as f32;
        losses.push(epoch_loss);
        
        // Print loss every 100 epochs
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, epoch_loss);
        }
    }
    
    // Print final loss
    println!("Initial loss: {}, Final loss: {}", losses[0], losses[losses.len() - 1]);
    
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
            
            // Write to CSV: x, y, expected, predicted
            writeln!(results_file, "{},{},{},{}", x, y, expected, prediction.iter().map(|p| p.to_float().to_string()).collect::<Vec<_>>().join(","))?;
        }
    }
    
    println!("Results saved to results.csv");
    Ok(())
}