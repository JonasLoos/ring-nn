use ring_nn::{Fixed32, RingNetwork, optimizer, loss, visualization};
use ring_nn::optimizer::Optimizer;
use ring_nn::loss::Loss;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define problem parameters
    let size = 3;
    
    // Create the network
    let mut network = RingNetwork::new();
    network.add_layer(size);
    network.add_layer(size);
    
    // Visualize initial network structure
    visualization::visualize_network_structure(&network);
    
    // Create synthetic data
    let data = vec![
        vec![Fixed32::from_float(0.1).unwrap(), Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.3).unwrap()],
        vec![Fixed32::from_float(0.4).unwrap(), Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.6).unwrap()],
        vec![Fixed32::from_float(0.7).unwrap(), Fixed32::from_float(0.8).unwrap(), Fixed32::from_float(0.9).unwrap()],
        vec![Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.4).unwrap(), Fixed32::from_float(0.6).unwrap()],
        vec![Fixed32::from_float(0.3).unwrap(), Fixed32::from_float(0.6).unwrap(), Fixed32::from_float(0.9).unwrap()],
        vec![Fixed32::from_float(0.1).unwrap(), Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.9).unwrap()]
    ];
    
    let targets = vec![
        vec![Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.3).unwrap(), Fixed32::from_float(0.4).unwrap()],
        vec![Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.6).unwrap(), Fixed32::from_float(0.7).unwrap()],
        vec![Fixed32::from_float(0.8).unwrap(), Fixed32::from_float(0.9).unwrap(), Fixed32::from_float(1.0).unwrap()],
        vec![Fixed32::from_float(0.3).unwrap(), Fixed32::from_float(0.5).unwrap(), Fixed32::from_float(0.7).unwrap()],
        vec![Fixed32::from_float(0.4).unwrap(), Fixed32::from_float(0.7).unwrap(), Fixed32::from_float(1.0).unwrap()],
        vec![Fixed32::from_float(0.2).unwrap(), Fixed32::from_float(0.6).unwrap(), Fixed32::from_float(1.0).unwrap()]
    ];
    
    // Create an optimizer (demonstrate both SGD and Adam)
    println!("\nTraining with SGD:");
    let mut network_sgd = network.clone();
    let mut sgd_optimizer = optimizer::SGD::new(0.02);
    
    // Train with SGD
    let sgd_losses = train_with_optimizer(
        &mut network_sgd,
        &data,
        &targets,
        2,
        100,
        &mut sgd_optimizer,
    );
    
    // Visualize SGD results
    visualization::plot_loss(&sgd_losses);
    
    // Now train with Adam
    println!("\nTraining with Adam:");
    let mut network_adam = network.clone();
    let mut adam_optimizer = optimizer::Adam::new(0.005, 0.9, 0.999, 1e-8);
    
    // Train with Adam
    let adam_losses = train_with_optimizer(
        &mut network_adam,
        &data,
        &targets,
        2,
        100,
        &mut adam_optimizer,
    );
    
    // Visualize Adam results
    visualization::plot_loss(&adam_losses);
    
    // Compare final losses
    println!("\nFinal SGD loss: {}", sgd_losses.last().unwrap_or(&f32::NAN));
    println!("Final Adam loss: {}", adam_losses.last().unwrap_or(&f32::NAN));
    
    // Visualize weight distributions
    println!("\nSGD Weight Distribution:");
    visualization::visualize_ring_weights(&network_sgd.layers[0].weights);
    
    println!("\nAdam Weight Distribution:");
    visualization::visualize_ring_weights(&network_adam.layers[0].weights);
    
    // Test both networks
    let test_input = vec![50, 100, 150];
    let sgd_prediction = network_sgd.forward(&test_input);
    let adam_prediction = network_adam.forward(&test_input);
    
    println!("\nTest Input: {:?}", test_input);
    println!("SGD Prediction: {}", sgd_prediction[0].to_float());
    println!("Adam Prediction: {}", adam_prediction[0].to_float());
    
    // Demonstrate cross entropy loss
    let y_true = vec![Fixed32::from_float(1.0).unwrap()];
    let y_pred = vec![Fixed32::from_float(0.7).unwrap()];
    let ce_loss = loss::CrossEntropyLoss::forward(&y_pred, &y_true);
    println!("\nCross Entropy Loss Example: {}", ce_loss);
    
    Ok(())
}


/// Train using a generic optimizer
fn train_with_optimizer<T: Optimizer>(
    network: &mut RingNetwork,
    data: &[Vec<Fixed32>],
    targets: &[Vec<Fixed32>],
    batch_size: usize,
    epochs: usize,
    optimizer: &mut T,
) -> Vec<f32> {
    use rand::seq::SliceRandom;
    
    let n_samples = data.len();
    assert_eq!(n_samples, targets.len(), "Data and targets must have same length");
    
    let mut losses = Vec::with_capacity(epochs);
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = rand::rng();
    
    for epoch in 0..epochs {
        // Shuffle the data
        indices.shuffle(&mut rng);
        
        let mut epoch_loss = 0.0;
        
        // Process in batches
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = std::cmp::min(batch_start + batch_size, n_samples);
            let batch_indices = &indices[batch_start..batch_end];
            
            // Accumulate gradients over the batch
            for &idx in batch_indices {
                // Forward pass with caching
                let (predictions, caches) = network.forward_with_cache(&data[idx]);
                
                // Calculate loss
                let loss = loss::MSELoss::forward(&predictions, &targets[idx]);
                epoch_loss += loss;
                
                // Calculate loss gradients
                let loss_grad = loss::MSELoss::backward(&predictions, &targets[idx]);
                
                // Backward pass
                network.backward(&loss_grad, &caches);
            }
            
            // Apply accumulated gradients using the optimizer
            optimizer.step(network);
        }
        
        // Record average loss for this epoch
        epoch_loss /= n_samples as f32;
        losses.push(epoch_loss);
        
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: Loss = {}", epoch, epoch_loss);
        }
    }
    
    losses
}
