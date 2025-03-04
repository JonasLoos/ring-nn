import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Run the 2d_run.rs file
print("Running 2D function approximation...")
os.system("cargo run --bin 2d_run")

# Read the results
print("Loading results...")
results = np.loadtxt("results.csv", delimiter=",")

# Extract data
x = results[:, 0]
y = results[:, 1]
z_expected = results[:, 2]
z_predicted = results[:, 3:]

# Reshape for grid plotting
grid_size = int(np.sqrt(len(x)))
X = x.reshape(grid_size, grid_size)
Y = y.reshape(grid_size, grid_size)
Z_expected = z_expected.reshape(grid_size, grid_size)
Z_predicted = z_predicted.reshape(grid_size, grid_size, z_predicted.shape[1])
Z_error = np.abs(Z_expected[...,None] - Z_predicted)

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 6*z_predicted.shape[1]))


for i in range(z_predicted.shape[1]):
    # Plot expected function
    ax1 = fig.add_subplot(z_predicted.shape[1], 3, i*3+1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_expected, cmap='viridis', alpha=0.8)
    ax1.set_title('Expected Function: f(x,y) = x² + y²')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot predicted function
    ax2 = fig.add_subplot(z_predicted.shape[1], 3, i*3+2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_predicted[...,i], cmap='plasma', alpha=0.8)
    ax2.set_title('Neural Network Prediction')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot error
    ax3 = fig.add_subplot(z_predicted.shape[1], 3, i*3+3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_error[...,i], cmap='hot', alpha=0.8)
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Error')

    # Add colorbar
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    # Calculate error statistics
    mean_error = np.mean(Z_error)
    max_error = np.max(Z_error)
    print(f"Mean absolute error: {mean_error:.4f}")
    print(f"Maximum absolute error: {max_error:.4f}")

    # Add text with error statistics
    plt.figtext(0.5, 0.01, f"Mean absolute error: {mean_error:.4f}, Maximum absolute error: {max_error:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout()
# plt.savefig("function_approximation.png")
# print("Visualization saved as function_approximation.png")
plt.show()
