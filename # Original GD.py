# Original GD
new_theta -= alpha * gradient                                                                            # type: ignore

# Extended to Stochastic Gradient Descent (SGD)
batch_indices = np.random.choice(len(data), batch_size, replace=False)                                  # type: ignore
batch_gradient = np.mean([gradient_function(data[i]) for i in batch_indices], axis=0)                       # type: ignore
new_theta -= alpha * batch_gradient             

# Momentum-based
velocity = beta * velocity + (1 - beta) * batch_gradient                                 # type: ignore
new_theta -= alpha * velocity                                                            # type: ignore

# Adam Optimizer
mean_grad = beta1 * mean_grad + (1 - beta1) * batch_gradient  # First moment                                # type: ignore
mean_squared_grad = beta2 * mean_squared_grad + (1 - beta2) * (batch_gradient ** 2)  # Second moment        # type: ignore

mean_grad_corrected = mean_grad / (1 - beta1 ** iteration)  # Bias correction for first moment                   # type: ignore
mean_squared_grad_corrected = mean_squared_grad / (1 - beta2 ** iteration)  # Bias correction for second moment     # type: ignore

new_theta -= alpha * mean_grad_corrected / (np.sqrt(mean_squared_grad_corrected)                        
                                            
                                            
 for iteration in range(max_iter):
    gradients = np.zeros_like(theta)                               # Initialize gradient vector
    
                                                                                # Loop through each data point
    for i in range(len(X)):
        prediction = np.dot(X[i], theta)            # Compute prediction for one sample
        error = prediction - y[i]  # Error for one sample
        
        # Update each gradient separately
        for j in range(len(theta)):
            gradients[j] += error * X[i][j]                                              # type: ignore
