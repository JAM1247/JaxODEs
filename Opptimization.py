
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

""" Please by aware that running this code can take a while """

# Gradient descent parameters
learning_rate = 0.1
num_iterations = 100

# All of these must be floats
k = 1.0  
a = 3.0  
x0 = 1.0
t_end = 1.0 

# Forward Euler method function
def forward_euler(deltaT, k, a, x0, t_end):
    steps = int(t_end / deltaT)
    t = jnp.linspace(0, t_end, steps + 1)  # array containing all time increments
    x = jnp.zeros(steps + 1)  # array to contain x values  
    x = x.at[0].set(x0)  # initializing x[0] to x-naught
    
    for n in range(steps):  # Forward Euler Formula 
        x = x.at[n + 1].set(x[n] * (1 - k * deltaT) + a * deltaT) 
    
    return t, x

# Loss function
def loss(x0, a):
    deltaT = 0.001
    t, x = forward_euler(deltaT, k, a, x0, t_end)
    return jnp.trapezoid(x**2, t)  # Integrating x^2

# Gradients of change in parameters
grad_x0 = jax.grad(loss, argnums=0)  # dL/dx0
grad_a = jax.grad(loss, argnums=1)   # dL/da

# Initialize arrays to store parameter values over iterations
x0_vals = [x0]
a_vals = [a]
loss_vals = []

# Gradient Descent Loop
for i in range(num_iterations):
    # Calculate the gradients
    grad_x0_value = grad_x0(x0, a)
    grad_a_value = grad_a(x0, a)
    
    # Update parameters using gradient descent
    x0 -= learning_rate * grad_x0_value
    a -= learning_rate * grad_a_value
    
    # Store the updated parameter values
    x0_vals.append(x0)
    a_vals.append(a)
    
    # Calculate and store the loss
    L_value = loss(x0, a)
    loss_vals.append(L_value)

# Plotting the convergence of x0 and a
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x0_vals, label='x0')
plt.plot(a_vals, label='a')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Convergence of x0 and a')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_vals, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.title('Convergence of Loss')

plt.tight_layout()
plt.show()















