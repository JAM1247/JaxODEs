import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Gradient descent parameters
learning_rate = 0.1
num_iterations = 300 # higher it goes the more it will converge, my computer cna't really handle more then this

# All of these must be floats
k = 1.0  
a = 3.0  
x0 = 1.0
t_end = 1.0 

# Wasn't able to figure it out using Jit constraints  
def forward_euler(deltaT, k, a, x0, t_end):
    steps = int(t_end / deltaT)
    t = jnp.linspace(0, t_end, steps + 1)  # array containing all time increments
    
    # updates the x_n value 
    def euler_update(x, k, a, deltaT):
        return x * (1 - k * deltaT) + a * deltaT 
    
    # calculates the x_n+1 value 
    def update_step(i, val):
        return val.at[i + 1].set(euler_update(val[i], k, a, deltaT))
    
    # Built in function that solves for x at each iteration of t
    x = jax.lax.fori_loop(0, steps, update_step, jnp.zeros(steps + 1).at[0].set(x0))
    return t, x

# Loss function
def loss(x0, a):
    deltaT = 0.001
    t, x = forward_euler(deltaT, k, a, x0, t_end)
    return jnp.trapezoid(x**2, t)   # Indefite integral of x^2 over t domain 

# Gradients of change in parameters
grad_x0 = jax.grad(loss, argnums=0)  # dL/dx0
grad_a = jax.grad(loss, argnums=1)   # dL/da

# Initialize arrays
x0_vals = []
a_vals = []
loss_vals = []

# Gradient Descent Loop
for i in range(num_iterations):
    # Calculating gradients
    grad_x0_value = grad_x0(x0, a)
    grad_a_value = grad_a(x0, a)
    
    # Update parameters using gradient descent
    x0 -= learning_rate * grad_x0_value
    a -= learning_rate * grad_a_value
   
   # Appending the updated values
    x0_vals.append(x0)
    a_vals.append(a)
    
    # Calculating the loss function and storing it for graphing
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

# Plotting the loss value
plt.subplot(1, 2, 2)
plt.plot(loss_vals, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.title('Convergence of Loss')

plt.tight_layout()
plt.show()
