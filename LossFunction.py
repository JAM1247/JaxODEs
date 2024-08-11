import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

""" Please by aware that running this code can take a while """


# All of these must be floats
k = 1.0  
a = 3.0  
x0 = 1.0
t_end = 1.0 

# Jax Doesn't allow these values in the Jit
deltaT = 0.001
steps = int(t_end / deltaT)


@jax.jit
# Forward Euler method function
def forward_euler(deltaT, k, a, x0, t_end):
    
    #steps = jnp.floor_divide(t_end, deltaT)  # JAX function for integer division
    t = jnp.linspace(0, t_end, steps + 1) # array containing all time increments
    x = jnp.zeros(steps + 1) # array to contain x values  
    x = x.at[0].set(x0) # initilziigin x[0] to x-naught
    
    for n in range(steps): # Forward Eulor Formula 
        x = x.at[n + 1].set(x[n] * (1 - k * deltaT) + a * deltaT) 
    
    return t, x

# Incredibly Tedious
def exact_solution():
    return (jnp.exp(-2*k) * (a**2 * (jnp.exp(2*k) * (2*k - 3) + 4 * jnp.exp(k) - 1) +  2 * a * (jnp.exp(k) - 1)**2 * k * x0 + (jnp.exp(2*k) - 1) * k**2 * x0**2)) / (2 * k**3)

@jax.jit
# Loss Function
def loss(x0, a):
    t, x = forward_euler(deltaT, k, a, x0, t_end)
    return jnp.trapezoid(x**2, t)  # Integrating x^2

# Gradients
grad_x0 = jax.grad(loss, argnums=0)  # dL/dx0
grad_a = jax.grad(loss, argnums=1)   # dL/da

# Calculate the loss and its gradients
L_value = loss(x0, a)
grad_x0_value = grad_x0(x0, a)
grad_a_value = grad_a(x0, a)

exact = exact_solution()

print(f"Loss L(x0, a) = {L_value}")
print(f"Gradient dL/dx0 = {grad_x0_value}")
print(f"Gradient dL/da = {grad_a_value}")
print(f"Exact Solution = {exact}")
print(f"Differene between exact and loss function: {abs(exact-L_value)}")



