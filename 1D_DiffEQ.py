
import jax.numpy as jnp
import matplotlib.pyplot as plt


# Define the parameters
k = 1
a = 3 # cannot be 1 for some reason
x0 = 1
t_end = 1

# Forward Euler method
def forward_euler(deltaT):
    steps = int(t_end/deltaT)
    t = jnp.linspace(0, t_end, steps + 1) # Initializing an array for t
    x = jnp.zeros(steps + 1) # initializing an array for x
    x = x.at[0].set(x0) # making first element x_0

    for n in range(steps):
        x = x.at[n + 1].set(x[n] * (1 - deltaT) + a * deltaT) # Forward Eulor Method Formula

    return t, x


# Solved by hand
def exact_solution(t):
    return a / k + (x0 - a / k) * jnp.exp(-k * t)


# Simulate with different deltaT values
t1, x1 = forward_euler(0.1)
t2, x2 = forward_euler(0.01)
t3, x3 = forward_euler(0.001)

exact_x = exact_solution(1)

# Final approximations
print(f"Final approximation with deltaT = 0.1: x({t1[-1]}) = {x1[-1]}")
print(f"Final approximation with deltaT = 0.01: x({t2[-1]}) = {x2[-1]}")
print(f"Final approximation with deltaT = 0.001: x({t3[-1]}) = {x3[-1]}")
print(f"  Exact solution: x = {exact_x}")


# Plotting the results
plt.figure(figsize=(10, 6))

plt.plot(t1, x1, label="deltaT = 0.1", marker='o')
plt.plot(t2, x2, label="deltaT = 0.01", marker='s')
plt.plot(t3, x3, label="deltaT = 0.001", marker='^')

# Plotting the exact solution
t_fine = jnp.linspace(0, t_end, 1000)
exact_x_fine = exact_solution(t_fine)
plt.plot(t_fine, exact_x_fine, label="Exact Solution", color='black', linestyle='--')



plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.title('Forward Euler Method for dx/dt = -kx + a')
plt.legend()
plt.grid(True)
plt.show()


