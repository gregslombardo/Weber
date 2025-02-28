import numpy as np
import matplotlib.pyplot as plt

# Onda cuadrada
def square_wave(x):
    return np.where((x % (2 * np.pi)) < np.pi, 1, -1)

# Serie de Fourier a N terminos
def fourier_series_approx(x, N):
    approximation = np.zeros_like(x)
    for k in range(1, N+1):
        approximation += (4 / (np.pi * (2*k - 1))) * np.sin((2*k - 1) * x)
    return approximation

x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Grafica de onda cuadrada
plt.figure(figsize=(10, 6))
plt.plot(x, square_wave(x), label='Square Wave', color='black', linewidth=2)

# Iteraciones sobre terminos crecientes
for N in [1, 3, 10, 30]:
    plt.plot(x, fourier_series_approx(x, N), label=f'N = {N}')

plt.title('Fourier Series Approximation of a Square Wave')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.ylim(-1.5, 1.5)
plt.legend()
plt.grid(True)
plt.show()