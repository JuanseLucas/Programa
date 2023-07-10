import numpy as np
from scipy.integrate import quad, simps, trapz, solve_ivp
import time

# Definir las funciones para las integrales y la ecuación diferencial
def f1(x):
    return np.exp(-x**2)

def f2(x):
    return 2*x**2 + 3*x + 1

def f3(t, y):
    return -2*y*t

# Definir los límites de integración y el intervalo de tiempo para la ecuación diferencial
a1, b1 = 0, 1
a2, b2 = 0, 2
t_span = [0, 1]
y0 = [1]  # valor inicial para la ecuación diferencial

# Definir una función para calcular y mostrar los resultados
def calcular_y_mostrar(f, a, b, nombre, valor_verdadero_simps, valor_verdadero_trapz):
    # Calcular la integral con el método de Simpson y el método del trapecio
    x = np.linspace(a, b, 1000)
    y = f(x)

    inicio = time.time()
    integral_simps = simps(y, x)
    fin = time.time()
    tiempo_simps = fin - inicio

    inicio = time.time()
    integral_trapz = trapz(y, x)
    fin = time.time()
    tiempo_trapz = fin - inicio

    # Calcular los errores
    error_absoluto_simps = abs(valor_verdadero_simps - integral_simps)
    error_relativo_simps = abs((valor_verdadero_simps - integral_simps) / valor_verdadero_simps)

    error_absoluto_trapz = abs(valor_verdadero_trapz - integral_trapz)
    error_relativo_trapz = abs((valor_verdadero_trapz - integral_trapz) / valor_verdadero_trapz)

    # Mostrar los resultados
    print(f"Resultados para {nombre}:")
    print(f"  Método de Simpson: {integral_simps:.6f}, tiempo: {tiempo_simps:.6f}, error absoluto: {error_absoluto_simps:.6f}, error relativo: {error_relativo_simps:.6f}")
    print(f"  Método del trapecio: {integral_trapz:.6f}, tiempo: {tiempo_trapz:.6f}, error absoluto: {error_absoluto_trapz:.6f}, error relativo: {error_relativo_trapz:.6f}")

# Definir una función para el método de Euler
def euler_method(f, t_span, y0, h=0.01):
    t = np.arange(t_span[0], t_span[1], h)
    y = np.zeros(len(t))
    y[0] = y0[0]
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])
    return y[-1]

# Calcular y mostrar los resultados para las integrales
calcular_y_mostrar(f1, a1, b1, "integral 1", 0.746824, 0.746824)
calcular_y_mostrar(f2, a2, b2, "integral 2", 13.333, 13.5)

# Calcular y mostrar los resultados para la ecuación diferencial
inicio = time.time()
solucion_euler = euler_method(f3, t_span, y0)
fin = time.time()
tiempo_euler = fin - inicio

inicio = time.time()
solucion_rk = solve_ivp(f3, t_span, y0, method='RK45').y[0][-1]
fin = time.time()
tiempo_rk = fin - inicio

error_absoluto_euler = abs(0.367879 - solucion_euler)
error_relativo_euler = abs((0.367879 - solucion_euler) / 0.367879)

error_absoluto_rk = abs(0.367879 - solucion_rk)
error_relativo_rk = abs((0.367879 - solucion_rk) / 0.367879)

print(f"Resultados para la ecuación diferencial:")
print(f"  Método de Euler: {solucion_euler:.6f}, tiempo: {tiempo_euler:.6f}, error absoluto: {error_absoluto_euler:.6f}, error relativo: {error_relativo_euler:.6f}")
print(f"  Método de Runge-Kutta: {solucion_rk:.6f}, tiempo: {tiempo_rk:.6f}, error absoluto: {error_absoluto_rk:.6f}, error relativo: {error_relativo_rk:.6f}")
