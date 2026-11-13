import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def esfera(x):
    return np.sum(x ** 2)

def gwo(funcion, dim=2, n_lobos=20, max_iter=100, lb=-20, ub=20):
    lobos = np.random.uniform(lb, ub, (n_lobos, dim))
    alfa, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    s_alfa = s_beta = s_delta = float("inf")

    hist_lobos, hist_lideres, hist_fit = [], [], []

    for t in range(max_iter):
        for i in range(n_lobos):
            lobos[i] = np.clip(lobos[i], lb, ub)
            fit = funcion(lobos[i])
            if fit < s_alfa:
                s_delta, delta = s_beta, beta.copy()
                s_beta, beta = s_alfa, alfa.copy()
                s_alfa, alfa = fit, lobos[i].copy()
            elif fit < s_beta:
                s_delta, delta = s_beta, beta.copy()
                s_beta, beta = fit, lobos[i].copy()
            elif fit < s_delta:
                s_delta, delta = fit, lobos[i].copy()

        hist_lobos.append(lobos.copy())
        hist_lideres.append(np.array([alfa, beta, delta]))
        hist_fit.append(s_alfa)

        a = 2 - 2 * t / max_iter
        for i in range(n_lobos):
            nuevas = []
            for lider in (alfa, beta, delta):
                A = 2 * a * np.random.rand(dim) - a
                C = 2 * np.random.rand(dim)
                D = np.abs(C * lider - lobos[i])
                nuevas.append(lider - A * D)
            lobos[i] = np.mean(nuevas, axis=0)

    return hist_lobos, hist_lideres, np.maximum(hist_fit, 1e-300)

# ---------- Ejecutar el algoritmo (2 dimensiones para poder dibujarlo) ----------
LB, UB, ITER = -20, 20, 100
lobos_h, lideres_h, fit_h = gwo(esfera, dim=2, n_lobos=20, max_iter=ITER, lb=LB, ub=UB)

# ---------- Preparar la figura ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Izquierda: mapa de colores de la función (el "terreno" donde cazan los lobos)
xs = np.linspace(LB, UB, 200)
X, Y = np.meshgrid(xs, xs)
ax1.contourf(X, Y, X ** 2 + Y ** 2, levels=30, cmap="Blues")
ax1.plot(0, 0, "rx", markersize=12, label="Presa (mínimo)")

omegas = ax1.scatter([], [], c="gray", s=40, label="Omegas (ω)")
p_alfa = ax1.scatter([], [], c="gold", marker="*", s=300, edgecolor="black", label="Alfa (α)")
p_beta = ax1.scatter([], [], c="orange", marker="*", s=200, edgecolor="black", label="Beta (β)")
p_delta = ax1.scatter([], [], c="brown", marker="*", s=150, edgecolor="black", label="Delta (δ)")
ax1.set_xlim(LB, UB); ax1.set_ylim(LB, UB)
ax1.legend(loc="upper right", fontsize=8)

# Derecha: curva de convergencia
linea, = ax2.plot([], [], "o-", color="#F2A541")
ax2.set_yscale("log")
ax2.set_xlim(0, ITER)
ax2.set_ylim(fit_h.min() / 10, fit_h.max() * 10)
ax2.set_xlabel("Iteración"); ax2.set_ylabel("Mejor fitness (escala log)")
ax2.set_title("Convergencia")

# ---------- Parámetros de fluidez ----------
SUBPASOS = 4          
FPS = 20              
TOTAL = (ITER - 1) * SUBPASOS + 1

# ---------- Dibujar cada cuadro de la animación ----------
def actualizar(f):
    i = f // SUBPASOS                 
    k = (f % SUBPASOS) / SUBPASOS     
    j = min(i + 1, ITER - 1)          

    # Mezcla lineal: posición = (1 - k) * actual + k * siguiente
    pos = lobos_h[i] * (1 - k) + lobos_h[j] * k
    lid = lideres_h[i] * (1 - k) + lideres_h[j] * k

    omegas.set_offsets(pos)
    p_alfa.set_offsets(lid[[0]])
    p_beta.set_offsets(lid[[1]])
    p_delta.set_offsets(lid[[2]])
    ax1.set_title(f"Manada de lobos · iteración {i + 1}/{ITER}")
    linea.set_data(range(i + 1), fit_h[: i + 1])

anim = FuncAnimation(fig, actualizar, frames=TOTAL, interval=1000 / FPS, repeat=False)
anim.save("gwo.gif", writer="pillow", fps=FPS)
plt.tight_layout()
plt.show()