import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. SIMULACIÓN DEL MODELO OFC (Red de 200x200 celdas)
# =====================================================================
N = 1000               # Tamaño de la red (40000 celdas en total)
pasos = 25000          # Número de eventos para tener excelente estadística
alpha = 0.21          # Elasticidad del terreno

cuadricula = np.random.uniform(0, 1.0, (N, N))
lista_tamaños = []

print("Simulando terremotos en la falla elástica... Por favor, espera.")

for t in range(pasos):
    F_max = np.max(cuadricula)
    delta_F = 1.0 - F_max
    cuadricula += delta_F
    
    inestables_x, inestables_y = np.where(cuadricula >= 1.0 - 1e-9)
    pila_inestables = list(zip(inestables_x, inestables_y))
    
    terremoto_tamaño = 0
    while pila_inestables:
        cx, cy = pila_inestables.pop()
        if cuadricula[cx, cy] >= 1.0:
            estres_acumulado = cuadricula[cx, cy]
            cuadricula[cx, cy] = 0.0
            terremoto_tamaño += 1
            
            estres_transferido = alpha * estres_acumulado
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N:
                    cuadricula[nx, ny] += estres_transferido
                    if cuadricula[nx, ny] >= 1.0:
                        pila_inestables.append((nx, ny))
                        
    lista_tamaños.append(terremoto_tamaño)

# Filtrar el transitorio inicial y sismos de tamaño 0
transitorio = int(pasos * 0.3)
S_datos = np.array([s for s in lista_tamaños[transitorio:] if s > 0])

# =====================================================================
# 2. CONVERSIÓN A ESCALA DE RICHTER Y CÁLCULO DE PROBABILIDADES
# =====================================================================
# Aplicamos nuestra fórmula de calibración física
M_datos = 1.5 * np.log10(S_datos) + 2.0

# Definimos los rangos oficiales de sismicidad
intervalos = [
    (2.0, 3.0, "Menor/Micro", "lightgray"),
    (3.0, 4.0, "Menor", "lightblue"),
    (4.0, 5.0, "Ligero", "lightgreen"),
    (5.0, 6.0, "Moderado", "khaki"),
    (6.0, 7.0, "Fuerte", "coral"),
    (7.0, 9.0, "Mayor/Gran Terremoto", "crimson")
]

n_total = len(M_datos)
probabilidades = []
etiquetas = []
colores = []

print("\n=======================================================")
print("  PROBABILIDAD DE OCURRENCIA POR MAGNITUD (RICHTER)")
print("=======================================================")
for limite_inf, limite_sup, nombre, color in intervalos:
    # Contamos cuántos terremotos caen en este rango de magnitudes
    conteo = np.sum((M_datos >= limite_inf) & (M_datos < limite_sup))
    
    # Calculamos su probabilidad (frecuencia relativa)
    probabilidad = (conteo / n_total) * 100
    
    probabilidades.append(probabilidad)
    etiquetas.append(f"M {limite_inf:.1f} - {limite_sup:.1f}\n({nombre})")
    colores.append(color)
    
    print(f"Magnitud [{limite_inf:.1f} a {limite_sup:.1f}) -> {probabilidad:6.3f}%  ({nombre})")
print("=======================================================\n")

# =====================================================================
# 3. GENERACIÓN DE GRÁFICAS PROFESIONALES
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfica A: Histograma de Probabilidades (Barras)
barras = ax1.bar(etiquetas, probabilidades, color=colores, edgecolor='black', alpha=0.8)
ax1.set_title("Probabilidad de que el Próximo Terremoto sea de Magnitud X", fontsize=11, fontweight='bold')
ax1.set_ylabel("Probabilidad de Ocurrencia (%)", fontsize=11)
ax1.grid(axis='y', linestyle=':', alpha=0.6)

# Añadir etiquetas de porcentaje sobre cada barra
for barra in barras:
    yval = barra.get_height()
    if yval > 0.001:
        ax1.text(barra.get_x() + barra.get_width()/2.0, yval + 1, f"{yval:.2f}%", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax1.text(barra.get_x() + barra.get_width()/2.0, yval + 1, "0.00%", 
                 ha='center', va='bottom', fontsize=9)

# Gráfica B: Probabilidad Acumulada P(>= M) en Escala Logarítmica
# Muestra la probabilidad de que ocurra un terremoto de al menos esa magnitud
magnitudes_ordenadas = np.sort(M_datos)
probabilidad_acumulada = 1.0 - np.arange(1, n_total + 1) / (n_total + 1)

ax2.plot(magnitudes_ordenadas, probabilidad_acumulada * 100, color='darkred', linewidth=2.5, label='Curva de Probabilidad')
ax2.set_yscale('log')
ax2.set_title("Probabilidad de que ocurra un Terremoto de Magnitud $\\geq M$", fontsize=11, fontweight='bold')
ax2.set_xlabel("Magnitud en Escala de Richter ($M$)", fontsize=11)
ax2.set_ylabel("Probabilidad (%)", fontsize=11)
ax2.grid(True, which="both", linestyle=':', alpha=0.5)
ax2.set_xlim(left=2.0)

plt.tight_layout()
plt.show()