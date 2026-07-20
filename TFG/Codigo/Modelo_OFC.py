import numpy as np
import matplotlib.pyplot as plt
import powerlaw

# --- 1. CONFIGURACIÓN DE LA SIMULACIÓN OFC ---
N = 40                 # Cuadrícula de 40x40 (1600 celdas)
pasos = 30000          # Número de terremotos a simular (más pasos = mejor estadística para MLE)
F_th = 1.0             # Umbral de ruptura
alpha = 0.21           # Elasticidad/Disipación (alpha < 0.25)

# Inicializar matriz con valores de estrés aleatorios en [0, F_th)
cuadricula = np.random.uniform(0, F_th, (N, N))
lista_tamaños_terremotos = []

print("Simulando terremotos... Por favor, espera.")

# --- 2. BUCLE DE SIMULACIÓN ---
for t in range(pasos):
    # Carga tectónica cuasi-estática (Driving)
    F_max = np.max(cuadricula)
    delta_F = F_th - F_max
    cuadricula += delta_F
    
    # Encontrar la celda desencadenante
    inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
    pila_inestables = list(zip(inestables_x, inestables_y))
    
    terremoto_tamaño = 0
    
    # Propagación de la avalancha
    while pila_inestables:
        cx, cy = pila_inestables.pop()
        
        if cuadricula[cx, cy] >= F_th:
            estres_acumulado = cuadricula[cx, cy]
            cuadricula[cx, cy] = 0.0  # Relajación
            terremoto_tamaño += 1
            
            # Distribución de estrés a los vecinos
            estres_transferido = alpha * estres_acumulado
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N:
                    cuadricula[nx, ny] += estres_transferido
                    if cuadricula[nx, ny] >= F_th:
                        pila_inestables.append((nx, ny))
                        
    lista_tamaños_terremotos.append(terremoto_tamaño)

print("Simulación completada. Procesando datos con MLE...")

# --- 3. PREPARACIÓN DE DATOS Y FILTRADO ---
# Descartamos el primer 30% como transitorio para asegurar el estado crítico
transitorio = int(pasos * 0.3)
datos_estacionarios = lista_tamaños_terremotos[transitorio:]

# MLE requiere trabajar únicamente con valores mayores que cero
datos_filtrados = [s for s in datos_estacionarios if s > 0]

# --- 4. AJUSTE DE MÁXIMA VEROSIMILITUD (MLE) ---
# Creamos el objeto Fit de la librería powerlaw.
# - discrete=True: Porque el tamaño de los terremotos (número de celdas rotas) son números enteros.
# - xmin=3: Descartamos terremotos extremadamente pequeños (1 o 2 celdas) para limpiar el ruido de escala.
ajuste = powerlaw.Fit(datos_filtrados, discrete=True, xmin=3)

# Extraemos el exponente de la ley de potencias (tau)
# Nota: La librería llama internamente 'alpha' al exponente por estándar estadístico, pero en física es nuestro 'tau'.
tau_mle = ajuste.alpha 
sigma_tau = ajuste.sigma  # El error estándar del exponente estimado

print(f"\n==================================================")
print(f"RESULTADO DEL AJUSTE (MLE):")
print(f"Exponente crítico (tau): {tau_mle:.3f} ± {sigma_tau:.3f}")
print(f"Límite inferior óptimo (xmin): {ajuste.xmin}")
print(f"==================================================\n")

# --- 5. GRÁFICO PROFESIONAL CON LA LIBRERÍA POWERLAW ---
plt.figure(figsize=(8, 6))

# Dibujar la Densidad de Probabilidad (PDF) de los datos reales agrupados de forma óptima
ajuste.plot_pdf(color='darkred', marker='o', linestyle='None', alpha=0.7, 
                label='Datos Simulación OFC', markersize=6)

# Dibujar la recta del ajuste teórico MLE obtenido
ajuste.power_law.plot_pdf(color='navy', linestyle='--', linewidth=2, 
                          label=f'Ajuste MLE ($\\tau$ = {tau_mle:.2f} $\\pm$ {sigma_tau:.2f})')

# Configuración del gráfico
plt.title(f"Ley de Gutenberg-Richter del Modelo OFC ({N}x{N})", fontsize=13, fontweight='bold')
plt.xlabel('Tamaño del Terremoto ($S$)', fontsize=12)
plt.ylabel('Densidad de Probabilidad $P(S)$', fontsize=12)
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()