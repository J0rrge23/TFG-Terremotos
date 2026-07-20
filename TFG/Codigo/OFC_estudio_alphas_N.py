import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import warnings

warnings.filterwarnings('ignore')

# --- REUTILIZAMOS EL MOTOR OFC ---
def simular_ofc_y_guardar_red(N, pasos, alpha, F_th=1.0):
    cuadricula = np.random.uniform(0, F_th, (N, N))
    lista_tamaños = []
    
    for t in range(pasos):
        F_max = np.max(cuadricula)
        delta_F = F_th - F_max
        cuadricula += delta_F
        
        inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
        pila_inestables = list(zip(inestables_x, inestables_y))
        
        terremoto_tamaño = 0
        while pila_inestables:
            cx, cy = pila_inestables.pop()
            if cuadricula[cx, cy] >= F_th:
                estres_acumulado = cuadricula[cx, cy]
                cuadricula[cx, cy] = 0.0
                terremoto_tamaño += 1
                
                estres_transferido = alpha * estres_acumulado
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        cuadricula[nx, ny] += estres_transferido
                        if cuadricula[nx, ny] >= F_th:
                            pila_inestables.append((nx, ny))
                            
        lista_tamaños.append(terremoto_tamaño)
        
    transitorio = int(pasos * 0.3)
    datos_estacionarios = [s for s in lista_tamaños[transitorio:] if s > 0]
    return datos_estacionarios, cuadricula

# --- SIMULACIÓN Y GENERACIÓN DE GRÁFICAS ---
N_fijo = 35
pasos_simulacion = 15000
valores_alpha = [0.16, 0.18, 0.20, 0.22, 0.24]

# Configurar el lienzo para los ajustes individuales (1 fila, 5 columnas)
fig_fits, ejes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
fig_fits.suptitle("Ajustes de Ley de Potencias Individuales por cada $\\alpha$ (MLE)", fontsize=14, fontweight='bold')

cuadricula_final = None

for i, al in enumerate(valores_alpha):
    print(f"Procesando curvas para alpha = {al}...")
    datos, cuadricula_final = simular_ofc_y_guardar_red(N_fijo, pasos_simulacion, al)
    
    # Ajuste powerlaw
    ajuste = powerlaw.Fit(datos, discrete=True, xmin=3, verbose=False)
    tau = ajuste.power_law.alpha
    sigma = ajuste.power_law.sigma
    
    # Graficar PDF de los datos
    ajuste.plot_pdf(ax=ejes[i], color='darkred', marker='o', linestyle='None', alpha=0.6, markersize=5)
    # Graficar Ajuste teórico
    ajuste.power_law.plot_pdf(ax=ejes[i], color='navy', linestyle='--', lw=2)
    
    ejes[i].set_title(f"$\\alpha$ = {al}\n$\\tau$ = {tau:.2f} $\\pm$ {sigma:.2f}", fontsize=11)
    ejes[i].set_xlabel("Tamaño $S$")
    ejes[i].grid(True, which="both", linestyle=':', alpha=0.5)
    if i == 0:
        ejes[i].set_ylabel("Densidad de Probabilidad $P(S)$")

plt.tight_layout()
plt.show()

# --- GRAFICAR LA CUADRÍCULA 2D AL FINAL DE LA SIMULACIÓN ---
plt.figure(figsize=(7, 6))
plt.imshow(cuadricula_final, cmap='magma', origin='lower', vmin=0, vmax=1.0)
cbar = plt.colorbar(label='Nivel de Estrés Mecánico ($F$)')
plt.title(f"Estado de la Cuadrícula OFC ({N_fijo}x{N_fijo}) al Final de la Simulación", fontsize=12, fontweight='bold')
plt.xlabel("Coordenada X (Celdas)")
plt.ylabel("Coordenada Y (Celdas)")
plt.show()

# =====================================================================
# 0. FUNCIÓN MOTOR DE LA SIMULACIÓN OFC
# =====================================================================
def simular_ofc(N, pasos, alpha, F_th=1.0):
    """
    Ejecuta el modelo OFC y devuelve los tamaños de los terremotos
    en el estado estacionario (descartando el primer 30% como transitorio).
    """
    cuadricula = np.random.uniform(0, F_th, (N, N))
    lista_tamaños = []
    
    for t in range(pasos):
        # Carga tectónica
        F_max = np.max(cuadricula)
        delta_F = F_th - F_max
        cuadricula += delta_F
        
        inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
        pila_inestables = list(zip(inestables_x, inestables_y))
        
        terremoto_tamaño = 0
        
        while pila_inestables:
            cx, cy = pila_inestables.pop()
            if cuadricula[cx, cy] >= F_th:
                estres_acumulado = cuadricula[cx, cy]
                cuadricula[cx, cy] = 0.0
                terremoto_tamaño += 1
                
                estres_transferido = alpha * estres_acumulado
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        cuadricula[nx, ny] += estres_transferido
                        if cuadricula[nx, ny] >= F_th:
                            pila_inestables.append((nx, ny))
                            
        lista_tamaños.append(terremoto_tamaño)
        
    # Descartar transitorio (30%)
    transitorio = int(pasos * 0.3)
    datos_estacionarios = lista_tamaños[transitorio:]
    return [s for s in datos_estacionarios if s > 0]


# =====================================================================
# PASO 2: ESTUDIO PARAMÉTRICO (Efecto de la disipación alpha)
# =====================================================================
print("--- EJECUTANDO PASO 2: ESTUDIO PARAMÉTRICO ---")
N_fijo = 35
pasos_simulacion = 15000  # Aumentar a 30000 para el TFG final
valores_alpha = [0.16, 0.18, 0.20, 0.22, 0.24]

lista_tau = []
lista_sigma = []

for al in valores_alpha:
    print(f"Simulando para alpha = {al}...")
    datos = simular_ofc(N=N_fijo, pasos=pasos_simulacion, alpha=al)
    
    # Ajuste por Máxima Verosimilitud (MLE)
    ajuste = powerlaw.Fit(datos, discrete=True, xmin=3, verbose=False)
    lista_tau.append(ajuste.power_law.alpha)  # <--- Acceso explícito y seguro
    lista_sigma.append(ajuste.power_law.sigma)

# Graficar Paso 2
plt.figure(figsize=(7, 5))
plt.errorbar(valores_alpha, lista_tau, yerr=lista_sigma, fmt='o-', color='crimson', 
             ecolor='black', capsize=5, elinewidth=1.5, markeredgecolor='black', markersize=8)
plt.title(f"Efecto de la Disipación en el Exponente Crítico (Red {N_fijo}x{N_fijo})", fontsize=11, fontweight='bold')
plt.xlabel("Parámetro de Elasticidad / Transmisión ($\\alpha$)", fontsize=10)
plt.ylabel("Exponente Crítico $\\tau$ (MLE)", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================================================
# PASO 3: COLAPSO DE TAMAÑO FINITO (FSS) - CORREGIDO
# =====================================================================
print("\n--- EJECUTANDO PASO 3: FINITE-SIZE SCALING (FSS) ---")
alpha_fijo = 0.21
tamaños_red = [20, 30, 40]  # Tres escalas espaciales distintas
pasos_fss = 20000

# Parámetros de escalamiento teóricos para OFC
D = 2.0      # Dimensión fractal del área de avalancha (S ~ L^D)
tau_teorico = 1.25  # Usamos un valor promedio para el colapso visual

resultados_fss = {}

for size in tamaños_red:
    print(f"Simulando para red de tamaño N = {size}...")
    resultados_fss[size] = simular_ofc(N=size, pasos=pasos_fss, alpha=alpha_fijo)

# Creación de gráficos para el Paso 3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot A: PDFs normales (Sin colapsar)
ax1.set_title("A. Distribuciones de Probabilidad Originales $P(S)$", fontsize=11, fontweight='bold')
colores = {20: 'teal', 30: 'darkorange', 40: 'indigo'}

for size in tamaños_red:
    datos = resultados_fss[size]
    
    # 1. Obtener los bordes y el PDF por separado
    bordes, pdf = powerlaw.pdf(datos, linear_bins=False)
    
    # 2. Calcular los centros geométricos para tener el tamaño correcto (n)
    x = np.sqrt(bordes[:-1] * bordes[1:])
    y = pdf
    
    ax1.scatter(x, y, label=f"N = {size}", color=colores[size], alpha=0.7, edgecolor='black', s=35)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel("Tamaño del Terremoto ($S$)", fontsize=10)
ax1.set_ylabel("$P(S)$", fontsize=10)
ax1.grid(True, which="both", linestyle=':', alpha=0.5)
ax1.legend()

# Subplot B: Colapso de Curvas (FSS)
ax2.set_title("B. Colapso de Curvas (Escalamiento Universal)", fontsize=11, fontweight='bold')

for size in tamaños_red:
    datos = resultados_fss[size]
    
    # 1. Obtener los bordes y el PDF por separado
    bordes, pdf = powerlaw.pdf(datos, linear_bins=False)
    
    # 2. Calcular los centros geométricos
    x = np.sqrt(bordes[:-1] * bordes[1:])
    y = pdf
    
    # --- APLICACIÓN DE LAS VARIABLES ESCALADAS ---
    x_escalado = x / (size**D)
    y_escalado = y * (x**tau_teorico)  # Equivalente a P(S)*S^tau
    
    ax2.scatter(x_escalado, y_escalado, label=f"N = {size} (Escalado)", 
                color=colores[size], alpha=0.7, edgecolor='black', s=35)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Variable Rescalada: $S / N^D$", fontsize=10)
ax2.set_ylabel("Función de Escala: $P(S) \\cdot S^{\\tau}$", fontsize=10)
ax2.grid(True, which="both", linestyle=':', alpha=0.5)
ax2.legend()

plt.suptitle(f"Análisis de Escala de Tamaño Finito (FSS) con $\\alpha$ = {alpha_fijo}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()