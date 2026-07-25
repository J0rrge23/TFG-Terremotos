import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import warnings

# Desactivar advertencias matemáticas de la librería powerlaw para una consola limpia
warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURACIÓN DE LA SIMULACIÓN ANISOTRÓPICA
# =====================================================================
N = 50                 # Tamaño de la corteza (50x50 = 2500 celdas)
pasos = 30000          # Número de sismos simulados para una estadística robusta
F_th = 1.0             # Límite de fricción estática (Umbral de ruptura)

# Parámetros del terreno
alpha_0 = 0.21         # Elasticidad/Acoplamiento medio del terreno
epsilon = 0.30         # Intensidad de la anisotropía (Falla horizontal en eje X)

# Definición explícita del tensor/matriz de pesos direccionales
# Eje X (Izquierda/Derecha) facilita la propagación. Eje Y (Arriba/Abajo) la amortigua.
pesos_direccionales = {
    (1, 0):  alpha_0 * (1 + epsilon),   # Derecha
    (-1, 0): alpha_0 * (1 + epsilon),   # Izquierda
    (0, 1):  alpha_0 * (1 - epsilon),   # Arriba
    (0, -1): alpha_0 * (1 - epsilon)    # Abajo
}

# Inicializar la corteza con un mapa de fuerzas aleatorio fuera del equilibrio
cuadricula = np.random.uniform(0, F_th, (N, N))
lista_tamaños = []

print("===============================================================")
print(f" EJECUTANDO MODELO OFC ANISOTRÓPICO (Red: {N}x{N} | Pasos: {pasos})")
print(f" Parámetro base (alpha_0): {alpha_0} | Intensidad de Falla (epsilon): {epsilon}")
print("===============================================================")
print("Simulando dinámicas de ruptura tectónica... Por favor, espera.")

# =====================================================================
# 2. MOTOR DE SIMULACIÓN DE AVALANCHAS (OFC MODIFICADO)
# =====================================================================
for t in range(pasos):
    # Carga tectónica cuasi-estática (Desplazamiento lento de placas)
    F_max = np.max(cuadricula)
    delta_F = F_th - F_max
    cuadricula += delta_F
    
    # Localizar focos de inestabilidad sísmica (Epicentros iniciales)
    inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
    pila_inestables = list(zip(inestables_x, inestables_y))
    
    terremoto_tamaño = 0
    
    # Propagación dinámica de la onda de ruptura (Avalancha)
    while pila_inestables:
        cx, cy = pila_inestables.pop()
        
        if cuadricula[cx, cy] >= F_th:
            estres_acumulado = cuadricula[cx, cy]
            cuadricula[cx, cy] = 0.0  # Relajación completa del foco (Deslizamiento)
            terremoto_tamaño += 1
            
            # Distribución anisotrópica del estrés de Coulomb a los 4 vecinos vecinos
            for (dx, dy), alpha_dir in pesos_direccionales.items():
                nx, ny = cx + dx, cy + dy
                
                # Condiciones de frontera disipativas (La energía se pierde en los bordes)
                if 0 <= nx < N and 0 <= ny < N:
                    cuadricula[nx, ny] += alpha_dir * estres_acumulado
                    if cuadricula[nx, ny] >= F_th:
                        pila_inestables.append((nx, ny))
                        
    lista_tamaños.append(terremoto_tamaño)

print("Simulación finalizada con éxito. Procesando análisis geofísico...")

# =====================================================================
# 3. FILTRADO ESTADÍSTICO Y CALIBRACIÓN SISMOLÓGICA
# =====================================================================
# Descartar el primer 30% de la serie temporal para asegurar el estado estacionario (SOC)
transitorio = int(pasos * 0.3)
datos_estacionarios = lista_tamaños[transitorio:]

# Filtrar sismos nulos para el análisis de leyes de potencias
S_datos = np.array([s for s in datos_estacionarios if s > 0])

# Calibración física a la Escala de Richter (Magnitud M)
M_datos = 1.5 * np.log10(S_datos) + 2.0

# =====================================================================
# 4. ANÁLISIS ESTADÍSTICO AVANZADO (MLE VIA POWERLAW)
# =====================================================================
# Ajuste de Máxima Verosimilitud excluyendo ruido de escala pequeña (xmin=3)
ajuste = powerlaw.Fit(S_datos, discrete=True, xmin=3, verbose=False)
tau_mle = ajuste.power_law.alpha
sigma_tau = ajuste.power_law.sigma

print("\n=======================================================")
print("             RESULTADOS DEL ANÁLISIS SÍSMICO           ")
print("=======================================================")
print(f" Exponente crítico de Gutenberg-Richter (tau): {tau_mle:.3f} ± {sigma_tau:.3f}")
print(f" Límite inferior óptimo de escala (xmin): {ajuste.xmin}")
print(f" Magnitud máxima registrada en la falla: M = {np.max(M_datos):.2f}")
print(f" Sismo promedio en estado estacionario: M = {np.mean(M_datos):.2f}")
print("=======================================================\n")

# =====================================================================
# 5. GENERACIÓN DE GRÁFICAS PARA LA MEMORIA DEL TFG
# =====================================================================
fig = plt.figure(figsize=(15, 6))

# --- GRÁFICA A: Ley de Gutenberg-Richter Anisotrópica (PDF Log-Log) ---
ax1 = fig.add_subplot(1, 2, 1)
ajuste.plot_pdf(ax=ax1, color='darkorange', marker='o', linestyle='None', alpha=0.6, 
                label='Datos Simulación Anisotrópica', markersize=6)
ajuste.power_law.plot_pdf(ax=ax1, color='navy', linestyle='--', linewidth=2.5, 
                          label=f'Ajuste MLE ($\\tau$ = {tau_mle:.2f} $\\pm$ {sigma_tau:.2f})')

ax1.set_title("A. Ley de Gutenberg-Richter en Falla Anisotrópica", fontsize=12, fontweight='bold')
ax1.set_xlabel('Tamaño del Terremoto / Área de Ruptura ($S$)', fontsize=11)
ax1.set_ylabel('Densidad de Probabilidad $P(S)$', fontsize=11)
ax1.grid(True, which="both", linestyle=':', alpha=0.5)
ax1.legend(fontsize=10, loc='upper right')

# --- GRÁFICA B: Estado de Estrés e Impronta Estructural (2D Campo) ---
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(cuadricula, cmap='magma', origin='lower', vmin=0, vmax=F_th)
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Nivel de Estrés Mecánico Residual ($F$)', fontsize=11)

ax2.set_title("B. Mapa de Estrés de la Falla en el Estado Final", fontsize=12, fontweight='bold')
ax2.set_xlabel("Coordenada X (Dirección de la Falla)", fontsize=11)
ax2.set_ylabel("Coordenada Y (Eje Perpendicular)", fontsize=11)

plt.tight_layout()
plt.show()