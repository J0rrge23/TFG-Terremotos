#Codigo de la pila de arena de una cuadricula 30x30, donde incluyo una partícula en una de las casillas a cada paso de tiempo
#las variables son discretas y el tiempo es discreto, la simulación se realiza en un bucle for que recorre un número determinado de pasos de tiempo. En cada paso de tiempo, se selecciona una casilla aleatoria de la cuadricula y se incrementa el valor de esa casilla en 1, representando la adición de una partícula a esa posición.
#las condiciones de frontera son disipativas, es decir, si una casilla alcanza un valor mayor o igual a 4, se "desborda" y distribuye sus partículas a las casillas vecinas (arriba, abajo, izquierda, derecha). Si una casilla está en el borde de la cuadricula y se desborda, las partículas que se desbordan se pierden.

#MEdir la carga media de la cuadricula a lo largo del tiempo, es decir, el promedio de partículas por casilla en cada paso de tiempo. Esto se puede hacer sumando todos los valores de la cuadricula y dividiéndolos por el número total de casillas (900 en este caso). Almacenar estos valores en una lista para poder analizarlos posteriormente.
#Tambien quiero medir la criticidad de la cuadricula, es decir, el número de casillas que tienen un valor mayor o igual a 4 en cada paso de tiempo. Esto se puede hacer recorriendo la cuadricula y contando cuántas casillas cumplen esta condición. Almacenar estos valores en una lista para poder analizarlos posteriormente.
import random
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. CONFIGURACIÓN DE LA SIMULACIÓN
# =====================================================================
N = 30                 # Tamaño de la cuadrícula (30x30 = 900 casillas)
pasos = 30000          # Número de pasos de tiempo (aumentado para mejor estadística)
cuadricula = [[0 for _ in range(N)] for _ in range(N)]  # Inicializar con ceros

# --- VARIABLES DE CONTROL OPTIMIZADAS O(1) ---
total_particulas = 0   # Contador global de granos en la red
sitios_criticos = 0    # Contador de casillas con exactamente 3 granos

# --- LISTAS PARA ALMACENAR MÉTRICAS ---
lista_carga_media = []
lista_tamano_avalancha = []  # Desbordes por paso
lista_sitios_criticos = []   # Casillas con 3 granos por paso

# =====================================================================
# 2. BUCLE PRINCIPAL DE TIEMPO (Pila de Arena BTW)
# =====================================================================
for t in range(pasos):
    # Seleccionar una casilla aleatoria e incrementar su valor
    x = random.randint(0, N - 1)
    y = random.randint(0, N - 1)
    
    val_antiguo = cuadricula[x][y]
    cuadricula[x][y] += 1
    total_particulas += 1
    
    # Actualización O(1) del estado de sitio crítico (h = 3)
    if val_antiguo == 3:
        sitios_criticos -= 1
    elif cuadricula[x][y] == 3:
        sitios_criticos += 1
    
    # Resolver desbordes (avalanchas) en cadena
    pila_inestables = []
    if cuadricula[x][y] >= 4:
        pila_inestables.append((x, y))
        
    desbordes_este_paso = 0
    
    while pila_inestables:
        cx, cy = pila_inestables.pop()
        
        if cuadricula[cx][cy] >= 4:
            cuadricula[cx][cy] -= 4
            desbordes_este_paso += 1
            
            if cuadricula[cx][cy] == 3:
                sitios_criticos += 1
            
            # Distribuir 1 grano a los 4 vecinos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < N and 0 <= ny < N:
                    val_antiguo_vecino = cuadricula[nx][ny]
                    cuadricula[nx][ny] += 1
                    
                    if val_antiguo_vecino == 3:
                        sitios_criticos -= 1
                    elif cuadricula[nx][ny] == 3:
                        sitios_criticos += 1
                    
                    if cuadricula[nx][ny] >= 4:
                        pila_inestables.append((nx, ny))
                else:
                    # Se pierde por el borde (disipación)
                    total_particulas -= 1

    # Registro de métricas tras completar la avalancha
    carga_media = total_particulas / (N * N)
    lista_carga_media.append(carga_media)
    lista_tamano_avalancha.append(desbordes_este_paso)
    lista_sitios_criticos.append(sitios_criticos)

print(
    f"Simulación finalizada tras {pasos} pasos.\n"
    f"Carga media final: {lista_carga_media[-1]:.4f}\n"
    f"Sitios críticos al final: {lista_sitios_criticos[-1]} de {N*N}\n"
    f"Desbordes totales: {sum(lista_tamano_avalancha)}\n"
)

# =====================================================================
# 3. FILTRADO DEL ESTADO ESTACIONARIO (SOC)
# =====================================================================
inicio_estacionario = int(pasos * 0.3)  # Descarta el primer 30% (transitorio)

tiempo_interes = np.arange(inicio_estacionario, pasos)
carga_interes = lista_carga_media[inicio_estacionario:]
criticos_interes = lista_sitios_criticos[inicio_estacionario:]

avalanchas_estacionarias = lista_tamano_avalancha[inicio_estacionario:]
avalanchas_filtradas = [s for s in avalanchas_estacionarias if s > 0]

# =====================================================================
# 4. FIGURA 1: EVOLUCIÓN TEMPORAL EN EQUILIBRIO
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle(f"Evolución en el Estado Estacionario (SOC) — Pasos {inicio_estacionario} a {pasos}", 
             fontsize=13, fontweight='bold')

# Carga Media
ax1.plot(tiempo_interes, carga_interes, color='teal', lw=1, label='Carga Media Activa')
ax1.axhline(17/8, color='crimson', linestyle='--', lw=1.5, label=r'Límite Teórico $N \to \infty$ (2.125)')
ax1.set_ylabel('Carga Media\n(partículas / casilla)', fontsize=10)
ax1.set_title('1. Fluctuaciones de la Carga Media', fontsize=11, loc='left')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')
ax1.set_ylim(min(carga_interes) * 0.98, max(carga_interes) * 1.02)

# Sitios Críticos
ax2.plot(tiempo_interes, criticos_interes, color='darkorange', lw=0.8, alpha=0.8, label=r'Sitios Críticos ($h = 3$)')
ax2.set_xlabel('Paso de tiempo (t)', fontsize=11)
ax2.set_ylabel('Nº de Sitios Críticos', fontsize=10)
ax2.set_title('2. Densidad de Sitios Críticos', fontsize=11, loc='left')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# =====================================================================
# 5. FIGURA 2: MARGINAL $P(S)$ VS. EXCEDENCIA $P(S \geq s)$
# =====================================================================
if len(avalanchas_filtradas) > 10:
    s_datos = np.array(avalanchas_filtradas)
    
    # Rango de ajuste de la Zona de Escala (Scaling Region)
    s_min = 2
    s_max = (N * N) / 4

    # -----------------------------------------------------------------
    # A) DISTRIBUCIÓN DE EXCEDENCIA (CCDF / P(S >= s))
    # -----------------------------------------------------------------
    s_ordenados = np.sort(s_datos)
    n_avalanchas = len(s_ordenados)
    ccdf_empirica = 1.0 - (np.arange(n_avalanchas) / n_avalanchas)

    s_unicos, indices = np.unique(s_ordenados, return_index=True)
    ccdf_unica = ccdf_empirica[indices]

    mascara_ccdf = (s_unicos >= s_min) & (s_unicos <= s_max)
    log_x_ccdf = np.log10(s_unicos[mascara_ccdf])
    log_y_ccdf = np.log10(ccdf_unica[mascara_ccdf])

    p_ccdf, int_ccdf = np.polyfit(log_x_ccdf, log_y_ccdf, 1)
    beta = -p_ccdf           
    tau_desde_ccdf = beta + 1 

    # -----------------------------------------------------------------
    # B) DISTRIBUCIÓN MARGINAL (PDF / P(S))
    # -----------------------------------------------------------------
    num_bins = 20
    bins_log = np.logspace(0, np.log10(max(s_datos)), num_bins)
    conteos, bordes_bins = np.histogram(s_datos, bins=bins_log)
    centros_bins = np.sqrt(bordes_bins[:-1] * bordes_bins[1:])
    anchos_bins = np.diff(bordes_bins)
    
    pdf = conteos / (sum(conteos) * anchos_bins)
    mascara_pdf = pdf > 0
    
    x_pdf = centros_bins[mascara_pdf]
    y_pdf = pdf[mascara_pdf]

    mascara_ajuste_pdf = (x_pdf >= s_min) & (x_pdf <= s_max)
    p_pdf, int_pdf = np.polyfit(np.log10(x_pdf[mascara_ajuste_pdf]), 
                                np.log10(y_pdf[mascara_ajuste_pdf]), 1)
    tau_pdf = -p_pdf

    # -----------------------------------------------------------------
    # C) GRÁFICO COMPARATIVO
    # -----------------------------------------------------------------
    fig, (ax_pdf, ax_ccdf) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Estudio de Avalanchas: Distribución Marginal vs. Excedencia', 
                 fontsize=13, fontweight='bold')

    # Panel Izquierdo: PDF Marginal
    ax_pdf.scatter(x_pdf, y_pdf, color='teal', edgecolor='k', s=35, alpha=0.8, label='Datos (Binned PDF)')
    x_linea = np.logspace(np.log10(s_min), np.log10(s_max), 100)
    y_linea_pdf = 10**int_pdf * x_linea**(-tau_pdf)
    ax_pdf.plot(x_linea, y_linea_pdf, 'r--', lw=2, label=fr'Ajuste PDF ($\tau \approx {tau_pdf:.2f}$)')
    
    ax_pdf.set_xscale('log')
    ax_pdf.set_yscale('log')
    ax_pdf.set_xlabel('Tamaño de Avalancha ($S$)', fontsize=11)
    ax_pdf.set_ylabel(r'Densidad de Probabilidad $P(S)$', fontsize=11)
    ax_pdf.set_title(r'Distribución Marginal $P(S)$', fontsize=12)
    ax_pdf.grid(True, which="both", linestyle=':', alpha=0.5)
    ax_pdf.legend(fontsize=10)

    # Panel Derecho: CCDF Excedencia (Corregidos comandos LaTeX)
    ax_ccdf.scatter(s_unicos, ccdf_unica, color='darkorange', s=15, alpha=0.6, label='Empírica CCDF')
    y_linea_ccdf = 10**int_ccdf * x_linea**(-beta)
    ax_ccdf.plot(x_linea, y_linea_ccdf, 'navy', lw=2, linestyle='--', 
                 label=fr'Ajuste CCDF ($\beta = {beta:.2f} \Rightarrow \tau = {tau_desde_ccdf:.2f}$)')
    
    ax_ccdf.set_xscale('log')
    ax_ccdf.set_yscale('log')
    ax_ccdf.set_xlabel('Tamaño de Avalancha ($S$)', fontsize=11)
    ax_ccdf.set_ylabel(r'Probabilidad de Excedencia $P(S \geq s)$', fontsize=11)
    ax_ccdf.set_title(r'Distribución de Excedencia (CCDF)', fontsize=12)
    ax_ccdf.grid(True, which="both", linestyle=':', alpha=0.5)
    ax_ccdf.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def calcular_tau_mle(datos_avalanchas, s_min, s_max=None):
    """
    Método 1: MÁXIMA VEROSIMILITUD (MLE / Estimador de Aki-Utsu)
    Es el método estadísticamente más riguroso (libre de sesgos por binning).
    """
    datos = np.array(datos_avalanchas)
    
    # Filtrar datos en la región de escala
    if s_max is not None:
        muestra = datos[(datos >= s_min) & (datos <= s_max)]
    else:
        muestra = datos[datos >= s_min]
        
    n = len(muestra)
    if n == 0:
        return None, None
    
    # Corrección continua para datos discretos (s_min - 0.5)
    suma_log = np.sum(np.log(muestra / (s_min - 0.5)))
    tau_mle = 1.0 + n / suma_log
    
    # Error estándar de la estimación
    error_mle = (tau_mle - 1.0) / np.sqrt(n)
    
    return tau_mle, error_mle


def calcular_tau_ols_pdf(datos_avalanchas, s_min, s_max, num_bins=20):
    """
    Método 2: MÍNIMOS CUADRADOS EN HISTOGRAMA LOGARÍTMICO (PDF)
    Ajuste lineal directo sobre log10(P(S)) vs log10(S).
    """
    datos = np.array(datos_avalanchas)
    bins_log = np.logspace(np.log10(min(datos)), np.log10(max(datos)), num_bins)
    
    conteos, bordes = np.histogram(datos, bins=bins_log)
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    anchos = np.diff(bordes)
    
    pdf = conteos / (sum(conteos) * anchos)
    
    # Máscara para la región de escala
    mascara = (centros >= s_min) & (centros <= s_max) & (pdf > 0)
    
    log_x = np.log10(centros[mascara])
    log_y = np.log10(pdf[mascara])
    
    # Ajuste lineal y_log = -tau * x_log + C
    pendiente, interseccion = np.polyfit(log_x, log_y, 1)
    tau_pdf = -pendiente
    
    return tau_pdf, centros[mascara], pdf[mascara], interseccion


def calcular_tau_ols_ccdf(datos_avalanchas, s_min, s_max):
    """
    Método 3: MÍNIMOS CUADRADOS EN DISTRIBUCIÓN DE EXCEDENCIA (CCDF)
    Ajuste sobre P(S >= s). Recupera tau mediante: tau = beta + 1.
    """
    datos = np.sort(np.array(datos_avalanchas))
    n = len(datos)
    ccdf = 1.0 - (np.arange(n) / n)
    
    s_unicos, idx = np.unique(datos, return_index=True)
    ccdf_unica = ccdf[idx]
    
    mascara = (s_unicos >= s_min) & (s_unicos <= s_max)
    
    log_x = np.log10(s_unicos[mascara])
    log_y = np.log10(ccdf_unica[mascara])
    
    pendiente, interseccion = np.polyfit(log_x, log_y, 1)
    beta = -pendiente
    tau_ccdf = beta + 1.0
    
    return tau_ccdf, beta, s_unicos[mascara], ccdf_unica[mascara]


# =====================================================================
# EJECUCIÓN Y EXTRACCIÓN CON TUS DATOS DE SIMULACIÓN
# =====================================================================

# Parámetros para la zona de escala limpia
s_min_ajuste = 2
s_max_ajuste = (N * N) / 4   # Límite superior por tamaño finito (225 en red 30x30)

# 1. Obtener valores de tau
tau_mle, error_mle = calcular_tau_mle(avalanchas_filtradas, s_min_ajuste, s_max_ajuste)
tau_pdf, x_puntos, y_puntos, int_pdf = calcular_tau_ols_pdf(avalanchas_filtradas, s_min_ajuste, s_max_ajuste)
tau_ccdf, beta_ccdf, x_ccdf, y_ccdf = calcular_tau_ols_ccdf(avalanchas_filtradas, s_min_ajuste, s_max_ajuste)

# 2. Mostrar reporte por consola
print("\n" + "="*55)
print("     REPORTE DE OBTENCIÓN DEL EXPONENTE CRÍTICO (tau)")
print("="*55)
print(f"Rango de ajuste empleado: S_min = {s_min_ajuste}, S_max = {s_max_ajuste:.1f}")
print("-" * 55)
print(f"1. Máxima Verosimilitud (MLE / Aki-Utsu) : tau = {tau_mle:.3f} +/- {error_mle:.3f}")
print(f"2. Mínimos Cuadrados en PDF Marginal     : tau = {tau_pdf:.3f}")
print(f"3. Mínimos Cuadrados en Excedencia (CCDF): tau = {tau_ccdf:.3f}  (beta = {beta_ccdf:.3f})")
print("="*55 + "\n")

# =====================================================================
# REPRESENTACIÓN GRÁFICA COMPARATIVA
# =====================================================================

x_linea = np.logspace(np.log10(s_min_ajuste), np.log10(s_max_ajuste), 100)

plt.figure(figsize=(8, 6))

# Puntos empíricos (PDF)
plt.scatter(x_puntos, y_puntos, color='darkorange', edgecolor='black', s=60, 
            zorder=3, label='Datos ($P(S)$ binned)')

# Recta 1: Ajuste OLS PDF
plt.plot(x_linea, 10**int_pdf * x_linea**(-tau_pdf), color='navy', linestyle='--', lw=2,
         label=fr'Ajuste OLS PDF ($\tau = {tau_pdf:.2f}$)')

# Recta 2: Recta Teórica MLE trazada desde la misma constante
C_mle = 10**int_pdf
plt.plot(x_linea, C_mle * x_linea**(-tau_mle), color='crimson', linestyle='-', lw=2,
         label=fr'Estimador MLE ($\tau = {tau_mle:.2f} \pm {error_mle:.2f}$)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tamaño de Avalancha ($S$)', fontsize=11)
plt.ylabel(r'Densidad de Probabilidad $P(S)$', fontsize=11)
plt.title('Comparación de Métodos para la Obtención de $\tau$', fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()