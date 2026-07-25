#Codigo de la pila de arena de una cuadricula 30x30, donde incluyo una partícula en una de las casillas a cada paso de tiempo
#las variables son discretas y el tiempo es discreto, la simulación se realiza en un bucle for que recorre un número determinado de pasos de tiempo. En cada paso de tiempo, se selecciona una casilla aleatoria de la cuadricula y se incrementa el valor de esa casilla en 1, representando la adición de una partícula a esa posición.
#las condiciones de frontera son disipativas, es decir, si una casilla alcanza un valor mayor o igual a 4, se "desborda" y distribuye sus partículas a las casillas vecinas (arriba, abajo, izquierda, derecha). Si una casilla está en el borde de la cuadricula y se desborda, las partículas que se desbordan se pierden.

#MEdir la carga media de la cuadricula a lo largo del tiempo, es decir, el promedio de partículas por casilla en cada paso de tiempo. Esto se puede hacer sumando todos los valores de la cuadricula y dividiéndolos por el número total de casillas (900 en este caso). Almacenar estos valores en una lista para poder analizarlos posteriormente.
#Tambien quiero medir la criticidad de la cuadricula, es decir, el número de casillas que tienen un valor mayor o igual a 4 en cada paso de tiempo. Esto se puede hacer recorriendo la cuadricula y contando cuántas casillas cumplen esta condición. Almacenar estos valores en una lista para poder analizarlos posteriormente.
# =====================================================================
# MODELO PILA DE ARENA (BTW) — CÓDIGO COMPLETO Y CORREGIDO
# =====================================================================
import random
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. CONFIGURACIÓN DE LA SIMULACIÓN
# =====================================================================
N = 30                 # Tamaño de la cuadrícula (30x30 = 900 casillas)
pasos = 30000          # Pasos de tiempo
cuadricula = [[0 for _ in range(N)] for _ in range(N)]

# --- VARIABLES DE CONTROL OPTIMIZADAS O(1) ---
total_particulas = 0   
sitios_criticos = 0    

# --- LISTAS PARA ALMACENAR MÉTRICAS ---
lista_carga_media = []
lista_tamano_avalancha = []  # Nombre estandarizado sin 'ñ'
lista_sitios_criticos = []   

# =====================================================================
# 2. BUCLE PRINCIPAL DE TIEMPO
# =====================================================================
print("Simulando pila de arena...")
for t in range(pasos):
    x = random.randint(0, N - 1)
    y = random.randint(0, N - 1)
    
    val_antiguo = cuadricula[x][y]
    cuadricula[x][y] += 1
    total_particulas += 1
    
    if val_antiguo == 3:
        sitios_criticos -= 1
    elif cuadricula[x][y] == 3:
        sitios_criticos += 1
    
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
                    total_particulas -= 1

    carga_media = total_particulas / (N * N)
    lista_carga_media.append(carga_media)
    lista_tamano_avalancha.append(desbordes_este_paso)
    lista_sitios_criticos.append(sitios_criticos)

print(f"Simulación finalizada tras {pasos} pasos.\n")

# =====================================================================
# 3. FILTRADO DEL ESTADO ESTACIONARIO (SOC)
# =====================================================================
inicio_estacionario = int(pasos * 0.3)

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

ax1.plot(tiempo_interes, carga_interes, color='teal', lw=1, label='Carga Media Activa')
ax1.axhline(17/8, color='crimson', linestyle='--', lw=1.5, label=r'Límite Teórico $N \to \infty$ (2.125)')
ax1.set_ylabel('Carga Media\n(partículas / casilla)', fontsize=10)
ax1.set_title('1. Fluctuaciones de la Carga Media', fontsize=11, loc='left')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')
ax1.set_ylim(min(carga_interes) * 0.98, max(carga_interes) * 1.02)

ax2.plot(tiempo_interes, criticos_interes, color='darkorange', lw=0.8, alpha=0.8, label=r'Sitios Críticos ($h = 3$)')
ax2.set_xlabel('Paso de tiempo (t)', fontsize=11)
ax2.set_ylabel('Nº de Sitios Críticos', fontsize=10)
ax2.set_title('2. Densidad de Sitios Críticos', fontsize=11, loc='left')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# =====================================================================
# 5. FUNCIONES Y EXTRACCIÓN DE TAU (PDF, CCDF Y MLE)
# =====================================================================
def calcular_tau_mle(datos_avalanchas, s_min, s_max=None):
    datos = np.array(datos_avalanchas)
    muestra = datos[(datos >= s_min) & (datos <= s_max)] if s_max is not None else datos[datos >= s_min]
    n = len(muestra)
    if n == 0:
        return None, None
    suma_log = np.sum(np.log(muestra / (s_min - 0.5)))
    tau_mle = 1.0 + n / suma_log
    error_mle = (tau_mle - 1.0) / np.sqrt(n)
    return tau_mle, error_mle

def calcular_tau_ols_pdf(datos_avalanchas, s_min, s_max, num_bins=20):
    datos = np.array(datos_avalanchas)
    bins_log = np.logspace(np.log10(min(datos)), np.log10(max(datos)), num_bins)
    conteos, bordes = np.histogram(datos, bins=bins_log)
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    anchos = np.diff(bordes)
    pdf = conteos / (sum(conteos) * anchos)
    mascara = (centros >= s_min) & (centros <= s_max) & (pdf > 0)
    log_x = np.log10(centros[mascara])
    log_y = np.log10(pdf[mascara])
    pendiente, interseccion = np.polyfit(log_x, log_y, 1)
    return -pendiente, centros[mascara], pdf[mascara], interseccion

def calcular_tau_ols_ccdf(datos_avalanchas, s_min, s_max):
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
    return beta + 1.0, beta, s_unicos[mascara], ccdf_unica[mascara]

s_min_ajuste = 2
s_max_ajuste = (N * N) / 4

tau_mle, error_mle = calcular_tau_mle(avalanchas_filtradas, s_min_ajuste, s_max_ajuste)
tau_pdf, x_puntos, y_puntos, int_pdf = calcular_tau_ols_pdf(avalanchas_filtradas, s_min_ajuste, s_max_ajuste)
tau_ccdf, beta_ccdf, x_ccdf, y_ccdf = calcular_tau_ols_ccdf(avalanchas_filtradas, s_min_ajuste, s_max_ajuste)

print("\n" + "="*55)
print("     REPORTE DE OBTENCIÓN DEL EXPONENTE CRÍTICO (tau)")
print("="*55)
print(f"1. Máxima Verosimilitud (MLE) : tau = {tau_mle:.3f} +/- {error_mle:.3f}")
print(f"2. Mínimos Cuadrados en PDF   : tau = {tau_pdf:.3f}")
print(f"3. Mínimos Cuadrados en CCDF  : tau = {tau_ccdf:.3f}")
print("="*55 + "\n")

# =====================================================================
# 6. FIGURA 2: COMPARACIÓN DE MÉTODOS DE AJUSTE
# =====================================================================
x_linea = np.logspace(np.log10(s_min_ajuste), np.log10(s_max_ajuste), 100)

plt.figure(figsize=(8, 6))
plt.scatter(x_puntos, y_puntos, color='darkorange', edgecolor='black', s=60, zorder=3, label='Datos ($P(S)$ binned)')
plt.plot(x_linea, 10**int_pdf * x_linea**(-tau_pdf), color='navy', linestyle='--', lw=2, label=fr'Ajuste OLS PDF ($\tau = {tau_pdf:.2f}$)')
plt.plot(x_linea, 10**int_pdf * x_linea**(-tau_mle), color='crimson', linestyle='-', lw=2, label=fr'Estimador MLE ($\tau = {tau_mle:.2f} \pm {error_mle:.2f}$)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tamaño de Avalancha ($S$)', fontsize=11)
plt.ylabel(r'Densidad de Probabilidad $P(S)$', fontsize=11)
plt.title('Comparación de Métodos para la Obtención de $\tau$', fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# =====================================================================
# 7. COMPROBACIÓN DE IMPREDICTIBILIDAD (MOLCHAN) Y OMORI
# =====================================================================
avalanchas_arr = np.array(lista_tamano_avalancha[inicio_estacionario:])
criticos_arr = np.array(lista_sitios_criticos[inicio_estacionario:])
T_est = len(avalanchas_arr)

# --- DIAGRAMA DE MOLCHAN ---
umbral_evento = np.percentile(avalanchas_arr[avalanchas_arr > 0], 90)
eventos_objetivo = avalanchas_arr >= umbral_evento
umbrales_criticos = np.linspace(min(criticos_arr), max(criticos_arr), 100)

tau_alarma = []
nu_fallos = []

for u in umbrales_criticos:
    alerta = np.roll(criticos_arr > u, 1)
    alerta[0] = False
    
    volumen_alarma = np.sum(alerta) / T_est
    eventos_totales = np.sum(eventos_objetivo)
    
    if eventos_totales > 0:
        eventos_detectados = np.sum(eventos_objetivo & alerta)
        tasa_fallos = 1.0 - (eventos_detectados / eventos_totales)
    else:
        tasa_fallos = 1.0
        
    tau_alarma.append(volumen_alarma)
    nu_fallos.append(tasa_fallos)

# --- ANÁLISIS DE OMORI ---
umbral_mainshock = np.percentile(avalanchas_arr, 98)
indices_mainshocks = np.where(avalanchas_arr >= umbral_mainshock)[0]

ventana_tiempo = 50
actividad_post = np.zeros(ventana_tiempo)
conteo_mainshocks = 0

for idx in indices_mainshocks:
    if idx + ventana_tiempo < T_est:
        actividad_post += avalanchas_arr[idx + 1 : idx + 1 + ventana_tiempo]
        conteo_mainshocks += 1

tasa_replicas_media = actividad_post / conteo_mainshocks if conteo_mainshocks > 0 else np.zeros(ventana_tiempo)

# --- REPRESENTACIÓN GRÁFICA ---
fig, (ax_mol, ax_omo) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Verificación de Propiedades Dinámicas en la Pila de Arena (BTW)', fontsize=13, fontweight='bold')

ax_mol.plot([0, 1], [1, 0], 'k--', lw=1.5, label='Línea de Azar (Sin Predictibilidad)')
ax_mol.plot(tau_alarma, nu_fallos, color='crimson', lw=2.5, label='Modelo BTW (Pila de Arena)')
ax_mol.set_xlabel(r'Volumen de Alarma ($\tau$)', fontsize=11)
ax_mol.set_ylabel(r'Tasa de Fallos ($\nu$)', fontsize=11)
ax_mol.set_title('Diagrama de Error de Molchan', fontsize=12)
ax_mol.set_xlim(-0.02, 1.02)
ax_mol.set_ylim(-0.02, 1.02)
ax_mol.grid(True, linestyle=':', alpha=0.6)
ax_mol.legend(fontsize=10)

dt_eje = np.arange(1, ventana_tiempo + 1)
ax_omo.plot(dt_eje, tasa_replicas_media, color='teal', marker='o', ms=4, lw=1.5, label='Respuesta Post-Mainshock')
ax_omo.axhline(np.mean(avalanchas_arr), color='gray', linestyle='--', label='Nivel de Ruido de Fondo')
ax_omo.set_xlabel('Tiempo tras el evento principal ($\Delta t$)', fontsize=11)
ax_omo.set_ylabel('Tamaño medio de avalancha', fontsize=11)
ax_omo.set_title('Búsqueda de Réplicas (Ley de Omori)', fontsize=12)
ax_omo.grid(True, linestyle=':', alpha=0.6)
ax_omo.legend(fontsize=10)

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