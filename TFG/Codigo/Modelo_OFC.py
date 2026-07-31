# =====================================================================
# MODELO OFC — ESTUDIO COMPLETO: G-R, ESTRÉS, MOLCHAN Y LEY DE OMORI
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

# =====================================================================
# 1. CONFIGURACIÓN DE LA SIMULACIÓN
# =====================================================================
N = 40                 # Cuadrícula de 40x40 (1600 celdas)
pasos = 30000          # Número de sismos
F_th = 1.0             # Umbral de ruptura
alpha = 0.21           # Parámetro de disipación/elasticidad (alpha < 0.25)

cuadricula = np.random.uniform(0, F_th, (N, N))
lista_tamaños_terremotos = []
lista_estres_medio = []

print("Simulando modelo OFC y midiendo dinámica de estrés...")

# =====================================================================
# 2. BUCLE PRINCIPAL DE SIMULACIÓN
# =====================================================================
for t in range(pasos):
    # Carga tectónica cuasi-estática (Driving)
    F_max = np.max(cuadricula)
    delta_F = F_th - F_max
    cuadricula += delta_F
    
    # Estrés medio antes de la ruptura
    lista_estres_medio.append(np.mean(cuadricula))
    
    # Celdas inestables
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
            
            # Transferencia de tensión disipativa
            estres_transferido = alpha * estres_acumulado
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N:
                    cuadricula[nx, ny] += estres_transferido
                    if cuadricula[nx, ny] >= F_th:
                        pila_inestables.append((nx, ny))
                        
    lista_tamaños_terremotos.append(terremoto_tamaño)

print("Simulación completada. Procesando métricas...\n")

# =====================================================================
# 3. FILTRADO Y MÁXIMA VEROSIMILITUD (MLE)
# =====================================================================
transitorio = int(pasos * 0.3)
sismos_estacionarios = lista_tamaños_terremotos[transitorio:]
estres_estacionario = lista_estres_medio[transitorio:]
sismos_filtrados = [s for s in sismos_estacionarios if s > 0]

# Ajuste MLE
ajuste = powerlaw.Fit(sismos_filtrados, discrete=True, xmin=3)
tau_mle = ajuste.power_law.alpha 
sigma_tau = ajuste.power_law.sigma
beta_teorico = tau_mle - 1.0  # Valor b equivalente de Gutenberg-Richter

print("="*55)
print("     RESULTADOS DEL MODELO OFC (alpha = 0.21)")
print("="*55)
print(f"Exponente PDF (tau)      : {tau_mle:.3f} +/- {sigma_tau:.3f}")
print(f"Valor b de G-R (beta)    : {beta_teorico:.3f} (b = tau - 1)")
print(f"Límite inferior (xmin)   : {ajuste.xmin}")
print("="*55 + "\n")

# =====================================================================
# 4. FIGURA 1: LEY DE GUTENBERG-RICHTER Y CCDF
# =====================================================================
sismos_ordenados = np.sort(sismos_filtrados)
n_total = len(sismos_ordenados)
ccdf_empirica = 1.0 - (np.arange(n_total) / n_total)

magnitudes = np.log10(sismos_ordenados)
conteo_acumulado = n_total * ccdf_empirica

fig, (ax_gr, ax_ccdf) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(fr'Ley de Gutenberg-Richter y Probabilidad de Excedencia (OFC {N}x{N}, $\alpha={alpha}$)', 
             fontsize=13, fontweight='bold')

# Panel 1: Gutenberg-Richter
ax_gr.scatter(magnitudes, np.log10(conteo_acumulado), color='firebrick', s=12, alpha=0.6, label='Sismos Simulación')
mascara_zona = sismos_ordenados >= ajuste.xmin
m_zona = magnitudes[mascara_zona]
m_ref = np.linspace(min(m_zona), max(m_zona), 100)
log_N_ref = np.log10(conteo_acumulado[mascara_zona][0]) - beta_teorico * (m_ref - min(m_zona))

ax_gr.plot(m_ref, log_N_ref, color='navy', linestyle='--', lw=2.5, 
            label=f'Ajuste G-R ($b = {beta_teorico:.2f}$)')

ax_gr.set_xlabel(r'Magnitud Equivalente $M = \log_{10}(S)$', fontsize=11)
ax_gr.set_ylabel(r'$\log_{10} N(\geq M)$', fontsize=11)
ax_gr.set_title(r'1. Ley de Gutenberg-Richter $\log_{10}N = a - bM$', fontsize=11)
ax_gr.grid(True, linestyle=':', alpha=0.6)
ax_gr.legend(fontsize=10)

# Panel 2: CCDF en escala log-log
ax_ccdf.plot(sismos_ordenados, ccdf_empirica, color='darkorange', lw=2, label=r'CCDF Empírica $P(S \geq s)$')
s_ref = np.logspace(np.log10(ajuste.xmin), np.log10(max(sismos_ordenados)), 100)
ccdf_ref = (s_ref / ajuste.xmin)**(-beta_teorico) * ccdf_empirica[mascara_zona][0]
ax_ccdf.plot(s_ref, ccdf_ref, color='navy', linestyle='--', lw=2, label=fr'Exponente $\beta = \tau - 1 = {beta_teorico:.2f}$')

ax_ccdf.set_xscale('log')
ax_ccdf.set_yscale('log')
ax_ccdf.set_xlabel(r'Tamaño del Terremoto ($S$)', fontsize=11)
ax_ccdf.set_ylabel(r'Probabilidad Acumulada $P(S \geq s)$', fontsize=11)
ax_ccdf.set_title('2. Distribución Acumulada (CCDF)', fontsize=11)
ax_ccdf.grid(True, which="both", linestyle=':', alpha=0.5)
ax_ccdf.legend(fontsize=10)

plt.tight_layout()
plt.show()

# =====================================================================
# 5. FIGURA 2: PROPIEDADES DINÁMICAS (MOLCHAN Y LEY DE OMORI)
# =====================================================================
sismos_arr = np.array(sismos_estacionarios)
estres_arr = np.array(estres_estacionario)
T_est = len(sismos_arr)

# --- 1. DIAGRAMA DE MOLCHAN ---
umbral_sismo = np.percentile(sismos_arr[sismos_arr > 0], 90)
eventos_objetivo = sismos_arr >= umbral_sismo
umbrales_estres = np.linspace(min(estres_arr), max(estres_arr), 100)

tau_alarma, nu_fallos = [], []
for u in umbrales_estres:
    alerta = estres_arr > u
    volumen_alarma = np.sum(alerta) / T_est
    eventos_totales = np.sum(eventos_objetivo)
    
    tasa_fallos = 1.0 - (np.sum(eventos_objetivo & alerta) / eventos_totales) if eventos_totales > 0 else 1.0
    tau_alarma.append(volumen_alarma)
    nu_fallos.append(tasa_fallos)

# --- 2. ANÁLISIS DE OMORI (ACTIVIDAD POST-MAINSHOCK) ---
umbral_mainshock = np.percentile(sismos_arr, 98)
indices_mainshocks = np.where(sismos_arr >= umbral_mainshock)[0]

ventana_tiempo = 50
actividad_post = np.zeros(ventana_tiempo)
conteo_mainshocks = 0

for idx in indices_mainshocks:
    if idx + ventana_tiempo < T_est:
        actividad_post += sismos_arr[idx + 1 : idx + 1 + ventana_tiempo]
        conteo_mainshocks += 1

tasa_replicas_media = actividad_post / conteo_mainshocks if conteo_mainshocks > 0 else np.zeros(ventana_tiempo)

# --- REPRESENTACIÓN GRÁFICA COMBINADA ---
fig, (ax_mol, ax_omo) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(fr'Verificación Dinámica en OFC (Predictibilidad y Réplicas, $\alpha={alpha}$)', 
             fontsize=13, fontweight='bold')

# Molchan
ax_mol.plot([0, 1], [1, 0], 'k--', lw=1.5, label='Línea de Azar')
ax_mol.plot(tau_alarma, nu_fallos, color='darkgreen', lw=2.5, label='Modelo OFC')
ax_mol.set_xlabel(r'Volumen de Alarma ($\tau_a$)', fontsize=11)
ax_mol.set_ylabel(r'Tasa de Fallos ($\nu$)', fontsize=11)
ax_mol.set_title('Diagrama de Error de Molchan', fontsize=11)
ax_mol.grid(True, linestyle=':', alpha=0.6)
ax_mol.legend(fontsize=10)

# Omori
dt_eje = np.arange(1, ventana_tiempo + 1)
ax_omo.plot(dt_eje, tasa_replicas_media, color='purple', marker='o', ms=4, lw=1.5, label='Respuesta Post-Mainshock')
ax_omo.axhline(np.mean(sismos_arr), color='gray', linestyle='--', label='Ruido de Fondo Medio')
ax_omo.set_xlabel(r'Tiempo tras el sismo principal ($\Delta t$)', fontsize=11)
ax_omo.set_ylabel('Tamaño medio de réplica', fontsize=11)
ax_omo.set_title('Búsqueda de Réplicas (Ley de Omori)', fontsize=11)
ax_omo.grid(True, linestyle=':', alpha=0.6)
ax_omo.legend(fontsize=10)

plt.tight_layout()
plt.show()