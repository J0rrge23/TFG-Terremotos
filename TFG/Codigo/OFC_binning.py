# =====================================================================
# MODELO OFC: BUCLE POR TAMAÑO N CON 10 ALPHAS Y GRÁFICAS INDEPENDIENTES
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. FUNCIÓN DE SIMULACIÓN BASE
# ---------------------------------------------------------------------
def simular_ofc(N, alpha, pasos=12000, seed=42):
    np.random.seed(seed)
    F_th = 1.0
    cuadricula = np.random.uniform(0, F_th, (N, N))
    lista_tamaños = np.zeros(pasos, dtype=int)
    
    for t in range(pasos):
        F_max = np.max(cuadricula)
        cuadricula += (F_th - F_max)
        
        inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
        pila = list(zip(inestables_x, inestables_y))
        terremoto_tamaño = 0
        
        while pila:
            cx, cy = pila.pop()
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
                            pila.append((nx, ny))
                            
        lista_tamaños[t] = terremoto_tamaño
        
    return lista_tamaños

# ---------------------------------------------------------------------
# 2. PARÁMETROS GENERALES DE LA EXPLORACIÓN
# ---------------------------------------------------------------------
redes_N = [40, 80, 120]                   # Lista de tamaños N a evaluar
alphas_10 = np.linspace(0.05, 0.245, 10)  # 10 valores de alpha
pasos_sim = 12000
transitorio = int(pasos_sim * 0.3)
ventana_t = 30

# ---------------------------------------------------------------------
# 3. BUCLE PRINCIPAL: SIMULACIÓN Y GENERACIÓN DE GRÁFICAS POR CADA N
# ---------------------------------------------------------------------
for N in redes_N:
    print(f"\n==========================================")
    print(f"   EJECUTANDO SIMULACIONES PARA N = {N}")
    print(f"==========================================")
    
    resultados_alpha = {}
    matriz_omori = np.zeros((len(alphas_10), ventana_t))
    
    # Simulación de los 10 alphas para la red N actual
    for i, a in enumerate(alphas_10):
        s_raw = simular_ofc(N, a, pasos=pasos_sim)
        s_est = s_raw[transitorio:]
        s_filt = s_est[s_est > 0]
        resultados_alpha[a] = s_filt
        
        # Omori post-mainshock
        if len(s_est) > 0:
            umbral_main = np.percentile(s_est, 98)
            indices_main = np.where(s_est >= umbral_main)[0]
            actividad = np.zeros(ventana_t)
            conteo = 0
            for idx in indices_main:
                if idx + ventana_t < len(s_est):
                    actividad += s_est[idx + 1 : idx + 1 + ventana_t]
                    conteo += 1
            matriz_omori[i, :] = actividad / conteo if conteo > 0 else np.zeros(ventana_t)

    # -----------------------------------------------------------------
    # FIGURA A (Específica para N): BINNING LINEAL VS LOGARÍTMICO
    # -----------------------------------------------------------------
    # Tomamos un alpha representativo de alta elasticidad (cercano a 0.23)
    a_ref = alphas_10[-2]
    sismos_ref = resultados_alpha[a_ref]
    
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Tamaño de Red N = {N}x{N} | Comparación de Binning G-R (alpha = {a_ref:.3f})', 
                 fontsize=12, fontweight='bold')
    
    # Panel Lineal
    conteo_lin, bordes_lin = np.histogram(sismos_ref, bins=35)
    centros_lin = (bordes_lin[:-1] + bordes_lin[1:]) / 2
    m_lin = conteo_lin > 0
    ax_lin.scatter(centros_lin[m_lin], conteo_lin[m_lin], color='crimson', edgecolors='k', s=25)
    ax_lin.set_xscale('log')
    ax_lin.set_yscale('log')
    ax_lin.set_title('Binning Lineal (Ruido en Cola)', fontsize=11)
    ax_lin.set_xlabel('Tamaño de Sismo (S)')
    ax_lin.set_ylabel('Conteo N(S)')
    ax_lin.grid(True, which="both", linestyle=':', alpha=0.5)
    
    # Panel Logarítmico Normalizado
    bordes_log = np.logspace(np.log10(min(sismos_ref)), np.log10(max(sismos_ref)), 20)
    conteo_log, _ = np.histogram(sismos_ref, bins=bordes_log)
    delta_s = np.diff(bordes_log)
    pdf_log = conteo_log / (len(sismos_ref) * delta_s)
    centros_log = np.sqrt(bordes_log[:-1] * bordes_log[1:])
    m_log = pdf_log > 0
    ax_log.scatter(centros_log[m_log], pdf_log[m_log], color='teal', edgecolors='k', s=25)
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_title('Binning Logarítmico Normalizado P(S)', fontsize=11)
    ax_log.set_xlabel('Tamaño de Sismo (S)')
    ax_log.set_ylabel('Densidad P(S)')
    ax_log.grid(True, which="both", linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # FIGURA B (Específica para N): GUTENBERG-RICHTER Y OMORI PARA 10 ALPHAS
    # -----------------------------------------------------------------
    fig, (ax_gr, ax_omo) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Tamaño de Red N = {N}x{N} | Comparativa de 10 Alphas', 
                 fontsize=12, fontweight='bold')
    
    colores = plt.cm.inferno(np.linspace(0.15, 0.85, 10))
    
    # Panel G-R (CCDF)
    for i, a in enumerate(alphas_10):
        s_ord = np.sort(resultados_alpha[a])
        ccdf = 1.0 - (np.arange(len(s_ord)) / len(s_ord))
        ax_gr.plot(s_ord, ccdf, label=f'$\\alpha={a:.3f}$', color=colores[i], lw=1.5)
        
    ax_gr.set_xscale('log')
    ax_gr.set_yscale('log')
    ax_gr.set_title('Gutenberg-Richter Acumulada (CCDF)', fontsize=11)
    ax_gr.set_xlabel('Tamaño de Sismo (S)')
    ax_gr.set_ylabel(r'Probabilidad $P(S \geq s)$')
    ax_gr.grid(True, which="both", linestyle=':', alpha=0.5)
    ax_gr.legend(fontsize=8, loc='lower left')
    
    # Panel Ley de Omori
    for i, a in enumerate(alphas_10):
        ax_omo.plot(range(1, ventana_t + 1), matriz_omori[i, :], 
                    label=f'$\\alpha={a:.3f}$', color=colores[i], lw=1.5)
        
    ax_omo.set_yscale('log')
    ax_omo.set_title('Emergencia Ley de Omori Post-Mainshock', fontsize=11)
    ax_omo.set_xlabel(r'Tiempo tras sismo ($\Delta t$)')
    ax_omo.set_ylabel(r'Tamaño medio réplica $\langle S \rangle$')
    ax_omo.grid(True, which="both", linestyle=':', alpha=0.5)
    ax_omo.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.show()