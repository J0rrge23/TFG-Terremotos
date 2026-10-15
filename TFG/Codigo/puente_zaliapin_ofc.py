# =====================================================================
# PUENTE_OFC_ZALIAPIN.PY
# Experimento Puente: Métrica de Zaliapin sobre Avalanchas del Modelo OFC
# + Módulo de Incertidumbre y Bootstrap para el Valor b
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------
# 1. SIMULACIÓN OFC Y GENERACIÓN DE CATÁLOGO SINTÉTICO
# ---------------------------------------------------------------------
def generar_catalogo_sintetico_ofc(N=50, alpha=0.20, pasos=20000, seed=42):
    """
    Simula el modelo OFC y registra cada avalancha como un evento con:
    t (tiempo de paso), x, y (centro de masa del colapso) y M (magnitud equivalente).
    """
    np.random.seed(seed)
    F_th = 1.0
    cuadricula = np.random.uniform(0, F_th, (N, N))
    
    eventos = []
    
    for t in range(pasos):
        F_max = np.max(cuadricula)
        cuadricula += (F_th - F_max)  # Carga lenta
        
        inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
        pila = list(zip(inestables_x, inestables_y))
        
        if not pila:
            continue
            
        sitios_colapsados_x = []
        sitios_colapsados_y = []
        
        while pila:
            cx, cy = pila.pop()
            if cuadricula[cx, cy] >= F_th:
                estres_acumulado = cuadricula[cx, cy]
                cuadricula[cx, cy] = 0.0
                
                sitios_colapsados_x.append(cx)
                sitios_colapsados_y.append(cy)
                
                estres_transferido = alpha * estres_acumulado
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        cuadricula[nx, ny] += estres_transferido
                        if cuadricula[nx, ny] >= F_th:
                            pila.append((nx, ny))
                            
        tamaño_S = len(sitios_colapsados_x)
        if tamaño_S > 0:
            # Centro de masa espacial del terremoto sintético
            x_cm = np.mean(sitios_colapsados_x)
            y_cm = np.mean(sitios_colapsados_y)
            # Magnitud sintética M = log10(S) + M0
            mag = np.log10(tamaño_S) + 2.0
            eventos.append([t, x_cm, y_cm, mag, tamaño_S])
            
    # Filtro del régimen transitorio (primeros 30% de pasos)
    corte_transitorio = int(len(eventos) * 0.3)
    eventos_est = eventos[corte_transitorio:]
    
    data = np.array(eventos_est)
    return {
        't': data[:, 0],
        'x': data[:, 1],
        'y': data[:, 2],
        'mag': data[:, 3],
        'S': data[:, 4]
    }

# ---------------------------------------------------------------------
# 2. CÁLCULO DE LA MÉTRICA DE ZALIAPIN SOBRE EL CATÁLOGO SINTÉTICO
# ---------------------------------------------------------------------
def calcular_zaliapin_sintetico(cat, b=1.0, d=1.6, m0=2.0):
    """
    Calcula log10(eta) para el catálogo sintético generado por OFC.
    """
    N_ev = len(cat['t'])
    t = cat['t']
    x = cat['x']
    y = cat['y']
    mag = cat['mag']
    
    eta_star = np.full(N_ev, np.nan)
    T_star = np.full(N_ev, np.nan)
    R_star = np.full(N_ev, np.nan)
    
    eps_t = 1e-4
    eps_r = 1e-3
    
    for j in range(1, N_ev):
        dt = np.maximum(t[j] - t[:j], eps_t)
        dr = np.maximum(np.sqrt((x[j] - x[:j])**2 + (y[j] - y[:j])**2), eps_r)
        
        mag_factor = 10 ** (-0.5 * b * (mag[:j] - m0))
        T_ij = dt * mag_factor
        R_ij = (dr ** d) * mag_factor
        eta_ij = T_ij * R_ij
        
        i_min = np.argmin(eta_ij)
        eta_star[j] = eta_ij[i_min]
        T_star[j] = T_ij[i_min]
        R_star[j] = R_ij[i_min]
        
    mask = np.isfinite(eta_star) & (eta_star > 0)
    return {
        'log10_eta': np.log10(eta_star[mask]),
        'log10_T': np.log10(T_star[mask]),
        'log10_R': np.log10(R_star[mask])
    }

# ---------------------------------------------------------------------
# 3. MÓDULO DE RIGOR ESTADÍSTICO: INCERTIDUMBRE Y BOOTSTRAP
# ---------------------------------------------------------------------
def calcular_b_aki_con_error(mags, Mc=2.0, bin_width=0.1):
    """Calcula el valor b de Aki con su desviación estándar teórica sigma_b."""
    mags_filt = mags[mags >= Mc]
    N = len(mags_filt)
    if N == 0:
        return np.nan, np.nan
    m_mean = np.mean(mags_filt)
    b = np.log10(np.e) / (m_mean - (Mc - bin_width / 2.0))
    sigma_b = b / np.sqrt(N)
    return b, sigma_b

def bootstrap_b_value(mags, Mc=2.0, n_samples=1000, ci=95):
    """Estima intervalos de confianza al 95% para el valor b mediante Resampling."""
    mags_filt = mags[mags >= Mc]
    N = len(mags_filt)
    if N < 10:
        return np.nan, (np.nan, np.nan)
        
    b_boot = []
    for _ in range(n_samples):
        sample = np.random.choice(mags_filt, size=N, replace=True)
        m_mean = np.mean(sample)
        b_i = np.log10(np.e) / (m_mean - (Mc - 0.05))
        b_boot.append(b_i)
        
    lower_p = (100 - ci) / 2.0
    upper_p = 100 - lower_p
    ci_lower = np.percentile(b_boot, lower_p)
    ci_upper = np.percentile(b_boot, upper_p)
    
    return np.mean(b_boot), (ci_lower, ci_upper)

# ---------------------------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL Y GENERACIÓN DE FIGURAS PUBLICABLES
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=================================================================")
    print(" EJECUTANDO EXPERIMENTO PUENTE: MODELO OFC -> MÉTRICA ZALIAPIN")
    print("=================================================================")
    
    alphas_evaluar = [0.10, 0.18, 0.245]
    nombres_regimen = ['Disipativo (alpha=0.10)', 'Intermedio (alpha=0.18)', 'Semi-conservativo (alpha=0.245)']
    colores = ['teal', 'darkorange', 'crimson']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(r'Emergencia de Bimodalidad Espacio-Temporal en OFC según $\alpha$', fontsize=14, fontweight='bold')
    
    for idx, (alpha, nombre, color) in enumerate(zip(alphas_evaluar, nombres_regimen, colores)):
        print(f"Procesando régimen: {nombre}...")
        cat = generar_catalogo_sintetico_ofc(N=50, alpha=alpha, pasos=25000)
        res_zal = calcular_zaliapin_sintetico(cat, b=1.0, d=1.6, m0=2.0)
        
        # Histograma 1D de log10(eta)
        axes[idx].hist(res_zal['log10_eta'], bins=40, color=color, edgecolor='black', alpha=0.7, density=True)
        axes[idx].axvline(x=0.0, color='black', linestyle='--', label=r'Umbral $\log_{10}\eta_0 = 0$')
        axes[idx].set_xlabel(r'$\log_{10} \eta^*$', fontsize=11)
        axes[idx].set_title(nombre, fontsize=11, fontweight='bold')
        axes[idx].grid(True, linestyle=':', alpha=0.6)
        axes[idx].legend(loc='upper left')
        
        if idx == 0:
            axes[idx].set_ylabel('Densidad de Frecuencia', fontsize=11)
            
    plt.tight_layout()
    plt.savefig("experimento_puente_ofc_zaliapin.png", dpi=300)
    plt.show()
    
    print("\n--- EJEMPLO DE MÓDULO DE INCERTIDUMBRE (BOOTSTRAP) ---")
    np.random.seed(42)
    mags_sinteticas = np.random.exponential(scale=1.0/np.log(10), size=1500) + 2.0
    b_mle, sigma_b = calcular_b_aki_con_error(mags_sinteticas, Mc=2.0)
    b_mean_boot, (ci_low, ci_high) = bootstrap_b_value(mags_sinteticas, Mc=2.0)
    
    print(f"Valor b (Aki MLE) : {b_mle:.3f} +/- {sigma_b:.3f}")
    print(f"Valor b (Bootstrap): {b_mean_boot:.3f} (IC 95%: [{ci_low:.3f}, {ci_high:.3f}])")
    print("=================================================================")