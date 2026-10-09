import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_gutenberg_richter(mags, Mc=2.0, bin_width=0.1):
    """
    Ajusta la Ley Gutenberg-Richter por Máxima Verosimilitud (Aki, 1965).
    log10 N(>=M) = a - b * M
    """
    mags_filtered = mags[mags >= Mc]
    if len(mags_filtered) == 0:
        return np.nan, np.nan
    m_mean = np.mean(mags_filtered)
    b = np.log10(np.e) / (m_mean - (Mc - bin_width / 2.0))
    a = np.log10(len(mags_filtered)) + b * Mc
    return a, b

def omori_law(t, K, c, p):
    """Ley Modificada de Omori: n(t) = K / (c + t)^p"""
    return K / ((t + c) ** p)

def fit_omori_single_cluster(cluster_df, mainshock_row, t_min_days=0.01, t_max_days=730.0):
    """
    Ajusta la Ley de Omori a las réplicas de un terremoto principal específico.
    
    Parámetros:
    -----------
    t_min_days : float (por defecto 0.01 días ≈ 14.4 min)
        Elimina el periodo inicial donde el catálogo pierde eventos pequeños.
    t_max_days : float (por defecto 730 días = 2 años)
        Ventana máxima de 2 años tras el evento principal.
    """
    t_main = mainshock_row['datetime']
    aftershocks = cluster_df[(cluster_df['event_type'] == 'aftershock') & 
                             (cluster_df['datetime'] > t_main)].copy()
    
    dt_days = (aftershocks['datetime'] - t_main).dt.total_seconds() / 86400.0
    dt_days = dt_days[(dt_days >= t_min_days) & (dt_days <= t_max_days)].values

    if len(dt_days) < 8:
        raise ValueError(f"Insuficientes réplicas ({len(dt_days)}) en la ventana de tiempo.")

    bins = np.logspace(np.log10(dt_days.min()), np.log10(dt_days.max()), num=18)
    counts, edges = np.histogram(dt_days, bins=bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    bin_widths = np.diff(edges)

    rates = counts / bin_widths
    valid_bins = rates > 0
    t_data = bin_centers[valid_bins]
    rate_data = rates[valid_bins]

    if len(t_data) < 3:
        raise ValueError("Insuficientes intervalos de datos para el ajuste.")

    p0 = [len(dt_days) * 0.1, 0.01, 1.0]
    bounds = ([0, 1e-6, 0.1], [np.inf, 10.0, 3.0])
    popt, _ = curve_fit(omori_law, t_data, rate_data, p0=p0, bounds=bounds)

    return t_data, rate_data, popt

def graficar_gutenberg_richter(mags, Mc=2.0, bin_width=0.1, titulo="Ley Gutenberg-Richter", filename=None):
    """Grafica la frecuencia acumulada y discreta junto a la recta de ajuste."""
    a, b = fit_gutenberg_richter(mags, Mc=Mc, bin_width=bin_width)
    if np.isnan(b):
        return a, b

    m_min_data = np.floor(mags.min() / bin_width) * bin_width
    m_max_data = np.ceil(mags.max() / bin_width) * bin_width
    bins = np.arange(m_min_data, m_max_data + bin_width, bin_width)
    
    counts, edges = np.histogram(mags, bins=bins)
    m_centers = (edges[:-1] + edges[1:]) / 2.0
    
    cum_counts = np.array([np.sum(mags >= m) for m in m_centers])
    
    plt.figure(figsize=(7, 5))
    
    valid_cum = cum_counts > 0
    plt.semilogy(m_centers[valid_cum], cum_counts[valid_cum], 'ko', label='Acumulado N(>= M)', markersize=5)
    
    valid_counts = counts > 0
    plt.semilogy(m_centers[valid_counts], counts[valid_counts], 's', color='gray', alpha=0.4, label='Discreto n(M)', markersize=4)
    
    m_fit = np.linspace(Mc, mags.max(), 100)
    N_fit = 10 ** (a - b * m_fit)
    plt.semilogy(m_fit, N_fit, 'r-', linewidth=2, label=f'Ajuste M >= {Mc} (b={b:.2f})')
    
    plt.axvline(x=Mc, color='blue', linestyle='--', alpha=0.7, label=f'Mc = {Mc}')
    
    plt.xlabel("Magnitud (M)")
    plt.ylabel("Número de eventos (N)")
    plt.title(f"{titulo}\n(a = {a:.2f}, b = {b:.2f})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()
    return a, b

def graficar_comparativa_gr(mags_bg, mags_af, Mc=2.0, bin_width=0.1, filename="gr_fondo_vs_replicas.png"):
    """Grafica en un solo panel la comparativa de Gutenberg-Richter entre Fondo y Réplicas."""
    a_bg, b_bg = fit_gutenberg_richter(mags_bg, Mc=Mc, bin_width=bin_width)
    a_af, b_af = fit_gutenberg_richter(mags_af, Mc=Mc, bin_width=bin_width)
    
    plt.figure(figsize=(8, 5))
    
    for mags, a, b, label, color in [(mags_bg, a_bg, b_bg, 'Fondo (Background)', 'blue'),
                                     (mags_af, a_af, b_af, 'Réplicas (Aftershocks)', 'red')]:
        if len(mags) == 0 or np.isnan(b):
            continue
        m_centers = np.arange(Mc, mags.max() + bin_width, bin_width)
        cum_counts = np.array([np.sum(mags >= m) for m in m_centers])
        
        valid = cum_counts > 0
        plt.semilogy(m_centers[valid], cum_counts[valid], 'o', color=color, alpha=0.6, label=f'{label} (b={b:.2f})')
        
        m_fit = np.linspace(Mc, mags.max(), 100)
        N_fit = 10 ** (a - b * m_fit)
        plt.semilogy(m_fit, N_fit, '-', color=color, linewidth=2)

    plt.axvline(x=Mc, color='black', linestyle='--', alpha=0.5, label=f'Mc = {Mc}')
    plt.xlabel("Magnitud (M)")
    plt.ylabel("Número acumulado de eventos (N)")
    plt.title("Comparativa Gutenberg-Richter: Fondo vs. Réplicas")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()