import numpy as np
from scipy.optimize import curve_fit

def fit_gutenberg_richter(mags, Mc=2.5, bin_width=0.1):
    """Calcula 'a' y 'b' por máxima verosimilitud."""
    mags_filtered = mags[mags >= Mc]
    if len(mags_filtered) == 0:
        return np.nan, np.nan
    m_mean = np.mean(mags_filtered)
    b = np.log10(np.e) / (m_mean - (Mc - bin_width / 2.0))
    a = np.log10(len(mags_filtered)) + b * Mc
    return a, b

def omori_law(t, K, c, p):
    return K / ((t + c) ** p)

def fit_omori(aftershocks, df):
    """Ajuste vectorizado de la Ley de Omori."""
    valid = aftershocks['parent_idx'] >= 0
    aft_valid = aftershocks[valid]
    p_indices = aft_valid['parent_idx'].astype(int).values
    
    t_after = aft_valid['datetime'].values
    t_parent = df.loc[p_indices, 'datetime'].values
    
    # Diferencia de tiempos vectorizada
    dt_days = (t_after - t_parent).astype('timedelta64[ms]').astype(float) / 86400000.0
    dt_days = dt_days[dt_days > 0]

    if len(dt_days) < 5:
        raise ValueError("Sin suficientes datos de réplicas para el ajuste.")

    bins = np.logspace(np.log10(max(dt_days.min(), 1e-4)), np.log10(dt_days.max()), num=20)
    counts, edges = np.histogram(dt_days, bins=bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    bin_widths = np.diff(edges)

    rates = counts / bin_widths
    valid_bins = rates > 0
    t_data = bin_centers[valid_bins]
    rate_data = rates[valid_bins]

    p0 = [len(dt_days) * 0.1, 0.01, 1.0]
    bounds = ([0, 1e-6, 0.1], [np.inf, 10.0, 3.0])
    popt, _ = curve_fit(omori_law, t_data, rate_data, p0=p0, bounds=bounds)

    return t_data, rate_data, popt, None