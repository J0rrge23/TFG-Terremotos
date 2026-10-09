import numpy as np

def calcular_zaliapin_nn(df, b=1.0, d=1.6, m0=2.0, max_days_back=None):
    """
    Calcula la métrica de vecino más cercano de Zaliapin (Zaliapin et al., 2008).
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        Catálogo sísmico con columnas 'datetime', 'mag', 'lat', 'lon'.
    b : float
        Valor b de la ley Gutenberg-Richter del catálogo.
    d : float
        Dimensión fractal espacial (por defecto 1.6).
    m0 : float
        Magnitud mínima / completitud (por defecto 2.0).
    max_days_back : float o None
        Si es None, evalúa todos los eventos anteriores (metodología original).
        
    Retorna:
    --------
    pandas.DataFrame con columnas 'parent_idx', 'log10_T', 'log10_R', 'log10_eta'.
    """
    df = df.sort_values('datetime').reset_index(drop=True)
    
    times = np.asarray((df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 86400.0, dtype=float)
    mags = np.asarray(df['mag'].values, dtype=float)
    lats = np.asarray(df['lat'].values, dtype=float)
    lons = np.asarray(df['lon'].values, dtype=float)
    N = len(df)

    # Coordenadas cartesianas 3D sobre la esfera terrestre
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    X = np.cos(lat_rad) * np.cos(lon_rad)
    Y = np.cos(lat_rad) * np.sin(lon_rad)
    Z = np.sin(lat_rad)

    mag_factor = 10.0 ** (-b * (mags - m0))

    log10_T = np.full(N, np.nan)
    log10_R = np.full(N, np.nan)
    log10_eta = np.full(N, np.nan)
    parent_idx = np.full(N, -1, dtype=int)

    R_earth = 6371.0  # Radio terrestre en km

    for i in range(1, N):
        t_i = times[i]
        
        if max_days_back is not None:
            j_start = np.searchsorted(times, t_i - max_days_back, side='left')
        else:
            j_start = 0
            
        j_end = i

        if j_start >= j_end:
            continue

        dt = np.maximum(t_i - times[j_start:j_end], 1e-6)

        # Distancia 3D y gran círculo
        dx = X[i] - X[j_start:j_end]
        dy = Y[i] - Y[j_start:j_end]
        dz = Z[i] - Z[j_start:j_end]
        chord = np.sqrt(np.maximum(dx*dx + dy*dy + dz*dz, 0.0))
        r = np.maximum(R_earth * 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)), 1e-3)

        # Proximidad espacio-temporal eta = dt * r^d * 10^(-b*(m_j - m0))
        eta = np.asarray(dt * (r ** d) * mag_factor[j_start:j_end])

        min_rel_j = np.argmin(eta)
        best_j = j_start + min_rel_j

        best_eta = eta[min_rel_j]
        best_dt = dt[min_rel_j]
        best_r = r[min_rel_j]
        best_mag_j = mags[best_j]

        # Componentes reescaladas T (tiempo) y R (distancia)
        mag_factor_half = 10.0 ** (-0.5 * b * (best_mag_j - m0))
        best_T = best_dt * mag_factor_half
        best_R = (best_r ** d) * mag_factor_half

        parent_idx[i] = best_j
        log10_eta[i] = np.log10(best_eta)
        log10_T[i] = np.log10(best_T)
        log10_R[i] = np.log10(best_R)

    df['parent_idx'] = parent_idx
    df['log10_T'] = log10_T
    df['log10_R'] = log10_R
    df['log10_eta'] = log10_eta
    return df