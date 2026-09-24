import numpy as np

def calcular_zaliapin_nn(df, b=1.0, d=1.6, m0=2.0, max_days_back=365):
    """
    Versión ultraoptimizada de Zaliapin NN.
    Corregida para trabajar 100% con arreglos de NumPy y evitar conflictos de índices en Pandas.
    """
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # .values convierte la serie de tiempo a un arreglo NumPy puro
    times = ((df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 86400.0).values
    mags = df['mag'].values
    lats = df['lat'].values
    lons = df['lon'].values
    N = len(df)

    # Precalcular coordenadas cartesianas 3D en esfera unitaria
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    X = np.cos(lat_rad) * np.cos(lon_rad)
    Y = np.cos(lat_rad) * np.sin(lon_rad)
    Z = np.sin(lat_rad)

    # Precalcular factor de magnitud: 10^(-b*(m_j - m0))
    mag_factor = 10.0 ** (-b * (mags - m0))

    log10_T = np.full(N, np.nan)
    log10_R = np.full(N, np.nan)
    log10_eta = np.full(N, np.nan)
    parent_idx = np.full(N, -1, dtype=int)

    R_earth = 6371.0

    for i in range(1, N):
        t_i = times[i]
        
        # Búsqueda binaria O(log N) para filtrar ventana temporal
        j_start = np.searchsorted(times, t_i - max_days_back, side='left')
        j_end = i

        if j_start >= j_end:
            continue

        dt = np.maximum(t_i - times[j_start:j_end], 1e-6)

        # Distancia entre coordenadas 3D convertida a km
        dx = X[i] - X[j_start:j_end]
        dy = Y[i] - Y[j_start:j_end]
        dz = Z[i] - Z[j_start:j_end]
        chord = np.sqrt(np.maximum(dx*dx + dy*dy + dz*dz, 0.0))
        r = np.maximum(R_earth * 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)), 1e-3)

        # Proximidad espacio-temporal (todas las variables son arreglos NumPy)
        eta = dt * (r ** d) * mag_factor[j_start:j_end]

        min_rel_j = np.argmin(eta)
        best_j = j_start + min_rel_j

        best_eta = eta[min_rel_j]
        best_dt = dt[min_rel_j]
        best_r = r[min_rel_j]
        best_mag_j = mags[best_j]

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