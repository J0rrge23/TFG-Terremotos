import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

# ==========================================
# 1. DESCARGA DE DATOS REALES DESDE EL SCEDC (CALTECH)
# ==========================================
def descargar_datos_scedc(minlat=32.0, maxlat=37.0, minlon=-122.0, maxlon=-114.0, 
                           starttime="1984-01-01", endtime="2004-12-31", minmag=3.0):
    """
    Descarga el catálogo de sismos del SCEDC usando la interfaz web FDSNWS en formato texto.
    Límites por defecto basados en el artículo de Zaliapin et al. (2008).
    """
    base_url = "https://service.scedc.caltech.edu/fdsnws/event/1/query"
    params = f"?starttime={starttime}&endtime={endtime}&minlatitude={minlat}&maxlatitude={maxlat}" \
             f"&minlongitude={minlon}&maxlongitude={maxlon}&minmagnitude={minmag}&format=text"
    
    url = base_url + params
    print("Descargando datos desde el SCEDC (Caltech)...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        text_data = response.read().decode('utf-8')
    
    # Procesar las líneas devueltas por el servidor
    lines = [line for line in text_data.split('\n') if line and not line.startswith('#')]
    data = []
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 10:
            time_str = parts[1].strip()
            lat = float(parts[2])
            lon = float(parts[3])
            depth = float(parts[4])
            mag = float(parts[10]) if len(parts) > 10 else float(parts[9])
            data.append([time_str, lat, lon, depth, mag])
            
    df = pd.DataFrame(data, columns=['time_str', 'lat', 'lon', 'depth', 'mag'])
    df['time'] = pd.to_datetime(df['time_str'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Convertir tiempo transcurrido a años
    t0 = df['time'].iloc[0]
    df['t_years'] = (df['time'] - t0).dt.total_seconds() / (365.25 * 86400.0)
    
    # Convertir lat/lon a coordenadas cartesianas aproximadas en km
    lat0 = df['lat'].mean()
    df['x_km'] = (df['lon'] - df['lon'].mean()) * 111.320 * np.cos(np.radians(lat0))
    df['y_km'] = (df['lat'] - lat0) * 110.574
    
    print(f"Catálogo descargado con éxito: {len(df)} eventos encontrados.")
    return df

# ==========================================
# 2. CÁLCULO DE LA MÉTRICA DE ZALIAPIN (NEAREST NEIGHBOR)
# ==========================================
def calcular_zaliapin_scedc(df, b=1.0, d=1.6, m0=3.0):
    """
    Calcula la distancia al vecino más cercano (eta_star) y sus componentes (T_star, R_star)
    incorporando deltas mínimos para prevenir valores infinitos.
    """
    N = len(df)
    t = df['t_years'].to_numpy()
    x = df['x_km'].to_numpy()
    y = df['y_km'].to_numpy()
    mag = df['mag'].to_numpy()
    
    T_star = np.full(N, np.nan)
    R_star = np.full(N, np.nan)
    eta_star = np.full(N, np.nan)
    parent_idx = np.full(N, -1, dtype=int)
    
    print("Calculando distancias de vecinos más cercanos...")
    
    # Umbrales mínimos para evitar divisiones o logaritmos de cero
    eps_t = 1e-6 / (365.25 * 86400.0)  # Evitar dt = 0 (mínimo ~1 microsegundo)
    eps_r = 1e-4                       # Evitar dr = 0 (mínimo 10 cm)

    for j in range(1, N):
        dt = np.maximum(t[j] - t[:j], eps_t)
        dr = np.maximum(np.sqrt((x[j] - x[:j])**2 + (y[j] - y[:j])**2), eps_r)
        
        # Métrica normalizada de Zaliapin et al. (2008)
        mag_factor = 10 ** (-0.5 * b * (mag[:j] - m0))
        T_ij = dt * mag_factor
        R_ij = (dr ** d) * mag_factor
        eta_ij = T_ij * R_ij
        
        # Selección del vecino más cercano
        i_min = np.argmin(eta_ij)
        eta_star[j] = eta_ij[i_min]
        T_star[j] = T_ij[i_min]
        R_star[j] = R_ij[i_min]
        parent_idx[j] = i_min
        
    df['log10_eta'] = np.log10(eta_star)
    df['log10_T'] = np.log10(T_star)
    df['log10_R'] = np.log10(R_star)
    df['parent_idx'] = parent_idx
    
    return df

# ==========================================
# 3. EJECUCIÓN PRINCIPAL, FILTRADO Y GRÁFICOS
# ==========================================
if __name__ == "__main__":
    # 1. Descargar datos de California del Sur (m >= 3.0, 1984-2004)
    df_california = descargar_datos_scedc(
        minlat=32.0, maxlat=37.0, minlon=-122.0, maxlon=-114.0, 
        starttime="1984-01-01", endtime="2004-12-31", minmag=3.0
    )

    # 2. Calcular métrica de Zaliapin
    df_california = calcular_zaliapin_scedc(df_california, b=1.0, d=1.6, m0=3.0)

    # 3. Filtrar valores estrictamente finitos (-inf, inf, NaN)
    mask_finitos = (
        np.isfinite(df_california['log10_eta']) & 
        np.isfinite(df_california['log10_T']) & 
        np.isfinite(df_california['log10_R'])
    )

    log10_eta = df_california.loc[mask_finitos, 'log10_eta']
    log10_T = df_california.loc[mask_finitos, 'log10_T']
    log10_R = df_california.loc[mask_finitos, 'log10_R']

    # 4. Generar figuras
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Histograma Bimodal de log10(eta*)
    axes[0].hist(log10_eta, bins=50, color='gray', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=-5.0, color='red', linestyle='--', label=r'Umbral ($\eta_0 = 10^{-5}$)')
    axes[0].set_xlabel(r'Distancia al vecino más cercano, $\log_{10}(\eta^*)$', fontsize=12)
    axes[0].set_ylabel('Número de observaciones', fontsize=12)
    axes[0].set_title('Distribución Bimodal en California del Sur', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Mapa de Densidad 2D (T, R)
    counts, xedges, yedges, im = axes[1].hist2d(log10_T, log10_R, bins=60, cmap='hot_r')
    plt.colorbar(im, ax=axes[1], label='Densidad de pares')
    axes[1].plot([-9, 0], [4, -5], 'k--', label=r'Diagonal Poisson ($\log_{10} T + \log_{10} R = C$)')
    axes[1].set_xlabel(r'Tiempo rescalado, $\log_{10}(T^*)$', fontsize=12)
    axes[1].set_ylabel(r'Distancia rescalada, $\log_{10}(R^*)$', fontsize=12)
    axes[1].set_title(r'Componentes Espacio-Tiempo $(T^*, R^*)$', fontsize=13)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
```[cite: 10]