import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIGURACIÓN DE ESTILO PROFESIONAL (ESTILO PUBLICATION / PRL)
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.1,
    'grid.color': '#d3d3d3',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.titlesize': 13
})


# =============================================================================
# 1. CARGA Y PARSEO COMPLETO DEL CATÁLOGO (TODOS LOS AÑOS)
# =============================================================================
def cargar_catalogo_scedc(filepath):
    """Parsea el archivo ASCII del catálogo SCEDC sin restringir fechas."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    date_str, time_str = parts[0], parts[1]
                    mag = float(parts[4])
                    lat = float(parts[6])
                    lon = float(parts[7])
                    dt = pd.to_datetime(f"{date_str} {time_str}")
                    data.append({'datetime': dt, 'lat': lat, 'lon': lon, 'mag': mag})
                except Exception:
                    continue
                    
    df = pd.DataFrame(data).dropna().sort_values('datetime').reset_index(drop=True)
    return df


# =============================================================================
# 2. CÁLCULO DE COMPONENTES REESCALADAS T Y R (Zaliapin et al., 2008)
# =============================================================================
def calcular_componentes_zaliapin(df, b=1.0, d=1.6, m0=3.0):
    """Calcula log10(T) y log10(R) para el conjunto completo de sismos con M >= m0."""
    df_sub = df[df['mag'] >= m0].sort_values('datetime').reset_index(drop=True)
    N = len(df_sub)
    if N < 2:
        return pd.DataFrame(), N

    # Tiempo en años transcurridos desde el origen del catálogo
    t_start = df_sub['datetime'].iloc[0]
    tau_years = (df_sub['datetime'] - t_start).dt.total_seconds().values / (86400.0 * 365.25)
    mags = df_sub['mag'].values
    lats = df_sub['lat'].values
    lons = df_sub['lon'].values

    # Coordenadas 3D cartesianas en esfera terrestre
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    X = np.cos(lat_rad) * np.cos(lon_rad)
    Y = np.cos(lat_rad) * np.sin(lon_rad)
    Z = np.sin(lat_rad)

    R_earth = 6371.0  # Radio de la Tierra en km
    mag_factor = 10.0 ** (-b * (mags - m0))

    log10_T = np.full(N, np.nan)
    log10_R = np.full(N, np.nan)

    for i in range(1, N):
        t_i = tau_years[i]
        dt = np.maximum(t_i - tau_years[:i], 1e-10)

        # Distancia sobre la superficie terrestre (gran círculo)
        dx = X[i] - X[:i]
        dy = Y[i] - Y[:i]
        dz = Z[i] - Z[:i]
        chord = np.sqrt(np.maximum(dx*dx + dy*dy + dz*dz, 0.0))
        r = np.maximum(R_earth * 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)), 1e-3)

        # Métrica espacio-tiempo eta
        eta = dt * (r ** d) * mag_factor[:i]

        min_j = np.argmin(eta)
        best_mag_j = mags[min_j]

        # Componentes reescaladas (Eq. 6 de Zaliapin et al., 2008)
        mag_factor_half = 10.0 ** (-0.5 * b * (best_mag_j - m0))
        best_T = dt[min_j] * mag_factor_half
        best_R = (r[min_j] ** d) * mag_factor_half

        log10_T[i] = np.log10(best_T)
        log10_R[i] = np.log10(best_R)

    df_res = pd.DataFrame({'log10_T': log10_T, 'log10_R': log10_R}).dropna()
    return df_res, N


# =============================================================================
# 3. RENDERIZADO DE DENSIDAD 2D CON CALIDAD EDITORIAL
# =============================================================================
def graficar_panel_densidad(ax, df_tr, titulo, n_events, niveles=14):
    """Genera curvas de nivel de densidad suavizadas (KDE Gaussiano)."""
    x = df_tr['log10_T'].values
    y = df_tr['log10_R'].values

    # Estimación de densidad mediante KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # Grilla de evaluación regular
    xmin, xmax = -9, 0
    ymin, ymax = -5, 3
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)

    # Paleta de grises suavizada con bordes nítidos
    ax.contourf(X, Y, Z, levels=niveles, cmap='Greys', alpha=0.85)
    ax.contour(X, Y, Z, levels=niveles, colors='black', linewidths=0.4, alpha=0.5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(titulo, pad=8, fontweight='bold')
    ax.set_xlabel(r"Tiempo reescalado, $\log_{10} T$")
    ax.set_ylabel(r"Distancia reescalada, $\log_{10} R$")
    ax.grid(True)
    
    # Anotación con el número de eventos en el panel
    ax.text(0.04, 0.93, f"$N = {n_events:,}$", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))


# =============================================================================
# 4. GENERACIÓN DE LA FIGURA 4 COMPLETA (4 PANELES)
# =============================================================================
def generar_figura_4_profesional(archivo_cat):
    print(f"Cargando catálogo completo desde:\n -> {archivo_cat}")
    df_raw = cargar_catalogo_scedc(archivo_cat)
    
    fecha_min = df_raw['datetime'].min().strftime('%Y-%m-%d')
    fecha_max = df_raw['datetime'].max().strftime('%Y-%m-%d')
    anio_min = df_raw['datetime'].dt.year.min()
    anio_max = df_raw['datetime'].dt.year.max()
    
    print(f"Catálogo cargado con éxito. Total eventos: {len(df_raw):,}")
    print(f"Rango de fechas detectado: {fecha_min} a {fecha_max} ({anio_min} - {anio_max})")

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 10.5))
    fig.suptitle(f"Distribución Bimodal de Componentes Espacio-Temporales $(T, R)$\nCatálogo SCEDC ({anio_min} – {anio_max})", 
                 fontweight='bold', y=0.98)

    # Línea diagonal de separación teórica: log10(T) + log10(R) = -5.0
    x_line = np.linspace(-9, 0, 100)
    y_line = -5.0 - x_line

    # -------------------------------------------------------------------------
    # Panel (a): m0 = 3.0, d = 1.6
    # -------------------------------------------------------------------------
    print("\nProcesando Panel (a): m0 = 3.0, d = 1.6...")
    df_a, n_a = calcular_componentes_zaliapin(df_raw, b=1.0, d=1.6, m0=3.0)
    graficar_panel_densidad(axs[0, 0], df_a, r"(a) $m_0 = 3.0,\ d = 1.6$", n_a)
    linea_a, = axs[0, 0].plot(x_line, y_line, 'k-', linewidth=1.6, label=r"$\log_{10} \eta_0 = -5.0$")
    axs[0, 0].legend(handles=[linea_a], loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel (b): m0 = 5.0, d = 1.6 (Scatter Plot)
    # -------------------------------------------------------------------------
    print("Procesando Panel (b): m0 = 5.0, d = 1.6...")
    df_b, n_b = calcular_componentes_zaliapin(df_raw, b=1.0, d=1.6, m0=5.0)
    axs[0, 1].scatter(df_b['log10_T'], df_b['log10_R'], facecolors='none', edgecolors='#2b2b2b', 
                      linewidths=0.8, alpha=0.8, s=28, label='Eventos ($M \ge 5.0$)')
    axs[0, 1].plot(x_line, y_line, 'k-', linewidth=1.6)
    axs[0, 1].set_xlim(-9, 0)
    axs[0, 1].set_ylim(-5, 3)
    axs[0, 1].set_title(r"(b) $m_0 = 5.0,\ d = 1.6$", pad=8, fontweight='bold')
    axs[0, 1].set_xlabel(r"Tiempo reescalado, $\log_{10} T$")
    axs[0, 1].set_ylabel(r"Distancia reescalada, $\log_{10} R$")
    axs[0, 1].grid(True)
    axs[0, 1].text(0.04, 0.93, f"$N = {n_b:,}$", transform=axs[0, 1].transAxes,
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    # -------------------------------------------------------------------------
    # Panel (c): m0 = 3.0, d = 1.0
    # -------------------------------------------------------------------------
    print("Procesando Panel (c): m0 = 3.0, d = 1.0...")
    df_c, n_c = calcular_componentes_zaliapin(df_raw, b=1.0, d=1.0, m0=3.0)
    graficar_panel_densidad(axs[1, 0], df_c, r"(c) $m_0 = 3.0,\ d = 1.0$", n_c)

    # -------------------------------------------------------------------------
    # Panel (d): m0 = 3.0, d = 2.0
    # -------------------------------------------------------------------------
    print("Procesando Panel (d): m0 = 3.0, d = 2.0...")
    df_d, n_d = calcular_componentes_zaliapin(df_raw, b=1.0, d=2.0, m0=3.0)
    graficar_panel_densidad(axs[1, 1], df_d, r"(d) $m_0 = 3.0,\ d = 2.0$", n_d)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Guardar gráfico en alta resolución (300 DPI)
    output_filename = "figura_4_zaliapin_profesional.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n¡Proceso completado! Figura guardada exitosamente como '{output_filename}'")
    plt.show()


# =============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    archivo_cat = r"C:\Users\jorge\OneDrive\Desktop\TFG-Terremotos\TFG\Codigo\SearchResults.txt"
    
    if os.path.exists(archivo_cat):
        generar_figura_4_profesional(archivo_cat)
    else:
        print(f"[ERROR] No se encuentra el archivo en la ruta:\n{archivo_cat}")