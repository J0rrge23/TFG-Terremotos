# =====================================================================
# PODA DE ÁRBOLES DE ZALIAPIN, MAPA LIMPIO Y LEYES SÍSMICAS DEL CLÚSTER
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit

# Importación de módulos locales de tu proyecto
try:
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from Zaliapin import calcular_zaliapin_nn
    from clustering import identificar_clusters
    from seismology_laws import fit_gutenberg_richter, omori_law
except ImportError:
    pass


def ejecutar_analisis_arbol_y_leyes(df_raw, log10_eta_0=0.0, Mc=3.0):
    """
    1. Filtra datos M >= Mc.
    2. Calcula la métrica de Zaliapin y poda ramas con log10(eta) > log10_eta_0.
    3. Extrae la componente conexa (clúster) más grande.
    4. Exporta 3 imágenes independientes (.png).
    """
    print(f"\n[1/5] Filtrando catálogo para M >= {Mc}...")
    df = df_raw[df_raw['mag'] >= Mc].copy().reset_index(drop=True)

    print("[2/5] Calculando métrica de vecino más cercano (Zaliapin)...")
    a_cat, b_cat = fit_gutenberg_richter(df['mag'].values, Mc=Mc)
    b_val = b_cat if not np.isnan(b_cat) else 1.0
    df = calcular_zaliapin_nn(df, b=b_val, d=1.6, m0=Mc)

    print(f"[3/5] Podando ramas con log10(eta) > {log10_eta_0} y agrupando árboles...")
    df = identificar_clusters(df, log10_eta_0=log10_eta_0)

    # Identificar el clúster con mayor número de terremotos
    cluster_counts = df[df['cluster_id'] != -1]['cluster_id'].value_counts()
    if len(cluster_counts) == 0:
        print("[ERROR] No se encontraron clústeres por debajo del umbral especificado.")
        return

    top_cluster_id = cluster_counts.index[0]
    n_eventos_top = cluster_counts.iloc[0]
    print(f"   -> Clúster más grande identificado: ID = {top_cluster_id} con N = {n_eventos_top} eventos.")

    df_cluster = df[df['cluster_id'] == top_cluster_id].copy()
    ms_row = df_cluster[df_cluster['event_type'] == 'mainshock'].iloc[0]

    # =================================================================
    # FIGURA 1: MAPA GEOGRÁFICO LIMPIO SIN 'MANCHA' CON EL ÁRBOLE SPANNING
    # =================================================================
    print("[4/5] Generando mapa geográfico del árbol del clúster principal...")
    fig1, ax1 = plt.subplots(figsize=(9, 10), dpi=300)

    # Límites geográficos
    lon_min, lon_max = -119.5, -114.5
    lat_min, lat_max = 31.8, 37.2
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)

    # Fondo Marino y Continental
    ax1.set_facecolor('#A6CAE5')
    coast_lon = [-120.0, -119.5, -118.5, -117.5, -117.2, -116.0, -114.0, -114.0, -120.0]
    coast_lat = [35.0, 34.4, 34.0, 33.5, 32.5, 31.8, 31.8, 38.0, 38.0]
    ax1.fill(coast_lon, coast_lat, color='#FDFEFE', zorder=0, edgecolor='none')

    # Fronteras y Línea de Costa
    ax1.plot([-120.0, -119.5, -118.5, -117.5, -117.2, -116.0], [35.0, 34.4, 34.0, 33.5, 32.5, 31.8], 
             color='#2980B9', lw=1.2, zorder=1)
    ax1.axhline(y=32.5, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax1.text(-115.2, 32.2, "Mexico", fontsize=10, fontweight='bold', color='#5D6D7E')
    ax1.text(-115.2, 32.7, "California", fontsize=10, fontweight='bold', color='#5D6D7E')

    # A) SISMICIDAD DE FONDO (Transparencia alta para evitar la 'mancha')
    df_bg = df[df['cluster_id'] != top_cluster_id]
    s_bg = 1.2 * np.exp(df_bg['mag'] - 2.0)
    ax1.scatter(df_bg['lon'], df_bg['lat'], s=s_bg, facecolors='none', edgecolors='#BDC3C7', 
                linewidths=0.3, alpha=0.25, label='Sismicidad Secundaria / Fondo', zorder=2)

    # B) CONEXIONES PADRE-HIJO DEL ÁRBOLE (RAMAS EN MORADO)
    G_sub = nx.DiGraph()
    indices_cluster = set(df_cluster.index)
    for idx, row in df_cluster.iterrows():
        p_idx = int(row['parent_idx'])
        if p_idx in indices_cluster:
            G_sub.add_edge(p_idx, idx)

    for u, v in G_sub.edges():
        lon1, lat1 = df.loc[u, 'lon'], df.loc[u, 'lat']
        lon2, lat2 = df.loc[v, 'lon'], df.loc[v, 'lat']
        ax1.plot([lon1, lon2], [lat1, lat2], color='#8E44AD', lw=0.5, alpha=0.5, zorder=3)

    # C) EVENTOS DEL CLÚSTER MAYOR
    df_after = df_cluster[df_cluster['event_type'] == 'aftershock']
    df_fore = df_cluster[df_cluster['event_type'] == 'foreshock']

    s_af = 3.5 * np.exp(df_after['mag'] - 2.0)
    ax1.scatter(df_after['lon'], df_after['lat'], s=s_af, facecolors='#2980B9', edgecolors='black', 
                linewidths=0.3, alpha=0.75, label=f'Réplicas Clúster ({len(df_after)})', zorder=4)

    if len(df_fore) > 0:
        s_fo = 4.0 * np.exp(df_fore['mag'] - 2.0)
        ax1.scatter(df_fore['lon'], df_fore['lat'], s=s_fo, facecolors='#F39C12', edgecolors='black', 
                    linewidths=0.4, alpha=0.9, label=f'Precursores ({len(df_fore)})', zorder=5)

    # Mainshock
    ax1.scatter(ms_row['lon'], ms_row['lat'], s=180, facecolors='#C0392B', edgecolors='black', 
                marker='*', lw=1.0, label=f'Mainshock (M{ms_row["mag"]:.1f})', zorder=6)

    # Ciudades Importantes
    ciudades = [
        {"nombre": "Los Angeles", "lat": 34.05, "lon": -118.25},
        {"nombre": "San Diego", "lat": 32.71, "lon": -117.16},
        {"nombre": "Mexicali", "lat": 32.66, "lon": -115.46}
    ]
    for c in ciudades:
        ax1.scatter(c['lon'], c['lat'], s=25, color='black', marker='s', zorder=5)
        ax1.text(c['lon'] + 0.1, c['lat'] - 0.12, c['nombre'], fontsize=8.5, fontweight='bold', color='#111111', zorder=5)

    ax1.set_xlabel('Longitud (°O)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Latitud (°N)', fontsize=11, fontweight='bold')
    ax1.set_title(r'Estructura de Árbol del Clúster Mayor ($\log_{10}\eta \leq 0.0, M \geq ' + f'{Mc}' + r'$)' + 
                  f'\n[Mainshock: M{ms_row["mag"]:.1f} - Total de eventos: N = {n_eventos_top}]', 
                  fontsize=11, fontweight='bold', pad=12)

    ax1.grid(True, linestyle=':', color='gray', alpha=0.5, zorder=1)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    file_mapa = "mapa_arbol_cluster_mayor.png"
    plt.savefig(file_mapa, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Mapa guardado como '{file_mapa}'")

    # =================================================================
    # FIGURA 2: GUTENBERG-RICHTER DEL CLÚSTER MAYOR
    # =================================================================
    print("[5/5] Calculando y graficando Gutenberg-Richter y Ley de Omori para el clúster...")
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    mags_cl = df_cluster['mag'].values
    a_cl, b_cl = fit_gutenberg_richter(mags_cl, Mc=Mc)

    bin_w = 0.1
    m_centers = np.arange(Mc, mags_cl.max() + bin_w, bin_w)
    cum_counts = np.array([np.sum(mags_cl >= m) for m in m_centers])
    disc_counts, _ = np.histogram(mags_cl, bins=np.append(m_centers, m_centers[-1]+bin_w))

    valid = cum_counts > 0
    ax2.semilogy(m_centers[valid], cum_counts[valid], 'o', color='#2980B9', markersize=5, alpha=0.85, label=r'Acumulado $N(\geq M)$')
    ax2.semilogy(m_centers[disc_counts > 0], disc_counts[disc_counts > 0], 's', color='#7F8C8D', markersize=4, alpha=0.5, label=r'Discreto $n(M)$')

    m_fit = np.linspace(Mc, mags_cl.max(), 100)
    N_fit = 10 ** (a_cl - b_cl * m_fit)
    ax2.semilogy(m_fit, N_fit, '-', color='#27AE60', lw=2.2, label=f'Ajuste Aki MLE ($a={a_cl:.2f}, b={b_cl:.2f}$)')

    ax2.set_xlabel('Magnitud (M)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Número de eventos ($N$)', fontsize=11, fontweight='bold')
    ax2.set_title(r'Ley de Gutenberg-Richter para el Clúster Mayor ($M \geq ' + f'{Mc}' + r'$)' + 
                  f'\n[$a = {a_cl:.2f}$, $b = {b_cl:.2f}$ por Máxima Verosimilitud]', 
                  fontsize=11, fontweight='bold', pad=12)
    ax2.grid(True, which="both", linestyle=':', alpha=0.5)
    ax2.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    file_gr = "gutenberg_richter_cluster_mayor.png"
    plt.savefig(file_gr, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Gutenberg-Richter guardado como '{file_gr}'")

    # =================================================================
    # FIGURA 3: LEY DE OMORI DEL CLÚSTER MAYOR
    # =================================================================
    fig3, ax3 = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    # Filtrar réplicas posteriores al mainshock
    t_ms = ms_row['datetime']
    df_afters = df_cluster[df_cluster['datetime'] > t_ms].copy()
    dt_days = (df_afters['datetime'] - t_ms).dt.total_seconds() / 86400.0

    # Intervalo de tiempo [0.01 días, 730 días]
    dt_valid = dt_days[(dt_days >= 0.01) & (dt_days <= 730.0)].values

    if len(dt_valid) > 10:
        t_bins = np.logspace(np.log10(0.01), np.log10(730.0), 20)
        counts_omori, _ = np.histogram(dt_valid, bins=t_bins)
        dt_bin_widths = np.diff(t_bins)
        t_bin_centers = np.sqrt(t_bins[:-1] * t_bins[1:])
        rates = counts_omori / dt_bin_widths

        valid_rates = rates > 0
        t_data = t_bin_centers[valid_rates]
        r_data = rates[valid_rates]

        # Ajuste no lineal n(t) = K / (t + c)^p
        popt, _ = curve_fit(omori_law, t_data, r_data, p0=[100.0, 0.1, 1.0], bounds=(0, [np.inf, 10.0, 3.0]))
        K_opt, c_opt, p_opt = popt

        ax3.loglog(t_data, r_data, 'o', color='#2C3E50', alpha=0.85, label='Tasa de réplicas observada')
        t_fit = np.logspace(np.log10(0.01), np.log10(730.0), 100)
        ax3.loglog(t_fit, omori_law(t_fit, *popt), '-', color='#8E44AD', lw=2.2, 
                   label=f'Ajuste Omori ($p={p_opt:.2f}, c={c_opt:.3f}$ d)')

        ax3.set_xlabel(r'Tiempo transcurrido $t - t_{\text{main}}$ (días)', fontsize=11, fontweight='bold')
        ax3.set_ylabel(r'Tasa de réplicas $n(t)$ (eventos / día)', fontsize=11, fontweight='bold')
        ax3.set_title(r'Ley Modificada de Omori para el Clúster Mayor' + f'\n[$p = {p_opt:.2f}$, $c = {c_opt:.3f}$ días]', 
                      fontsize=11, fontweight='bold', pad=12)
        ax3.grid(True, which="both", linestyle=':', alpha=0.5)
        ax3.legend(fontsize=9, framealpha=0.9)

        plt.tight_layout()
        file_omori = "ley_omori_cluster_mayor.png"
        plt.savefig(file_omori, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> Ley de Omori guardada como '{file_omori}'")

    print("\n[ÉXITO COMPLETO] Se han generado los 3 archivos .png independientes con éxito.")


if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        # Se fija Mc=3.0 como solicitaste para el análisis filtrado
        ejecutar_analisis_arbol_y_leyes(df_raw, log10_eta_0=0.0, Mc=3.0)
    else:
        print(f"[ERROR] No se encontró el archivo del catálogo en '{path}'.")