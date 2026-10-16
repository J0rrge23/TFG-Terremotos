# =====================================================================
# MAPA CARTOGRÁFICO DE ALTA PRECISIÓN Y ÁRBOL DE ZALIAPIN (CORREGIDO)
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Importar funciones de tus módulos locales
try:
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from Zaliapin import calcular_zaliapin_nn
    from clustering import identificar_clusters
    from seismology_laws import fit_gutenberg_richter
except ImportError:
    pass


def generar_mapa_profesional_zaliapin(df_raw, log10_eta_0=0.0, Mc=3.0, output_filename="figura_5_mapa_profesional.png"):
    """
    Genera una representación cartográfica estilizada y nítida de California del Sur,
    eliminando el solapamiento masivo de puntos e integrando las fallas tectónicas,
    las ramas del árbol de Zaliapin y los eventos clasificados.
    """
    print(f"\n[1/4] Filtrando catálogo para M >= {Mc}...")
    df = df_raw[df_raw['mag'] >= Mc].copy().reset_index(drop=True)

    print("[2/4] Calculando métrica de Zaliapin y podando árbol...")
    a_cat, b_cat = fit_gutenberg_richter(df['mag'].values, Mc=Mc)
    b_val = b_cat if not np.isnan(b_cat) else 1.0
    df = calcular_zaliapin_nn(df, b=b_val, d=1.6, m0=Mc)
    df = identificar_clusters(df, log10_eta_0=log10_eta_0)

    # Identificar el clúster mayor
    cluster_counts = df[df['cluster_id'] != -1]['cluster_id'].value_counts()
    top_cluster_id = cluster_counts.index[0]
    df_cluster = df[df['cluster_id'] == top_cluster_id].copy()
    ms_row = df_cluster[df_cluster['event_type'] == 'mainshock'].iloc[0]

    print("[3/4] Maquetando lienzo cartográfico de alta calidad...")
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    fig.patch.set_facecolor('white')

    # -----------------------------------------------------------------
    # 1. LÍMITES Y COLORACIÓN GEOGRÁFICA BASE
    # -----------------------------------------------------------------
    lon_min, lon_max = -119.5, -114.5
    lat_min, lat_max = 31.8, 37.2
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Color del Océano Pacífico y Polígono Continental
    ax.set_facecolor('#B2D3E6')  # Azul océano limpio

    coast_lon = [-120.5, -119.6, -118.8, -118.0, -117.3, -116.2, -114.0, -114.0, -120.5]
    coast_lat = [35.2, 34.4, 34.0, 33.6, 32.6, 31.8, 31.8, 38.0, 38.0]
    ax.fill(coast_lon, coast_lat, color='#F9F9F6', zorder=0, edgecolor='none')  # Beige suave terrestre

    # Línea de costa principal
    ax.plot([-120.5, -119.6, -118.8, -118.0, -117.3, -116.2], 
            [35.2, 34.4, 34.0, 33.6, 32.6, 31.8], color='#34495E', lw=1.2, zorder=1)

    # -----------------------------------------------------------------
    # 2. FRONTERAS Y SISTEMA DE FALLAS PRINCIPALES
    # -----------------------------------------------------------------
    # Frontera Internacional México-EE.UU. (Separación por espacios manual)
    ax.axhline(y=32.5, color='#7F8C8D', linestyle='--', linewidth=1.1, zorder=2)
    ax.text(-115.1, 32.22, "M É X I C O", fontsize=10, fontweight='bold', color='#7F8C8D')
    ax.text(-115.1, 32.65, "C A L I F O R N I A", fontsize=10, fontweight='bold', color='#7F8C8D')
    
    # Límite estatal con Nevada
    ax.plot([-117.0, -114.0], [36.0, 35.0], color='#7F8C8D', linestyle=':', lw=1.1, zorder=2)
    ax.text(-116.1, 36.6, "N E V A D A", fontsize=10, fontweight='bold', color='#7F8C8D')

    # Trazas esquemáticas de las fallas tectónicas dominantes (Verde azulado / Teal)
    COLOR_FALLAS = '#16A085'
    
    # Falla de San Andrés (San Andreas Fault Zone)
    saf_lon = [-119.5, -118.5, -117.5, -116.3, -115.5]
    saf_lat = [35.3, 34.5, 34.2, 33.5, 33.0]
    ax.plot(saf_lon, saf_lat, color=COLOR_FALLAS, lw=1.6, zorder=2, label='Fallas Principales')

    # Falla de San Jacinto & Garlock
    sjf_lon = [-117.3, -116.8, -116.0]
    sjf_lat = [34.3, 33.8, 33.0]
    ax.plot(sjf_lon, sjf_lat, color=COLOR_FALLAS, lw=1.3, zorder=2)

    garlock_lon = [-118.5, -117.5, -116.8]
    garlock_lat = [34.8, 35.4, 35.6]
    ax.plot(garlock_lon, garlock_lat, color=COLOR_FALLAS, lw=1.3, zorder=2)

    # -----------------------------------------------------------------
    # 3. SISMICIDAD DE FONDO (SIN EFECTO 'MANCHA')
    # -----------------------------------------------------------------
    df_bg = df[df['cluster_id'] != top_cluster_id]
    sizes_bg = 0.8 * np.exp(0.85 * (df_bg['mag'] - Mc))
    
    ax.scatter(df_bg['lon'], df_bg['lat'], s=sizes_bg, 
               facecolors='none', edgecolors='#95A5A6', linewidths=0.35, 
               alpha=0.22, label='Sismicidad Secundaria / Fondo', zorder=3)

    # -----------------------------------------------------------------
    # 4. GRAFO DEL ÁRBOLE SPANNING Y CLÚSTER MAYOR
    # -----------------------------------------------------------------
    # A) Ramas del árbol (Conexiones moradas)
    G_sub = nx.DiGraph()
    indices_cluster = set(df_cluster.index)
    for idx, row in df_cluster.iterrows():
        p_idx = int(row['parent_idx'])
        if p_idx in indices_cluster:
            G_sub.add_edge(p_idx, idx)

    for u, v in G_sub.edges():
        lon1, lat1 = df.loc[u, 'lon'], df.loc[u, 'lat']
        lon2, lat2 = df.loc[v, 'lon'], df.loc[v, 'lat']
        ax.plot([lon1, lon2], [lat1, lat2], color='#8E44AD', lw=0.55, alpha=0.55, zorder=4)

    # B) Réplicas del clúster (Azul acero)
    df_after = df_cluster[df_cluster['event_type'] == 'aftershock']
    sizes_af = 3.2 * np.exp(0.95 * (df_after['mag'] - Mc))
    ax.scatter(df_after['lon'], df_after['lat'], s=sizes_af, 
               facecolors='#2980B9', edgecolors='#1B4F72', linewidths=0.4, 
               alpha=0.75, label=f'Réplicas Clúster ($N={len(df_after)}$)', zorder=5)

    # C) Precursores
    df_fore = df_cluster[df_cluster['event_type'] == 'foreshock']
    if len(df_fore) > 0:
        sizes_fo = 3.5 * np.exp(0.95 * (df_fore['mag'] - Mc))
        ax.scatter(df_fore['lon'], df_fore['lat'], s=sizes_fo, 
                   facecolors='#F39C12', edgecolors='black', linewidths=0.5, 
                   alpha=0.9, label=f'Precursores ($N={len(df_fore)}$)', zorder=6)

    # D) Evento Principal (Estrella carmesí)
    ax.scatter(ms_row['lon'], ms_row['lat'], s=220, facecolors='#C0392B', 
               edgecolors='black', marker='*', lw=1.1, 
               label=f'Mainshock (M{ms_row["mag"]:.1f})', zorder=7)

    # -----------------------------------------------------------------
    # 5. TERREMOTOS EMBLEMÁTICOS REGIONALES Y CIUDADES
    # -----------------------------------------------------------------
    sismos_hist = [
        {"nombre": "Landers M7.3 (1992)", "lat": 34.20, "lon": -116.43},
        {"nombre": "Hector Mine M7.1 (1999)", "lat": 34.59, "lon": -116.27},
        {"nombre": "Northridge M6.7 (1994)", "lat": 34.21, "lon": -118.54}
    ]
    for sh in sismos_hist:
        ax.annotate(sh['nombre'], (sh['lon'], sh['lat']),
                    xytext=(sh['lon'] + 0.18, sh['lat'] + 0.12),
                    fontsize=8, fontweight='bold', color='#2C3E50',
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", lw=0.6, alpha=0.92),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color='#2C3E50', lw=0.7),
                    zorder=8)

    ciudades = [
        {"nombre": "Los Angeles", "lat": 34.05, "lon": -118.25},
        {"nombre": "San Diego", "lat": 32.71, "lon": -117.16},
        {"nombre": "Mexicali", "lat": 32.66, "lon": -115.46}
    ]
    for c in ciudades:
        ax.scatter(c['lon'], c['lat'], s=25, color='#2C3E50', marker='s', zorder=7)
        ax.text(c['lon'] + 0.1, c['lat'] - 0.12, c['nombre'], fontsize=8.5, fontweight='bold', color='#2C3E50', zorder=8)

    # -----------------------------------------------------------------
    # 6. ELEMENTOS CARTOGRÁFICOS (ROSA DE LOS VIENTOS Y ESCALA)
    # -----------------------------------------------------------------
    # Flecha Norte
    ax.annotate('N', xy=(-119.1, 36.8), xytext=(-119.1, 36.4),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50',
                arrowprops=dict(facecolor='#2C3E50', width=2, headwidth=7), zorder=9)

    # Escala de 100 km
    ax.plot([-119.0, -117.9], [32.1, 32.1], color='#2C3E50', lw=3, zorder=9)
    ax.text(-118.45, 32.2, "100 km", ha='center', fontsize=8.5, fontweight='bold', color='#2C3E50', zorder=9)

    # -----------------------------------------------------------------
    # 7. LEYENDAS Y ESTÉTICA GENERAL
    # -----------------------------------------------------------------
    ax.set_xlabel('Longitud (°O)', fontsize=10.5, fontweight='bold', labelpad=6)
    ax.set_ylabel('Latitud (°N)', fontsize=10.5, fontweight='bold', labelpad=6)
    ax.set_title(r'Figura 5: Identificación de Réplicas y Árbol de Desencadenamiento ($M \geq ' + f'{Mc}' + r'$)' + '\n' +
                 r'Sur de California | Umbral de Corte $\log_{10}\eta_0 = 0.0$', 
                 fontsize=11.5, fontweight='bold', color='#2C3E50', pad=12)

    leg = ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95, edgecolor='#BDC3C7')
    leg.set_zorder(10)

    ax.grid(True, linestyle=':', color='#B0BEC5', alpha=0.6, zorder=1)
    
    for spine in ax.spines.values():
        spine.set_color('#7F8C8D')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Mapa profesional guardado como '{output_filename}'")
    plt.close()


if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        generar_mapa_profesional_zaliapin(df_raw, log10_eta_0=0.0, Mc=3.0)
    else:
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")