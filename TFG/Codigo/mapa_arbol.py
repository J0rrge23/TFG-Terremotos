# =====================================================================
# MAPA CARTOGRÁFICO DE ALTA PRECISIÓN Y ÁRBOL DE ZALIAPIN (SIN INSET MAP)
# =====================================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Importar funciones de los módulos locales
try:
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from Zaliapin import calcular_zaliapin_nn
    from clustering import identificar_clusters
    from seismology_laws import fit_gutenberg_richter
except ImportError:
    pass


def cargar_fallas_geojson(ax, filepath="fallas.geojson", color='#16A085', linewidth=0.8, alpha=0.75):
    """
    Carga y dibuja la geometría exacta de las fallas desde el GeoJSON en el eje de Matplotlib.
    """
    if not os.path.exists(filepath):
        print(f"[Aviso] No se encontró '{filepath}'. Se omitirán las trazas GeoJSON.")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        for feature in features:
            geom = feature.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])

            if gtype == "LineString":
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.plot(lons, lats, color=color, lw=linewidth, alpha=alpha, zorder=2)
            elif gtype == "MultiLineString":
                for line in coords:
                    lons = [c[0] for c in line]
                    lats = [c[1] for c in line]
                    ax.plot(lons, lats, color=color, lw=linewidth, alpha=alpha, zorder=2)

        return True
    except Exception as e:
        print(f"[Error] Fallo al leer GeoJSON de fallas: {e}")
        return False


def generar_mapa_profesional_zaliapin(df_raw, log10_eta_0=0.0, Mc=3.0, output_filename="figura_5_mapa_profesional.png"):
    """
    Genera una representación cartográfica de precisión para el TFG,
    integrando fallas reales desde GeoJSON y etiquetado explícito del Mainshock.
    """
    print(f"\n[1/4] Filtrando catálogo para M >= {Mc}...")
    df = df_raw[df_raw['mag'] >= Mc].copy().reset_index(drop=True)

    print("[2/4] Calculando métrica de Zaliapin e identificando clústeres...")
    a_cat, b_cat = fit_gutenberg_richter(df['mag'].values, Mc=Mc)
    b_val = b_cat if not np.isnan(b_cat) else 1.0
    df = calcular_zaliapin_nn(df, b=b_val, d=1.6, m0=Mc)
    df = identificar_clusters(df, log10_eta_0=log10_eta_0)

    # Identificar el clúster mayor
    cluster_counts = df[df['cluster_id'] != -1]['cluster_id'].value_counts()
    top_cluster_id = cluster_counts.index[0]
    df_cluster = df[df['cluster_id'] == top_cluster_id].copy()
    ms_row = df_cluster[df_cluster['event_type'] == 'mainshock'].iloc[0]

    print("[3/4] Maquetando lienzo cartográfico...")
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    fig.patch.set_facecolor('white')

    # -----------------------------------------------------------------
    # 1. LÍMITES Y BASE GEOGRÁFICA
    # -----------------------------------------------------------------
    lon_min, lon_max = -119.5, -114.5
    lat_min, lat_max = 31.8, 37.2
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_facecolor('#D4E6F1')  # Océano Pacífico

    # Polígono costero
    coast_lon = [-120.5, -120.0, -119.6, -118.9, -118.5, -118.0, -117.3, -116.5, -114.0, -114.0, -120.5]
    coast_lat = [35.5, 34.8, 34.0, 33.8, 33.7, 33.5, 32.5, 31.5, 31.5, 37.5, 37.5]
    ax.fill(coast_lon, coast_lat, color='#FAFAFA', zorder=0, edgecolor='none')

    ax.plot([-120.5, -120.0, -119.6, -118.9, -118.5, -118.0, -117.3, -116.5], 
            [35.5, 34.8, 34.0, 33.8, 33.7, 33.5, 32.5, 31.5], 
            color='#2C3E50', lw=1.1, zorder=1)

    # -----------------------------------------------------------------
    # 2. FRONTERAS Y FALLAS GEOLÓGICAS (DESDE GEOJSON)
    # -----------------------------------------------------------------
    ax.axhline(y=32.5, color='#7F8C8D', linestyle='--', linewidth=1.0, zorder=2)
    ax.text(-115.1, 32.22, "M É X I C O", fontsize=9.5, fontweight='bold', color='#7F8C8D')
    ax.text(-115.1, 32.65, "C A L I F O R N I A", fontsize=9.5, fontweight='bold', color='#7F8C8D')

    ax.plot([-117.0, -114.0], [36.0, 35.0], color='#7F8C8D', linestyle=':', lw=1.0, zorder=2)
    ax.text(-116.2, 36.6, "N E V A D A", fontsize=9.5, fontweight='bold', color='#7F8C8D')

    fallas_cargadas = cargar_fallas_geojson(ax, filepath="fallas.geojson", color='#16A085', linewidth=0.9, alpha=0.8)

    if not fallas_cargadas:
        COLOR_FALLAS = '#16A085'
        ax.plot([-119.5, -118.5, -117.5, -116.3, -115.5], [35.3, 34.5, 34.2, 33.5, 33.0], 
                color=COLOR_FALLAS, lw=1.4, zorder=2, label='Fallas Principales')
        ax.plot([-117.3, -116.8, -116.0], [34.3, 33.8, 33.0], color=COLOR_FALLAS, lw=1.2, zorder=2)
        ax.plot([-118.5, -117.5, -116.8], [34.8, 35.4, 35.6], color=COLOR_FALLAS, lw=1.2, zorder=2)
    else:
        ax.plot([], [], color='#16A085', lw=1.2, label='Fallas Principales')

    # -----------------------------------------------------------------
    # 3. SISMICIDAD DE FONDO
    # -----------------------------------------------------------------
    df_bg = df[df['cluster_id'] != top_cluster_id]
    sizes_bg = 0.8 * np.exp(0.85 * (df_bg['mag'] - Mc))

    ax.scatter(df_bg['lon'], df_bg['lat'], s=sizes_bg, 
               facecolors='none', edgecolors='#95A5A6', linewidths=0.35, 
               alpha=0.22, label='Sismicidad Secundaria / Fondo', zorder=3)

    # -----------------------------------------------------------------
    # 4. ÁRBOL DE ZALIAPIN Y EVENTOS CLASIFICADOS
    # -----------------------------------------------------------------
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

    ax.plot([], [], color='#8E44AD', lw=0.8, label=r'Conexión de Árbol ($\log_{10}\eta \leq 0.0$)')

    # Réplicas
    df_after = df_cluster[df_cluster['event_type'] == 'aftershock']
    sizes_af = 3.2 * np.exp(0.95 * (df_after['mag'] - Mc))
    ax.scatter(df_after['lon'], df_after['lat'], s=sizes_af, 
               facecolors='#2980B9', edgecolors='#1B4F72', linewidths=0.4, 
               alpha=0.75, label=f'Réplicas Clúster ($N={len(df_after)}$)', zorder=5)

    # Precursores
    df_fore = df_cluster[df_cluster['event_type'] == 'foreshock']
    if len(df_fore) > 0:
        sizes_fo = 3.5 * np.exp(0.95 * (df_fore['mag'] - Mc))
        ax.scatter(df_fore['lon'], df_fore['lat'], s=sizes_fo, 
                   facecolors='#F39C12', edgecolors='black', linewidths=0.5, 
                   alpha=0.9, label=f'Precursores ($N={len(df_fore)}$)', zorder=6)

    # Evento Principal (Mainshock)
    ax.scatter(ms_row['lon'], ms_row['lat'], s=240, facecolors='#C0392B', 
               edgecolors='black', marker='*', lw=1.1, 
               label=f'Mainshock (M{ms_row["mag"]:.1f})', zorder=7)

    # Anotación destacada del Mainshock
    label_mainshock = f"Mainshock: Ridgecrest M{ms_row['mag']:.1f} (2019)"
    ax.annotate(
        label_mainshock, 
        xy=(ms_row['lon'], ms_row['lat']),
        xytext=(ms_row['lon'] - 1.25, ms_row['lat'] + 0.25),
        fontsize=8.5, 
        fontweight='bold', 
        color='#7B241C',
        bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC", ec="#C0392B", lw=1.0, alpha=0.95),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='#C0392B', lw=1.1),
        zorder=9
    )

    # -----------------------------------------------------------------
    # 5. TERREMOTOS HISTÓRICOS Y CIUDADES
    # -----------------------------------------------------------------
    sismos_hist = [
        {"nombre": "Landers M7.3 (1992)", "lat": 34.20, "lon": -116.43},
        {"nombre": "Hector Mine M7.1 (1999)", "lat": 34.59, "lon": -116.27},
        {"nombre": "Northridge M6.7 (1994)", "lat": 34.21, "lon": -118.54}
    ]
    for sh in sismos_hist:
        ax.scatter(sh['lon'], sh['lat'], marker='D', s=25, facecolors='none', edgecolors='#2C3E50', lw=1.0, zorder=7)
        ax.annotate(sh['nombre'], (sh['lon'], sh['lat']),
                    xytext=(sh['lon'] + 0.18, sh['lat'] + 0.10),
                    fontsize=7.8, fontweight='bold', color='#2C3E50',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", lw=0.5, alpha=0.92),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color='#2C3E50', lw=0.6),
                    zorder=8)

    ax.scatter([], [], marker='D', s=25, facecolors='none', edgecolors='#2C3E50', label='Sismos Históricos (M>6.5)')

    ciudades = [
        {"nombre": "Los Angeles", "lat": 34.05, "lon": -118.25},
        {"nombre": "San Diego", "lat": 32.71, "lon": -117.16},
        {"nombre": "Mexicali", "lat": 32.66, "lon": -115.46}
    ]
    for c in ciudades:
        ax.scatter(c['lon'], c['lat'], s=22, color='#2C3E50', marker='s', zorder=7)
        ax.text(c['lon'] + 0.1, c['lat'] - 0.10, c['nombre'], fontsize=8.5, fontweight='bold', color='#2C3E50', zorder=8)

    # -----------------------------------------------------------------
    # 6. ELEMENTOS CARTOGRÁFICOS
    # -----------------------------------------------------------------
    # Flecha Norte
    ax.annotate('N', xy=(-119.1, 36.8), xytext=(-119.1, 36.3),
                ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#2C3E50',
                arrowprops=dict(facecolor='#2C3E50', width=1.8, headwidth=6.5), zorder=9)

    # Escala
    ax.plot([-119.0, -117.9], [32.1, 32.1], color='#2C3E50', lw=2.5, zorder=9)
    ax.text(-118.45, 32.18, "100 km", ha='center', fontsize=8.0, fontweight='bold', color='#2C3E50', zorder=9)

    # -----------------------------------------------------------------
    # 7. TÍTULO Y LEYENDA
    # -----------------------------------------------------------------
    ax.set_xlabel('Longitud (°O)', fontsize=10, fontweight='bold', labelpad=5)
    ax.set_ylabel('Latitud (°N)', fontsize=10, fontweight='bold', labelpad=5)
    
    ax.set_title(r'Identificación de Réplicas y Árbol de Desencadenamiento ($M \geq ' + f'{Mc}' + r'$)' + '\n' +
                 r'Sur de California | Umbral de Corte $\log_{10}\eta_0 = 0.0$', 
                 fontsize=11, fontweight='bold', color='#2C3E50', pad=12)

    leg = ax.legend(loc='lower left', fontsize=8.0, framealpha=0.95, edgecolor='#BDC3C7')
    leg.set_zorder(10)

    ax.grid(True, linestyle=':', color='#B0BEC5', alpha=0.5, zorder=1)
    
    for spine in ax.spines.values():
        spine.set_color('#7F8C8D')
        spine.set_linewidth(0.8)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Figura guardada como '{output_filename}'")
    plt.close()


if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        generar_mapa_profesional_zaliapin(df_raw, log10_eta_0=0.0, Mc=3.0)
    else:
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")