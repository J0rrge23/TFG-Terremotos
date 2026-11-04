# =====================================================================
# MAPA CARTOGRÁFICO DE ALTA PRECISIÓN Y ÁRBOL DE ZALIAPIN
# =====================================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Opcional para costa de máxima precisión
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Importar funciones de los módulos locales
try:
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from Zaliapin import calcular_zaliapin_nn
    from clustering import identificar_clusters
    from seismology_laws import fit_gutenberg_richter
except ImportError:
    pass


def cargar_geojson_geometria(ax, filepath, color='#16A085', linewidth=0.8, alpha=0.75, is_polygon=False, fill_color=None):
    """
    Carga y dibuja geometrías desde un archivo GeoJSON (líneas o polígonos).
    """
    if not os.path.exists(filepath):
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
            elif gtype == "Polygon" and is_polygon:
                for poly in coords:
                    lons = [c[0] for c in poly]
                    lats = [c[1] for c in poly]
                    if fill_color:
                        ax.fill(lons, lats, color=fill_color, zorder=0)
                    ax.plot(lons, lats, color=color, lw=linewidth, zorder=1)
        return True
    except Exception as e:
        print(f"[Error] Fallo al leer GeoJSON '{filepath}': {e}")
        return False


def obtener_cluster_por_coordenadas(df, target_lat, target_lon, max_dist_deg=0.5):
    """
    Localiza el ID del clúster cuyo mainshock esté más cercano a (target_lat, target_lon).
    """
    mainshocks = df[df['event_type'] == 'mainshock'].copy()
    if mainshocks.empty:
        return None

    mainshocks['dist'] = np.sqrt((mainshocks['lat'] - target_lat)**2 + (mainshocks['lon'] - target_lon)**2)
    candidatos = mainshocks[mainshocks['dist'] <= max_dist_deg]

    if candidatos.empty:
        print(f"[Aviso] No se encontró ningún mainshock en un radio de {max_dist_deg}° alrededor de ({target_lat}, {target_lon}).")
        return None

    evento_cercano = candidatos.sort_values('dist').iloc[0]
    return evento_cercano['cluster_id']


def generar_mapa_profesional_zaliapin(df_procesado, cluster_id, nombre_evento="Evento", output_filename="mapa_terremoto.png", Mc=3.0):
    """
    Genera el mapa cartográfico para un clúster de sismos específico con costa precisa.
    """
    df_cluster = df_procesado[df_procesado['cluster_id'] == cluster_id].copy()
    if df_cluster.empty:
        print(f"[Error] El clúster ID {cluster_id} no contiene eventos.")
        return

    ms_rows = df_cluster[df_cluster['event_type'] == 'mainshock']
    if ms_rows.empty:
        # En caso de no tener etiqueta explícita de mainshock, tomar el de mayor magnitud
        ms_row = df_cluster.sort_values('mag', ascending=False).iloc[0]
    else:
        ms_row = ms_rows.iloc[0]

    # Configuración de lienzo con Cartopy si está disponible, o Matplotlib estándar
    if HAS_CARTOPY:
        fig, ax = plt.subplots(figsize=(10, 11), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-119.5, -114.5, 31.8, 37.2], crs=ccrs.PlateCarree())
        
        # Añadir elementos cartográficos reales de alta resolución
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#D4E6F1', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#FAFAFA', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#2C3E50', linewidth=1.1, zorder=1)
        ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#D4E6F1', edgecolor='#2C3E50', linewidth=0.5, zorder=1)
    else:
        fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
        lon_min, lon_max = -119.5, -114.5
        lat_min, lat_max = 31.8, 37.2
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_facecolor('#D4E6F1')  # Color del océano por defecto

        # Intento de carga de costa vía GeoJSON
        costa_cargada = cargar_geojson_geometria(ax, filepath="costa_california.geojson", color='#2C3E50', linewidth=1.1, is_polygon=True, fill_color='#FAFAFA')
        if not costa_cargada:
            # Polígono de respaldo si no existe el GeoJSON
            coast_lon = [-120.5, -120.0, -119.6, -118.9, -118.5, -118.0, -117.3, -116.5, -114.0, -114.0, -120.5]
            coast_lat = [35.5, 34.8, 34.0, 33.8, 33.7, 33.5, 32.5, 31.5, 31.5, 37.5, 37.5]
            ax.fill(coast_lon, coast_lat, color='#FAFAFA', zorder=0)
            ax.plot(coast_lon[:-2], coast_lat[:-2], color='#2C3E50', lw=1.1, zorder=1)

    fig.patch.set_facecolor('white')

    # -----------------------------------------------------------------
    # FRONTERAS Y FALLAS GEOLÓGICAS
    # -----------------------------------------------------------------
    ax.axhline(y=32.5, color='#7F8C8D', linestyle='--', linewidth=1.0, zorder=2)
    ax.text(-115.1, 32.22, "M É X I C O", fontsize=9.5, fontweight='bold', color='#7F8C8D')
    ax.text(-115.1, 32.65, "C A L I F O R N I A", fontsize=9.5, fontweight='bold', color='#7F8C8D')

    ax.plot([-117.0, -114.0], [36.0, 35.0], color='#7F8C8D', linestyle=':', lw=1.0, zorder=2)
    ax.text(-116.2, 36.6, "N E V A D A", fontsize=9.5, fontweight='bold', color='#7F8C8D')

    cargar_geojson_geometria(ax, filepath="fallas.geojson", color='#16A085', linewidth=0.9, alpha=0.8)
    ax.plot([], [], color='#16A085', lw=1.2, label='Fallas Principales')

    # -----------------------------------------------------------------
    # SISMICIDAD DE FONDO
    # -----------------------------------------------------------------
    df_bg = df_procesado[df_procesado['cluster_id'] != cluster_id]
    sizes_bg = 0.8 * np.exp(0.85 * (df_bg['mag'] - Mc))

    ax.scatter(df_bg['lon'], df_bg['lat'], s=sizes_bg, 
               facecolors='none', edgecolors='#95A5A6', linewidths=0.35, 
               alpha=0.22, label='Sismicidad Secundaria / Fondo', zorder=3)

    # -----------------------------------------------------------------
    # ÁRBOL DE ZALIAPIN Y EVENTOS
    # -----------------------------------------------------------------
    G_sub = nx.DiGraph()
    indices_cluster = set(df_cluster.index)
    for idx, row in df_cluster.iterrows():
        p_idx = int(row['parent_idx'])
        if p_idx in indices_cluster:
            G_sub.add_edge(p_idx, idx)

    for u, v in G_sub.edges():
        lon1, lat1 = df_procesado.loc[u, 'lon'], df_procesado.loc[u, 'lat']
        lon2, lat2 = df_procesado.loc[v, 'lon'], df_procesado.loc[v, 'lat']
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

    # Mainshock
    ax.scatter(ms_row['lon'], ms_row['lat'], s=240, facecolors='#C0392B', 
               edgecolors='black', marker='*', lw=1.1, 
               label=f'Mainshock (M{ms_row["mag"]:.1f})', zorder=7)

    # Anotación destacada
    label_mainshock = f"Mainshock: {nombre_evento} M{ms_row['mag']:.1f}"
    ax.annotate(
        label_mainshock, 
        xy=(ms_row['lon'], ms_row['lat']),
        xytext=(ms_row['lon'] - 1.25, ms_row['lat'] + 0.25),
        fontsize=8.5, fontweight='bold', color='#7B241C',
        bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC", ec="#C0392B", lw=1.0, alpha=0.95),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='#C0392B', lw=1.1),
        zorder=9
    )

    # Ciudades principales
    ciudades = [
        {"nombre": "Los Angeles", "lat": 34.05, "lon": -118.25},
        {"nombre": "San Diego", "lat": 32.71, "lon": -117.16},
        {"nombre": "Mexicali", "lat": 32.66, "lon": -115.46}
    ]
    for c in ciudades:
        ax.scatter(c['lon'], c['lat'], s=22, color='#2C3E50', marker='s', zorder=7)
        ax.text(c['lon'] + 0.1, c['lat'] - 0.10, c['nombre'], fontsize=8.5, fontweight='bold', color='#2C3E50', zorder=8)

    # Elementos cartográficos (Escala y Norte)
    ax.annotate('N', xy=(-119.1, 36.8), xytext=(-119.1, 36.3),
                ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#2C3E50',
                arrowprops=dict(facecolor='#2C3E50', width=1.8, headwidth=6.5), zorder=9)

    ax.plot([-119.0, -117.9], [32.1, 32.1], color='#2C3E50', lw=2.5, zorder=9)
    ax.text(-118.45, 32.18, "100 km", ha='center', fontsize=8.0, fontweight='bold', color='#2C3E50', zorder=9)

    # Título y Leyenda
    ax.set_xlabel('Longitud (°O)', fontsize=10, fontweight='bold', labelpad=5)
    ax.set_ylabel('Latitud (°N)', fontsize=10, fontweight='bold', labelpad=5)
    ax.set_title(f'Identificación de Réplicas y Árbol de Desencadenamiento - {nombre_evento}\n' +
                 r'Sur de California | Umbral $\log_{10}\eta_0 = 0.0$', 
                 fontsize=11, fontweight='bold', color='#2C3E50', pad=12)

    leg = ax.legend(loc='lower left', fontsize=8.0, framealpha=0.95, edgecolor='#BDC3C7')
    leg.set_zorder(10)
    ax.grid(True, linestyle=':', color='#B0BEC5', alpha=0.5, zorder=1)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Figura guardada como '{output_filename}'")
    plt.close()


def ejecutar_analisis_multievento(log10_eta_0=0.0, Mc=3.0):
    path = buscar_archivo_catalogo()
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")
        return

    df_raw = cargar_catalogo_scedc(path)
    df = df_raw[df_raw['mag'] >= Mc].copy().reset_index(drop=True)

    # Cálculo global de Zaliapin
    a_cat, b_cat = fit_gutenberg_richter(df['mag'].values, Mc=Mc)
    b_val = b_cat if not np.isnan(b_cat) else 1.0
    df = calcular_zaliapin_nn(df, b=b_val, d=1.6, m0=Mc)
    df = identificar_clusters(df, log10_eta_0=log10_eta_0)

    # Definición de terremotos a analizar (Coordenadas aproximadas del epicentro)
    eventos_objetivo = [
        {"nombre": "Northridge (1994)", "lat": 34.21, "lon": -118.54, "archivo": "figura_northridge_1994.png"},
        {"nombre": "Landers (1992)", "lat": 34.20, "lon": -116.43, "archivo": "figura_landers_1992.png"},
        {"nombre": "Hector Mine (1999)", "lat": 34.59, "lon": -116.27, "archivo": "figura_hector_mine_1999.png"},
        {"nombre": "Ridgecrest (2019)", "lat": 35.71, "lon": -117.50, "archivo": "figura_ridgecrest_2019.png"}
    ]

    for ev in eventos_objetivo:
        cid = obtener_cluster_por_coordenadas(df, ev['lat'], ev['lon'])
        if cid is not None:
            print(f"\nGenerando mapa para {ev['nombre']} (Cluster ID: {cid})...")
            generar_mapa_profesional_zaliapin(
                df_procesado=df, 
                cluster_id=cid, 
                nombre_evento=ev['nombre'], 
                output_filename=ev['archivo'], 
                Mc=Mc
            )


if __name__ == "__main__":
    ejecutar_analisis_multievento(log10_eta_0=0.0, Mc=3.0)