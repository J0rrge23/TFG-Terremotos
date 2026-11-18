import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("[AVISO] Instala cartopy ('pip install cartopy') para máxima resolución costera.")

from main import buscar_archivo_catalogo, cargar_catalogo_scedc
from Zaliapin import calcular_zaliapin_nn
from clustering import identificar_clusters
from seismology_laws import fit_gutenberg_richter


def cargar_fallas_geojson(ax, filepath="fallas.geojson", color='#16A085', linewidth=0.85, alpha=0.8):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        kw = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            if gtype == "LineString":
                ax.plot([c[0] for c in coords], [c[1] for c in coords], color=color, lw=linewidth, alpha=alpha, zorder=2, **kw)
            elif gtype == "MultiLineString":
                for line in coords:
                    ax.plot([c[0] for c in line], [c[1] for c in line], color=color, lw=linewidth, alpha=alpha, zorder=2, **kw)
        return True
    except Exception:
        return False


def generar_mapa_tres_arboles(df_raw, log10_eta_0=-3.3, Mc=3.0, output_filename="figura_5_mapa_TFG.png"):
    print(f"[1/4] Filtrando catálogo para M >= {Mc}...")
    df = df_raw[df_raw['mag'] >= Mc].copy().reset_index(drop=True)

    print("[2/4] Calculando métrica de Zaliapin e identificando clústeres...")
    _, b_cat = fit_gutenberg_richter(df['mag'].values, Mc=Mc)
    b_val = b_cat if not np.isnan(b_cat) else 1.0
    df = calcular_zaliapin_nn(df, b=b_val, d=1.6, m0=Mc)
    df = identificar_clusters(df, log10_eta_0=log10_eta_0)

    print("[3/4] Maquetando mapa con línea de costa real y 3 árboles...")
    fig = plt.figure(figsize=(10, 11), dpi=300)
    fig.patch.set_facecolor('white')

    lon_min, lon_max = -119.5, -114.5
    lat_min, lat_max = 31.8, 37.2

    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        # Línea de costa real de alta resolución (10m)
        ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='#D4E6F1', edgecolor='none')
        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='#FAFAFA', edgecolor='none')
        coast_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', facecolor='none', edgecolor='#2C3E50', linewidth=1.1)
        borders_10m = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', facecolor='none', edgecolor='#7F8C8D', linestyle='--', linewidth=1.0)
        states_10m = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', facecolor='none', edgecolor='#7F8C8D', linestyle=':', linewidth=0.9)

        ax.add_feature(ocean_10m, zorder=0)
        ax.add_feature(land_10m, zorder=0)
        ax.add_feature(coast_10m, zorder=2)
        ax.add_feature(borders_10m, zorder=2)
        ax.add_feature(states_10m, zorder=2)
        kw_geo = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_facecolor('#D4E6F1')
        kw_geo = {}

    # Fallas geológicas
    cargar_fallas_geojson(ax, filepath="fallas.geojson")
    ax.plot([], [], color='#16A085', lw=1.2, label='Fallas Principales')

    # Los 3 terremotos principales cuyos árboles queremos representar
    sismos_objetivo = [
        {"nombre": "Ridgecrest M7.1 (2019)", "min_mag": 7.0, "year": 2019, "offset": (-1.35, 0.25)},
        {"nombre": "Landers M7.3 (1992)",    "min_mag": 7.2, "year": 1992, "offset": (0.20, -0.25)},
        {"nombre": "Hector Mine M7.1 (1999)","min_mag": 7.0, "year": 1999, "offset": (0.20, 0.15)},
    ]

    clusters_objetivo = set()
    eventos_principales = []

    for sismo in sismos_objetivo:
        sub = df[(df['mag'] >= sismo['min_mag']) & (df['datetime'].dt.year == sismo['year'])]
        if not sub.empty:
            ms_row = sub.sort_values('mag', ascending=False).iloc[0]
            c_id = ms_row['cluster_id']
            if c_id != -1:
                clusters_objetivo.add(c_id)
                eventos_principales.append((sismo, ms_row, c_id))

    # 1. Sismicidad secundaria / fondo (todo lo que NO pertenece a los 3 árboles)
    df_bg = df[~df['cluster_id'].isin(clusters_objetivo)]
    sizes_bg = 0.8 * np.exp(0.85 * (df_bg['mag'] - Mc))
    ax.scatter(
        df_bg['lon'], df_bg['lat'], s=sizes_bg,
        facecolors='none', edgecolors='#95A5A6', linewidths=0.35,
        alpha=0.25, label='Sismicidad Secundaria / Fondo', zorder=3, **kw_geo
    )

    total_after = 0
    total_fore = 0

    # 2. Dibujar cada uno de los 3 árboles completos
    for sismo, ms_row, c_id in eventos_principales:
        df_cluster = df[df['cluster_id'] == c_id].copy()
        indices_cluster = set(df_cluster.index)

        # Conexiones del árbol
        for idx, row in df_cluster.iterrows():
            p_idx = int(row['parent_idx'])
            if p_idx in indices_cluster:
                lon1, lat1 = df.loc[p_idx, 'lon'], df.loc[p_idx, 'lat']
                lon2, lat2 = df.loc[idx, 'lon'], df.loc[idx, 'lat']
                ax.plot([lon1, lon2], [lat1, lat2], color='#8E44AD', lw=0.6, alpha=0.6, zorder=4, **kw_geo)

        # Réplicas del árbol
        df_after = df_cluster[df_cluster['event_type'] == 'aftershock']
        total_after += len(df_after)
        sizes_af = 3.2 * np.exp(0.95 * (df_after['mag'] - Mc))
        ax.scatter(
            df_after['lon'], df_after['lat'], s=sizes_af,
            facecolors='#2980B9', edgecolors='#1B4F72', linewidths=0.4,
            alpha=0.78, zorder=5, **kw_geo
        )

        # Precursores del árbol
        df_fore = df_cluster[df_cluster['event_type'] == 'foreshock']
        total_fore += len(df_fore)
        if len(df_fore) > 0:
            sizes_fo = 3.5 * np.exp(0.95 * (df_fore['mag'] - Mc))
            ax.scatter(
                df_fore['lon'], df_fore['lat'], s=sizes_fo,
                facecolors='#F39C12', edgecolors='black', linewidths=0.5,
                alpha=0.9, zorder=6, **kw_geo
            )

        # Estrella roja del Mainshock
        ax.scatter(
            ms_row['lon'], ms_row['lat'], s=240, facecolors='#C0392B',
            edgecolors='black', marker='*', lw=1.1, zorder=7, **kw_geo
        )

        # Etiqueta roja destacada para cada uno de los 3 terremotos
        dx, dy = sismo['offset']
        ax.annotate(
            f"Mainshock: {sismo['nombre']}",
            xy=(ms_row['lon'], ms_row['lat']),
            xytext=(ms_row['lon'] + dx, ms_row['lat'] + dy),
            fontsize=8.3, fontweight='bold', color='#7B241C',
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC", ec="#C0392B", lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.12", color='#C0392B', lw=1.1),
            zorder=9, **kw_geo
        )

    # Entradas de leyenda consolidadas
    ax.plot([], [], color='#8E44AD', lw=0.9, label=rf'Conexión de Árbol ($\log_{{10}}\eta \leq {log10_eta_0}$)')
    ax.scatter([], [], s=35, facecolors='#2980B9', edgecolors='#1B4F72', label=f'Réplicas 3 Árboles ($N={total_after}$)')
    ax.scatter([], [], s=35, facecolors='#F39C12', edgecolors='black', label=f'Precursores ($N={total_fore}$)')
    ax.scatter([], [], s=200, facecolors='#C0392B', edgecolors='black', marker='*', label='Mainshocks (3 Árboles)')

    # Northridge como sismo histórico de referencia
    ax.scatter(-118.54, 34.21, marker='D', s=28, facecolors='none', edgecolors='#2C3E50', lw=1.0, label='Sismo Histórico (Northridge M6.7)', zorder=7, **kw_geo)
    ax.annotate("Northridge M6.7 (1994)", (-118.54, 34.21), xytext=(-118.35, 34.32),
                fontsize=7.8, fontweight='bold', color='#2C3E50',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", lw=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color='#2C3E50', lw=0.6), zorder=8, **kw_geo)

    # Ciudades y etiquetas geográficas
    ciudades = [
        {"nombre": "Los Angeles", "lat": 34.05, "lon": -118.25},
        {"nombre": "San Diego",   "lat": 32.71, "lon": -117.16},
        {"nombre": "Mexicali",    "lat": 32.66, "lon": -115.46}
    ]
    for c in ciudades:
        ax.scatter(c['lon'], c['lat'], s=22, color='#2C3E50', marker='s', zorder=7, **kw_geo)
        ax.text(c['lon'] + 0.08, c['lat'] - 0.09, c['nombre'], fontsize=8.5, fontweight='bold', color='#2C3E50', zorder=8, **kw_geo)

    ax.text(-115.1, 32.22, "M É X I C O", fontsize=9.5, fontweight='bold', color='#7F8C8D', **kw_geo)
    ax.text(-115.1, 32.65, "C A L I F O R N I A", fontsize=9.5, fontweight='bold', color='#7F8C8D', **kw_geo)
    ax.text(-116.2, 36.6, "N E V A D A", fontsize=9.5, fontweight='bold', color='#7F8C8D', **kw_geo)

    # Flecha Norte y Escala
    ax.annotate('N', xy=(-119.1, 36.8), xytext=(-119.1, 36.3),
                ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#2C3E50',
                arrowprops=dict(facecolor='#2C3E50', width=1.8, headwidth=6.5), zorder=9, **kw_geo)
    ax.plot([-119.0, -117.9], [32.1, 32.1], color='#2C3E50', lw=2.5, zorder=9, **kw_geo)
    ax.text(-118.45, 32.18, "100 km", ha='center', fontsize=8.0, fontweight='bold', color='#2C3E50', zorder=9, **kw_geo)

    ax.set_title(
        rf'Identificación de Réplicas y Árboles de Desencadenamiento ($M \geq {Mc}$)' + '\n' +
        rf'Sur de California | Umbral de Corte $\log_{{10}}\eta_0 = {log10_eta_0}$',
        fontsize=11, fontweight='bold', color='#2C3E50', pad=12
    )

    if HAS_CARTOPY:
        gl = ax.gridlines(draw_labels=True, linestyle=':', color='#B0BEC5', alpha=0.6, zorder=1)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9.5}
        gl.ylabel_style = {'size': 9.5}
        ax.text(0.5, -0.05, 'Longitud (°O)', va='bottom', ha='center', rotation='horizontal',
                transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.text(-0.08, 0.5, 'Latitud (°N)', va='bottom', ha='center', rotation='vertical',
                transform=ax.transAxes, fontsize=10, fontweight='bold')

    leg = ax.legend(loc='lower left', fontsize=8.0, framealpha=0.95, edgecolor='#BDC3C7')
    leg.set_zorder(10)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Figura guardada como '{output_filename}'")
    plt.close()


if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        generar_mapa_tres_arboles(df_raw, log10_eta_0=-3.3, Mc=3.0)