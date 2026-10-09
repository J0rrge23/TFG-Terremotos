import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


def descargar_fallas_california(output_geojson="qfaults_california.geojson"):
    """
    Descarga el conjunto de datos oficial de Fallas Cuaternarias de la USGS 
    si no existe localmente.
    """
    url_usgs = "https://earthquake.usgs.gov/static/lqt/faults/qfaults.geojson"
    if not os.path.exists(output_geojson):
        print("Descargando mapa de fallas activas de la USGS (California)...")
        try:
            r = requests.get(url_usgs, timeout=30)
            if r.status_code == 200:
                with open(output_geojson, 'wb') as f:
                    f.write(r.content)
                print("Descarga completada.")
            else:
                print(f"Error al descargar dataset de fallas. Código HTTP: {r.status_code}")
                return None
        except Exception as e:
            print(f"No se pudo descargar automáticamente las fallas: {e}")
            return None
    return output_geojson


def generar_mapa_geofisico_fallas(df, output_filename="mapa_fallas_sismicidad.pdf"):
    """
    Genera un mapa geofísico de alta resolución superponiendo el catálogo 
    clasificado por Zaliapin sobre el mapa de fallas tectónicas de California.
    """
    geojson_path = descargar_fallas_california()
    
    # Convertir catálogo de terremotos a GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf_events = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4362" if hasattr(gpd, 'crs') else "EPSG:4326")
    gdf_events = gdf_events.set_crs("EPSG:4326", allow_override=True)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # 1. Cargar y representar Fallas Tectónicas
    if geojson_path and os.path.exists(geojson_path):
        try:
            gdf_faults = gpd.read_file(geojson_path)
            # Recortar al área de California Sur
            bounds = [-122.0, -114.0, 31.5, 37.5] # [lon_min, lon_max, lat_min, lat_max]
            gdf_faults = gdf_faults.cx[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            
            gdf_faults.plot(
                ax=ax, 
                color='black', 
                linewidth=0.6, 
                alpha=0.7, 
                label='Fallas Activas (USGS)'
            )
        except Exception as e:
            print(f"Advertencia al procesar fallas: {e}")

    # 2. Representar Sismicidad de Fondo (Background)
    bg_mask = gdf_events['event_type'] == 'background'
    ax.scatter(
        gdf_events.loc[bg_mask, 'lon'], 
        gdf_events.loc[bg_mask, 'lat'], 
        s=1.5, 
        color='gray', 
        alpha=0.25, 
        label='Sismicidad de Fondo',
        zorder=2
    )

    # 3. Representar Réplicas (Aftershocks)
    af_mask = gdf_events['event_type'] == 'aftershock'
    ax.scatter(
        gdf_events.loc[af_mask, 'lon'], 
        gdf_events.loc[af_mask, 'lat'], 
        s=4, 
        color='royalblue', 
        alpha=0.4, 
        label='Réplicas Agrupadas',
        zorder=3
    )

    # 4. Destacar Terremotos Principales (Mainshocks M >= 6.5)
    ms_mask = (gdf_events['event_type'] == 'mainshock') & (gdf_events['mag'] >= 6.5)
    mainshocks = gdf_events[ms_mask]

    sc = ax.scatter(
        mainshocks['lon'], 
        mainshocks['lat'], 
        s=10 ** (mainshocks['mag'] / 2.0) * 8, 
        c=mainshocks['mag'], 
        cmap='plasma', 
        edgecolor='black', 
        linewidth=0.8,
        zorder=4,
        label='Grandes Terremotos (M $\geq$ 6.5)'
    )

    # Añadir etiquetas con el nombre/magnitud de los eventos más grandes
    for _, row in mainshocks.iterrows():
        ax.annotate(
            f"M{row['mag']:.1f}", 
            (row['lon'], row['lat']),
            textcoords="offset points", 
            xytext=(5, 5), 
            ha='left',
            fontsize=8,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8),
            zorder=5
        )

    # Configuración del Mapa
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Magnitud ($M$)", fontsize=10)

    ax.set_xlim([-121.5, -114.5])
    ax.set_ylim([32.0, 37.0])
    ax.set_xlabel("Longitud (°)", fontsize=11)
    ax.set_ylabel("Latitud (°)", fontsize=11)
    ax.set_title("Relación entre la Sismicidad Agrupada y el Sistema de Fallas de California", fontsize=12, pad=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_filename, format='pdf' if output_filename.endswith('.pdf') else 'png', dpi=300)
    plt.savefig(output_filename.replace('.pdf', '.png'), dpi=300)
    print(f"Mapa geofísico guardado con éxito como '{output_filename}' y formato PNG.")
    plt.show()


if __name__ == "__main__":
    from main import cargar_catalogo_scedc, buscar_archivo_catalogo
    from Zaliapin import calcular_zaliapin_nn
    from clustering import identificar_clusters

    archivo = buscar_archivo_catalogo()
    if os.path.exists(archivo):
        df_raw = cargar_catalogo_scedc(archivo)
        df = df_raw[df_raw['mag'] >= 2.0].reset_index(drop=True)
        df = calcular_zaliapin_nn(df, b=0.86, d=1.6, m0=2.0, max_days_back=None)
        df = identificar_clusters(df, log10_eta_0=0.0)
        generar_mapa_geofisico_fallas(df, output_filename="mapa_fallas_sismicidad.pdf")