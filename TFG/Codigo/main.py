import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Zaliapin import calcular_zaliapin_nn
from clustering import identificar_clusters
from seismology_laws import (
    fit_gutenberg_richter, 
    fit_omori_single_cluster, 
    omori_law, 
    graficar_gutenberg_richter, 
    graficar_comparativa_gr
)


def cargar_catalogo_scedc(filepath):
    """Parsea el archivo ASCII del catálogo SCEDC descartando cabeceras."""
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


def buscar_archivo_catalogo():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nombres_posibles = ["SearchResults", "SearchResults.txt", "scedc_catalog.txt"]
    for nombre in nombres_posibles:
        ruta = os.path.join(script_dir, nombre)
        if os.path.exists(ruta):
            return ruta
    return os.path.join(script_dir, "SearchResults.txt")


def main():
    archivo_scedc = buscar_archivo_catalogo()
    print(f"Ruta del catálogo seleccionada: {archivo_scedc}")
    
    if not os.path.exists(archivo_scedc):
        print(f"[ERROR] No se encontró el archivo del catálogo en: {archivo_scedc}")
        return

    print("Cargando y parseando catálogo...")
    df_raw = cargar_catalogo_scedc(archivo_scedc)
    print(f"Total de eventos cargados sin filtrar: {len(df_raw)}")
    
    # =========================================================================
    # PASO 1: Filtrar M >= 2.0 y ajustar Gutenberg-Richter inicial (con gráfico)
    # =========================================================================
    m_min = 2.0
    df = df_raw[df_raw['mag'] >= m_min].reset_index(drop=True)
    print(f"Eventos a procesar (M >= {m_min}): {len(df)}")

    print(f"\n--- PASO 1: Gutenberg-Richter Catálogo Completo (M >= {m_min}) ---")
    a_cat, b_cat = graficar_gutenberg_richter(
        df['mag'].values, 
        Mc=m_min, 
        titulo="Gutenberg-Richter - Catálogo Completo", 
        filename="gutenberg_richter_catalogo_completo.png"
    )
    print(f"a: {a_cat:.2f}, b: {b_cat:.2f}")

    # =========================================================================
    # PASO 2: Método de Zaliapin, Histograma Bimodal y Figuras 3 y 4
    # =========================================================================
    print("\n--- PASO 2: Aplicando Método de Zaliapin (Catálogo Completo) ---")
    b_usado = b_cat if not np.isnan(b_cat) else 1.0
    df = calcular_zaliapin_nn(df, b=b_usado, d=1.6, m0=m_min, max_days_back=None)

    df_valid = df.dropna(subset=['log10_eta', 'log10_T', 'log10_R'])

    # Histograma 1D para visualizar las dos poblaciones (fondo y agrupados)
    plt.figure(figsize=(8, 4))
    plt.hist(df_valid['log10_eta'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel(r"$\log_{10} \eta$")
    plt.ylabel("Frecuencia")
    plt.title(r"Histograma Bimodal de Proximidad Espacio-Temporal ($\log_{10} \eta$)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("histograma_zaliapin_bimodal.png", dpi=300)
    plt.show()

    # Selección del umbral log10_eta_0 (por defecto -5.0 según el paper)
    log10_eta_0 = 0.0
    print(f"Fijando umbral de separación log10_eta_0 = {log10_eta_0}")
    
    df = identificar_clusters(df, log10_eta_0=log10_eta_0)

    print("\n--- Clasificación de Eventos ---")
    print(df['event_type'].value_counts())

    # Reproducción de las Figuras 3 y 4 del Paper
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Figura 3: Diagrama T - R
    sc = axs[0].scatter(df_valid['log10_T'], df_valid['log10_R'], c=df_valid['log10_eta'], cmap='coolwarm', s=8, alpha=0.5)
    axs[0].set_xlabel(r"$\log_{10} T$ (Tiempo reescalado)")
    axs[0].set_ylabel(r"$\log_{10} R$ (Distancia reescalada)")
    axs[0].set_title("Diagrama T - R (Reproducción Fig. 3 Zaliapin)")
    axs[0].grid(True)
    fig.colorbar(sc, ax=axs[0], label=r"$\log_{10} \eta$")

    # Figura 4: Distribución de tamaños de clústeres
    cluster_sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts()
    axs[1].hist(cluster_sizes, bins=np.logspace(0, np.log10(cluster_sizes.max()), 20), color='salmon', edgecolor='black')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Tamaño de la Familia / Clúster (N)")
    axs[1].set_ylabel("Frecuencia de Familias")
    axs[1].set_title("Distribución de Tamaño de Clústeres (Reproducción Fig. 4)")
    axs[1].grid(True, which="both")

    plt.tight_layout()
    plt.savefig("figuras_3_4_zaliapin.png", dpi=300)
    plt.show()

    # =========================================================================
    # PASO 3: Ley de Omori en 2-3 terremotos grandes (Ventana de 2 años)
    # =========================================================================
    print("\n--- PASO 3: Ley de Omori en los Terremotos Más Grandes ---")
    mainshocks = df[df['event_type'] == 'mainshock'].sort_values('mag', ascending=False)
    top_mainshocks = mainshocks.head(3)

    for idx, ms in top_mainshocks.iterrows():
        c_id = ms['cluster_id']
        cluster_events = df[df['cluster_id'] == c_id]
        print(f"\nAnalizando Mainshock Mag {ms['mag']} | Fecha: {ms['datetime']} | Cluster ID: {c_id}")
        
        try:
            # t_min_days=0.01 descarta las primeras ~15 min por falta de completitud
            t_data, rate_data, popt = fit_omori_single_cluster(cluster_events, ms, t_min_days=0.01, t_max_days=730.0)
            K, c, p = popt
            print(f"  Parámetros Omori -> K: {K:.2f}, c: {c:.4f} días, p: {p:.2f}")

            plt.figure(figsize=(6, 4))
            plt.loglog(t_data, rate_data, 'ko', label='Réplicas observadas')
            t_fit = np.logspace(np.log10(t_data.min()), np.log10(t_data.max()), 100)
            plt.loglog(t_fit, omori_law(t_fit, *popt), 'r-', label=f'Omori (p={p:.2f})')
            plt.xlabel("Tiempo $t - t_{main}$ (días)")
            plt.ylabel("Tasa $n(t)$ (réplicas/día)")
            plt.title(f"Ajuste Omori (Mainshock M={ms['mag']})")
            plt.legend()
            plt.grid(True, which="both")
            plt.tight_layout()
            plt.savefig(f"omori_mainshock_cluster_{c_id}.png", dpi=300)
            plt.show()
        except Exception as e:
            print(f"  No se pudo realizar el ajuste de Omori: {e}")

    # =========================================================================
    # PASO 4: Gutenberg-Richter en Fondo (Background) vs Réplicas (Aftershocks)
    # =========================================================================
    print("\n--- PASO 4: Comparación Gutenberg-Richter (Fondo vs Réplicas) ---")
    bg_events = df[df['event_type'] == 'background']
    af_events = df[df['event_type'] == 'aftershock']

    a_bg, b_bg = fit_gutenberg_richter(bg_events['mag'].values, Mc=m_min)
    a_af, b_af = fit_gutenberg_richter(af_events['mag'].values, Mc=m_min)

    print(f"Sismos de Fondo (Background, N={len(bg_events)}) -> a: {a_bg:.2f}, b: {b_bg:.2f}")
    print(f"Réplicas (Aftershocks, N={len(af_events)})       -> a: {a_af:.2f}, b: {b_af:.2f}")

    # Gráfico comparativo
    graficar_comparativa_gr(
        bg_events['mag'].values, 
        af_events['mag'].values, 
        Mc=m_min, 
        filename="gutenberg_richter_fondo_vs_replicas.png"
    )

    print("\n¡Procesamiento finalizado con éxito!")


if __name__ == "__main__":
    main()