import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Zaliapin import calcular_zaliapin_nn
from clustering import identificar_clusters
from seismology_laws import fit_gutenberg_richter, fit_omori, omori_law

def cargar_catalogo_scedc(filepath):
    """Lee y parsea el catálogo SCEDC."""
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
                    
    return pd.DataFrame(data).dropna().sort_values('datetime').reset_index(drop=True)

def buscar_archivo_catalogo():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nombres_posibles = ["SearchResults", "SearchResults.txt", "scedc_catalog.txt"]
    for nombre in nombres_posibles:
        ruta = os.path.join(script_dir, nombre)
        if os.path.exists(ruta):
            return ruta
    return os.path.join(script_dir, "SearchResults")

def main():
    archivo_scedc = buscar_archivo_catalogo()
    print(f"Ruta seleccionada: {archivo_scedc}")
    
    if not os.path.exists(archivo_scedc):
        print("\n[ERROR] No se encontró el archivo del catálogo.")
        return

    print("Cargando catálogo SCEDC...")
    df = cargar_catalogo_scedc(archivo_scedc)
    print(f"Total de eventos cargados: {len(df)}")
    
    # Filtro opcional para acelerar aún más (descartar sismos menores a M=2.0)
    m_min = 2.0
    df = df[df['mag'] >= m_min].reset_index(drop=True)
    print(f"Eventos a procesar (M >= {m_min}): {len(df)}")

    print("Calculando vecino más cercano de Zaliapin (Optimizado)...")
    df = calcular_zaliapin_nn(df, b=1.0, d=1.6, m0=2.0, max_days_back=365)
    
    print("Identificando clústeres y réplicas...")
    df = identificar_clusters(df, log10_eta_0=-5.0)
    
    print("\n--- Clasificación de Sismos ---")
    print(df['event_type'].value_counts())
    
    Mc = 2.5
    a_cat, b_cat = fit_gutenberg_richter(df['mag'].values, Mc=Mc)
    aftershocks = df[df['event_type'] == 'aftershock']
    a_aft, b_aft = fit_gutenberg_richter(aftershocks['mag'].values, Mc=Mc)
    
    print(f"\n--- Gutenberg-Richter (Mc = {Mc}) ---")
    print(f"Catálogo completo -> a: {a_cat:.2f}, b: {b_cat:.2f}")
    print(f"Réplicas (Aftershocks) -> a: {a_aft:.2f}, b: {b_aft:.2f}")
    
    print("\n--- Ley de Omori ---")
    t_data, rate_data = None, None
    try:
        t_data, rate_data, (K, c, p), _ = fit_omori(aftershocks, df)
        print(f"Parámetros de Omori -> K: {K:.2f}, c: {c:.4f} días, p: {p:.2f}")
    except Exception as e:
        print(f"No se pudo ajustar Omori: {e}")
        
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    df_valid = df.dropna(subset=['log10_T', 'log10_R', 'log10_eta'])
    axs[0].scatter(df_valid['log10_T'], df_valid['log10_R'], c=df_valid['log10_eta'], cmap='coolwarm', s=12, alpha=0.6)
    axs[0].set_xlabel(r"$\log_{10} T$ (Tiempo reescalado)")
    axs[0].set_ylabel(r"$\log_{10} R$ (Distancia reescalada)")
    axs[0].set_title("Distribución Bimodal (Zaliapin et al., 2008)")
    axs[0].grid(True)
    
    if t_data is not None:
        axs[1].loglog(t_data, rate_data, 'ko', label='Datos Réplicas')
        t_fit = np.logspace(np.log10(t_data.min()), np.log10(t_data.max()), 100)
        axs[1].loglog(t_fit, omori_law(t_fit, K, c, p), 'r-', label=f'Omori (p={p:.2f})')
        axs[1].set_xlabel("Tiempo $t$ (días)")
        axs[1].set_ylabel("Tasa $n(t)$ (réplicas/día)")
        axs[1].set_title("Ajuste de Ley de Omori")
        axs[1].legend()
        axs[1].grid(True, which="both")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()