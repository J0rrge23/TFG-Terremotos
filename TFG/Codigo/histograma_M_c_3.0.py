# =====================================================================
# FIGURA 3: HISTOGRAMA BIMODAL 1D (M >= 3.0) - CORREGIDO
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from Zaliapin import calcular_zaliapin_nn
    from seismology_laws import fit_gutenberg_richter
except ImportError:
    pass

def generar_figura3_histograma_m3(df_events, output_filename="figura_3_histograma_m3.png"):
    """
    Genera el histograma de la distancia al vecino más cercano log10(eta*)
    para el catálogo con M >= 3.0 (Zaliapin et al., 2008).
    """
    # 1. Filtrar estrictamente eventos con M >= 3.0
    df_m3 = df_events[df_events['mag'] >= 3.0].copy()

    if 'log10_eta' not in df_m3.columns:
        a_cat, b_cat = fit_gutenberg_richter(df_m3['mag'].values, Mc=3.0)
        b_val = b_cat if not np.isnan(b_cat) else 1.0
        df_m3 = calcular_zaliapin_nn(df_m3, b=b_val, d=1.6, m0=3.0)

    df_m3 = df_m3.dropna(subset=['log10_eta'])
    log10_eta = df_m3['log10_eta'].values

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    # 2. Histograma de conteos
    counts, bins, patches = ax.hist(
        log10_eta, 
        bins=45, 
        color='#4682B4', 
        edgecolor='black', 
        alpha=0.8, 
        label=r'Observaciones ($M \geq 3.0$, $N = ' + str(len(df_m3)) + r'$)'
    )

    

    # 5. Etiquetas de ejes y título (con \geq corregido)
    ax.set_xlabel(r'Distancia al vecino más cercano, $\log_{10}(\eta^*)$', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Número de Observaciones ($M \geq 3.0$)', fontsize=11, fontweight='bold')
    ax.set_title(r'Figura 3: Distribución Bimodal de $\log_{10}(\eta^*)$ para $M \geq 3.0$' + '\n[California del Sur, 1984–2004]', 
                 fontsize=11.5, fontweight='bold', pad=12)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Figura 3 guardada correctamente como '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        generar_figura3_histograma_m3(df_raw)
    else:
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")