# =====================================================================
# FIGURA 3: HISTOGRAMA BIMODAL 3 PANELES (M>=2, M>=3, M>=4)
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

def generar_figura3_tres_paneles(df_events, output_filename="figura_3_histograma_3_paneles.png"):
    """
    Genera los 3 histogramas apilados (M>=2, M>=3, M>=4) ajustando el eje X a [-9, 3].
    """
    magnitudes = [2.0, 3.0, 4.0]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, dpi=300)
    
    bins_range = np.linspace(-9, 3, 50)
    
    for ax, mc in zip(axes, magnitudes):
        df_m = df_events[df_events['mag'] >= mc].copy()

        if 'log10_eta' not in df_m.columns:
            a_cat, b_cat = fit_gutenberg_richter(df_m['mag'].values, Mc=mc)
            b_val = b_cat if not np.isnan(b_cat) else 1.0
            df_m = calcular_zaliapin_nn(df_m, b=b_val, d=1.6, m0=mc)

        df_m = df_m.dropna(subset=['log10_eta'])
        log10_eta = df_m['log10_eta'].values

        # Histograma
        ax.hist(log10_eta, bins=bins_range, color='#4A6572', edgecolor='white', linewidth=0.5, alpha=0.85)

        # Sombreado de poblaciones
        ax.axvspan(-9, -5, color='#FADBD8', alpha=0.3, zorder=0)
        ax.axvspan(-5, 3, color='#D4EFDF', alpha=0.3, zorder=0)
        ax.axvline(x=-5.0, color='#900C3F', linestyle='--', linewidth=1.2)

        # Texto N y m
        ax.text(0.05, 0.70, f"$m \\geq {int(mc)}$\n$N = {len(df_m):,}$", 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.9))

        ax.set_ylabel('Número de eventos', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Ajustar el límite del eje X a -9
    axes[-1].set_xlim(-9, 3)
    axes[-1].set_xlabel(r'Distancia al vecino más cercano, $\log_{10}\eta^*$', fontsize=11, fontweight='bold')

    fig.suptitle(r'Figura 3: Distribución Bimodal de $\log_{10}\eta^*$ en Rango [-9, +3]' + '\n' +
                 r'Catálogo del Sur de California (SCEDC) | Umbral de Corte $\log_{10}\eta_0 = -5.0$', 
                 fontsize=11.5, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Figura de 3 paneles guardada como '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        generar_figura3_tres_paneles(df_raw)
    else:
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")