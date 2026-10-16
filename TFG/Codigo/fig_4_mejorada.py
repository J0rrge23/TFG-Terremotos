# =====================================================================
# FIGURA 3: VISUALIZACIÓN COMPLETA Y PROFESIONAL (RANGO -12 A +3)
# Distribución bimodal de log10(eta*) mostrando la cola derecha completa
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from Zaliapin import calcular_zaliapin_nn
    from seismology_laws import fit_gutenberg_richter
except ImportError:
    pass


def generar_figura3_profesional_completa(df_events, output_filename="figura_3_zaliapin_profesional_ampliada.png"):
    """
    Genera la Figura 3 de Zaliapin et al. (2008) extendiendo el eje X hasta +3.0
    para visualizar la cola completa de la sismicidad de fondo (log10 eta* > 0).
    """
    # 1. Filtro base m >= 2.0 y cálculo de la métrica de Zaliapin
    df_base = df_events[df_events['mag'] >= 2.0].copy()

    if 'log10_eta' not in df_base.columns:
        a_cat, b_cat = fit_gutenberg_richter(df_base['mag'].values, Mc=2.0)
        b_val = b_cat if not np.isnan(b_cat) else 1.0
        df_base = calcular_zaliapin_nn(df_base, b=b_val, d=1.6, m0=2.0)

    df_base = df_base.dropna(subset=['log10_eta'])

    # 2. Estilo global y paleta cromática profesional
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    COLOR_BARS = '#34495E'       # Gris pizarra elegante
    COLOR_KDE = '#1F618D'        # Azul profundo para la curva KDE
    COLOR_THRESHOLD = '#C0392B'  # Rojo carmesí para el umbral principal (-5)
    COLOR_SEC_THRES = '#27AE60'  # Verde esmeralda para log10(eta) = 0
    COLOR_CLUSTER_BG = '#FDEDEC' # Fondo rosado tenue para réplicas
    COLOR_POISSON_BG = '#EAFAF1' # Fondo verde tenue para sismicidad de fondo

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, dpi=300)
    fig.patch.set_facecolor('white')

    cutoffs = [2.0, 3.0, 4.0]
    # AMPLIACIÓN DEL RANGO: De -12.0 a +3.0 en lugar de parar en 0.0
    bins_edges = np.linspace(-12.0, 3.0, 55)

    for idx, m_cut in enumerate(cutoffs):
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')

        df_sub = df_base[df_base['mag'] >= m_cut]
        data_eta = df_sub['log10_eta'].values
        N_obs = len(data_eta)

        # 3. Sombreado de los dos regímenes físicos hasta +3.2
        ax.axvspan(-12.5, -5.0, color=COLOR_CLUSTER_BG, alpha=0.7, zorder=0)
        ax.axvspan(-5.0, 3.5, color=COLOR_POISSON_BG, alpha=0.7, zorder=0)

        # 4. Histograma extendido sin truncar en 0
        counts, bins, patches = ax.hist(
            data_eta,
            bins=bins_edges,
            color=COLOR_BARS,
            edgecolor='white',
            linewidth=0.6,
            alpha=0.85,
            zorder=2
        )

        # 5. Ajuste continuo por Estimación de Densidad Kernel (KDE) extendido
        if len(data_eta) > 10:
            kde = gaussian_kde(data_eta, bw_method='scott')
            x_kde = np.linspace(-12.0, 3.0, 400)
            y_kde = kde(x_kde) * len(data_eta) * (bins_edges[1] - bins_edges[0])
            ax.plot(x_kde, y_kde, color=COLOR_KDE, lw=1.8, linestyle='-', zorder=3)

        # 6. Umbral principal (log10 eta_0 = -5.0) y referencia log10 eta_0 = 0.0
        ax.axvline(x=-5.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=1.8, zorder=4)
        ax.axvline(x=0.0, color=COLOR_SEC_THRES, linestyle=':', linewidth=1.4, zorder=4)

        # 7. Tarjeta informativa con el tamaño de muestra N
        badge_text = r'$m \geq ' + f'{int(m_cut)}' + r'$' + f'\n$N = {N_obs:,}$'
        ax.text(
            0.03, 0.72,
            badge_text,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BDC3C7", lw=0.8, alpha=0.95),
            zorder=5
        )

        # Etiquetado temático de regiones solo en el panel superior
        if idx == 0:
            max_y = max(counts)
            ax.text(-8.5, max_y * 0.88, 'Población Agrupada\n(Réplicas / Enjambres)', 
                    fontsize=9, fontweight='bold', color='#A93226', ha='center', zorder=5)
            ax.text(-1.0, max_y * 0.88, 'Población Independiente\n(Sismicidad de Fondo)', 
                    fontsize=9, fontweight='bold', color='#1E8449', ha='center', zorder=5)

        # Formato de ejes
        ax.set_ylabel('Número de eventos', fontsize=10, fontweight='bold', labelpad=8)
        ax.grid(True, which='major', linestyle=':', color='#B0BEC5', alpha=0.6, zorder=1)
        
        # LÍMITE DEL EJE X AMPLIADO
        ax.set_xlim(-12.5, 3.2)

        # Ocultar espinas superior y derecha
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#7F8C8D')
        ax.spines['bottom'].set_color('#7F8C8D')

    # Configuración del eje X global y título principal
    axes[2].set_xlabel(r'Distancia al vecino más cercano, $\log_{10}\eta^*$', fontsize=11, fontweight='bold', labelpad=8)

    fig.suptitle(
        r'Figura 3: Distribución Bimodal de $\log_{10}\eta^*$ en Rango Ampliado $[-12, +3]$' + '\n' +
        r'Catálogo del Sur de California (SCEDC) | Umbral de Corte $\log_{10}\eta_0 = -5.0$',
        fontsize=12, fontweight='bold', color='#2C3E50', y=0.98
    )

    plt.subplots_adjust(hspace=0.18)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[ÉXITO] Figura 3 ampliada guardada como '{output_filename}'")
    plt.close()


if __name__ == "__main__":
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        generar_figura3_profesional_completa(df_raw)
    else:
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")