import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from main import buscar_archivo_catalogo, cargar_catalogo_scedc
from Zaliapin import calcular_zaliapin_nn


def encontrar_valle_bimodal(log10_eta_vals, rango=(-4.5, -1.0)):
    """
    Calcula automáticamente el mínimo local (valle) entre la población
    agrupada (réplicas) y la población independiente (fondo).
    """
    vals = log10_eta_vals[(log10_eta_vals >= -8.0) & (log10_eta_vals <= 2.0)]
    if len(vals) < 50:
        return -2.4

    kde = gaussian_kde(vals, bw_method=0.18)
    grid = np.linspace(rango[0], rango[1], 400)
    densidad = kde(grid)

    # Buscar mínimos locales invirtiendo la densidad
    minimos, _ = find_peaks(-densidad)
    if len(minimos) > 0:
        # Elegir el mínimo más profundo dentro del rango del valle
        idx_min = minimos[np.argmin(densidad[minimos])]
        return round(float(grid[idx_min]), 2)

    return -2.4


def generar_figura_4_TR(df_raw, output_filename="figura_4_zaliapin_TFG.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5), dpi=300)
    fig.suptitle(
        "Distribución Bimodal de Componentes Espacio-Temporales $(T, R)$\n"
        "Catálogo SCEDC (1984 – 2026)",
        fontsize=14, fontweight='bold', y=0.98
    )

    configuraciones = [
        {"ax": axes[0, 0], "letra": "(a)", "m0": 3.0, "d": 1.6, "kde": True},
        {"ax": axes[0, 1], "letra": "(b)", "m0": 5.0, "d": 1.6, "kde": False},
        {"ax": axes[1, 0], "letra": "(c)", "m0": 3.0, "d": 1.0, "kde": True},
        {"ax": axes[1, 1], "letra": "(d)", "m0": 3.0, "d": 2.0, "kde": True},
    ]

    x_line = np.linspace(-9, 0, 200)
    umbral_referencia = None

    for cfg in configuraciones:
        ax = cfg["ax"]
        m0 = cfg["m0"]
        d = cfg["d"]

        df_sub = df_raw[df_raw['mag'] >= m0].copy().reset_index(drop=True)
        df_sub = calcular_zaliapin_nn(df_sub, b=1.0, d=d, m0=m0)
        df_val = df_sub.dropna(subset=['log10_T', 'log10_R', 'log10_eta'])

        N_ev = len(df_val)

        # Calcular el umbral exacto del valle para d=1.6, d=1.0, d=2.0
        if cfg["kde"]:
            umbral_eta = encontrar_valle_bimodal(df_val['log10_eta'].values)
            if cfg["letra"] == "(a)":
                umbral_referencia = umbral_eta
        else:
            # En (b) m0=5.0 hay pocos eventos; se usa el mismo umbral que en (a)
            umbral_eta = umbral_referencia if umbral_referencia is not None else -2.4

        if cfg["kde"] and N_ev > 10:
            sns.kdeplot(
                data=df_val, x='log10_T', y='log10_R',
                fill=True, cmap="Greys", levels=14, thresh=0.05, ax=ax
            )
            sns.kdeplot(
                data=df_val, x='log10_T', y='log10_R',
                color="#333333", linewidths=0.4, levels=14, thresh=0.05, ax=ax
            )
        else:
            ax.scatter(
                df_val['log10_T'], df_val['log10_R'],
                facecolors='none', edgecolors='black', s=35, linewidths=0.8
            )

        # Ecuación de la recta separadora: log10(R) = log10(eta_0) - log10(T)
        y_line = umbral_eta - x_line
        ax.plot(
            x_line, y_line, color='black', linestyle='-', linewidth=2.0,
            label=rf'$\log_{{10}}\eta_0 = {umbral_eta:.2f}$', zorder=10
        )

        ax.set_xlim(-9, 0)
        ax.set_ylim(-5, 3)
        ax.set_xlabel(r"Tiempo reescalado, $\log_{10} T$", fontsize=11)
        ax.set_ylabel(r"Distancia reescalada, $\log_{10} R$", fontsize=11)
        ax.set_title(rf"$\mathbf{{{cfg['letra']}}}$ $m_0 = {m0:.1f}$,  $d = {d:.1f}$", fontsize=12, pad=8)
        ax.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)

        ax.text(
            0.03, 0.95, f"$N = {N_ev:,}$", transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9)
        )
        ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_filename, dpi=300)
    print(f"[ÉXITO] Figura guardada en '{output_filename}' con umbral valle = {umbral_referencia:.2f}")
    plt.show()


if __name__ == "__main__":
    ruta = buscar_archivo_catalogo()
    if os.path.exists(ruta):
        df_cat = cargar_catalogo_scedc(ruta)
        generar_figura_4_TR(df_cat)