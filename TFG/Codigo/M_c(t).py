import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from GR_Mc import estimar_mc_gft
  # Si guardas el archivo anterior como G-R_Mc.py, usa la función incluida abajo


def cargar_catalogo_temporal(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('YYYY') or line.startswith('SCSN'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    dt = pd.to_datetime(f"{parts[0]} {parts[1]}")
                    mag = float(parts[4])
                    records.append({'datetime': dt, 'mag': mag})
                except ValueError:
                    continue
    return pd.DataFrame(records).dropna().sort_values('datetime').reset_index(drop=True)


def calcular_mc_ventana_gft(mags, bin_width=0.1, threshold_r=10.0):
    """Calcula Mc mediante Goodness-of-Fit (Wiemer & Wyss, 2000) para una ventana temporal."""
    min_m = np.floor(mags.min() * 10) / 10
    max_m = np.ceil(mags.max() * 10) / 10
    bins = np.arange(min_m - bin_width / 2.0, max_m + 1.5 * bin_width, bin_width)
    counts, bin_edges = np.histogram(mags, bins=bins)
    bin_centers = np.round(0.5 * (bin_edges[:-1] + bin_edges[1:]), 2)

    mc_maxc = bin_centers[np.argmax(counts)]
    candidatos = bin_centers[(bin_centers >= mc_maxc - 0.1) & (bin_centers <= max_m - 0.8)]

    for mc_cand in candidatos:
        mags_comp = mags[mags >= mc_cand - bin_width / 2.0]
        n_comp = len(mags_comp)
        if n_comp < 30:
            continue
        denom = np.mean(mags_comp) - (mc_cand - bin_width / 2.0)
        if denom <= 0:
            continue

        b_val = np.log10(np.e) / denom
        a_val = np.log10(n_comp) + b_val * mc_cand

        m_eval = bin_centers[bin_centers >= mc_cand]
        obs_cum = np.array([np.sum(mags >= m - bin_width / 2.0) for m in m_eval])
        pred_cum = 10.0 ** (a_val - b_val * m_eval)

        R = (np.sum(np.abs(obs_cum - pred_cum)) / np.sum(obs_cum)) * 100.0
        if R <= threshold_r:
            return float(mc_cand)

    return round(float(mc_maxc + 0.2), 2)


def estimar_mc_bootstrap(mags, n_boot=30):
    rng = np.random.default_rng(42)
    n = len(mags)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(mags, size=n, replace=True)
        vals.append(calcular_mc_ventana_gft(sample))
    return float(np.mean(vals)), float(np.std(vals))


def graficar_mc_temporal(df, window_size=3000, step=1200, output_fig="figura_Mc_vs_tiempo.png"):
    fechas = []
    mc_vals = []
    mc_stds = []

    print(f"Calculando evolución temporal de Mc por Goodness-of-Fit (ventana = {window_size} sismos)...")
    for i in range(0, len(df) - window_size, step):
        sub = df.iloc[i : i + window_size]
        mc_mean, mc_std = estimar_mc_bootstrap(sub['mag'].values, n_boot=30)
        fechas.append(sub['datetime'].iloc[window_size // 2])
        mc_vals.append(mc_mean)
        mc_stds.append(mc_std)

    fechas = np.array(fechas)
    mc_vals = np.array(mc_vals)
    mc_stds = np.array(mc_stds)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    ax.fill_between(
        fechas, mc_vals - mc_stds, mc_vals + mc_stds,
        color='#AED6F1', alpha=0.5, label=r'Incertidumbre $\pm 1\sigma$ (Bootstrap)'
    )

    ax.plot(
        fechas, mc_vals, color='#1F618D', lw=1.6, marker='o', markersize=2.5,
        label=rf'Magnitud de Completitud $M_c(t)$ (GFT, $N_w={window_size}$)'
    )

    mc_global = np.median(mc_vals)
    ax.axhline(
        mc_global, color='#27AE60', linestyle='--', lw=1.6,
        label=rf'Mediana Temporal ($M_c = {mc_global:.2f}$)'
    )

    ax.set_xlabel("Tiempo (Año)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Magnitud de Completitud ($M_c$)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Evolución Temporal de la Magnitud de Completitud $M_c(t)$\n"
        "Catálogo del Sur de California (SCEDC) | Método Goodness-of-Fit (Wiemer & Wyss, 2000)",
        fontsize=12, fontweight="bold", color="#2C3E50", pad=12
    )

    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim(0.5, 3.2)
    ax.grid(True, linestyle=':', color='#BDC3C7', alpha=0.8)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"[ÉXITO] Gráfica temporal de Mc guardada en '{output_fig}' | Mediana Mc = {mc_global:.2f}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(script_dir, "SearchResults.txt")
    if os.path.exists(ruta):
        df_cat = cargar_catalogo_temporal(ruta)
        print(f"Eventos cargados: {len(df_cat):,}")
        graficar_mc_temporal(df_cat, window_size=3000, step=1200)
    else:
        print(f"[ERROR] No se encontró '{ruta}'.")
