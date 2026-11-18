import os
import matplotlib.pyplot as plt
import numpy as np


def fit_gutenberg_richter_mle(mags, Mc, bin_width=0.1):
    """Ajusta a y b mediante Máxima Verosimilitud (Aki, 1965; Utsu, 1965)."""
    mags_complete = mags[mags >= Mc - bin_width / 2.0]
    N_total = len(mags_complete)

    if N_total < 10:
        return np.nan, np.nan, np.nan, np.nan, N_total

    mean_mag = np.mean(mags_complete)
    denom = mean_mag - (Mc - bin_width / 2.0)
    if denom <= 0:
        return np.nan, np.nan, np.nan, np.nan, N_total

    b_val = np.log10(np.e) / denom
    b_std = b_val / np.sqrt(N_total)
    a_val = np.log10(N_total) + b_val * Mc
    a_std = np.sqrt(((np.log10(np.e)) ** 2 / N_total) + (Mc * b_std) ** 2)

    return a_val, b_val, a_std, b_std, N_total


def estimar_mc_gft(mags, bin_width=0.1, threshold_r=10.0):
    """
    Calcula la Magnitud de Completitud (Mc) mediante el método
    Goodness-of-Fit Test (GFT) de Wiemer & Wyss (2000).
    
    Busca la menor magnitud candidata donde el residuo absoluto R (%)
    entre la distribución acumulada observada y la ley de Gutenberg-Richter
    sea <= threshold_r (por defecto 10%, equivalente a 90% de bondad de ajuste).
    """
    mags = np.asarray(mags)
    mags = mags[~np.isnan(mags)]

    min_m = np.floor(mags.min() * 10) / 10
    max_m = np.ceil(mags.max() * 10) / 10
    bins = np.arange(min_m - bin_width / 2.0, max_m + 1.5 * bin_width, bin_width)
    counts, bin_edges = np.histogram(mags, bins=bins)
    bin_centers = np.round(0.5 * (bin_edges[:-1] + bin_edges[1:]), 2)

    # Punto de partida: máxima curvatura (MAXC)
    mc_maxc = bin_centers[np.argmax(counts)]
    candidatos = bin_centers[(bin_centers >= mc_maxc - 0.2) & (bin_centers <= max_m - 1.0)]

    for mc_cand in candidatos:
        a_val, b_val, _, _, n_comp = fit_gutenberg_richter_mle(mags, mc_cand, bin_width)
        if np.isnan(b_val) or n_comp < 30:
            continue

        m_eval = bin_centers[bin_centers >= mc_cand]
        obs_cum = np.array([np.sum(mags >= m - bin_width / 2.0) for m in m_eval])
        pred_cum = 10.0 ** (a_val - b_val * m_eval)

        # Residuo relativo porcentual R (Wiemer & Wyss, 2000)
        R = (np.sum(np.abs(obs_cum - pred_cum)) / np.sum(obs_cum)) * 100.0
        if R <= threshold_r:
            return float(mc_cand), float(R)

    # Respaldo estándar en sismología: MAXC + 0.2 (Woessner & Wiemer, 2005)
    return round(float(mc_maxc + 0.2), 2), np.nan


def graficar_gutenberg_richter(
    magnitudes,
    bin_width=0.1,
    output_fig="figura_gutenberg_richter.png",
):
    mags = np.asarray(magnitudes)
    mags = mags[~np.isnan(mags)]

    # Cálculo automático de Mc
    Mc, residuo_R = estimar_mc_gft(mags, bin_width=bin_width, threshold_r=10.0)
    a_val, b_val, a_std, b_std, N_comp = fit_gutenberg_richter_mle(
        mags, Mc=Mc, bin_width=bin_width
    )

    min_m, max_m = np.floor(mags.min() * 10) / 10, np.ceil(mags.max() * 10) / 10
    bins = np.arange(min_m - bin_width / 2.0, max_m + 1.5 * bin_width, bin_width)
    counts, bin_edges = np.histogram(mags, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cum_counts = np.array([np.sum(mags >= m - bin_width / 2.0) for m in bin_centers])

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    ax.scatter(
        bin_centers[counts > 0], counts[counts > 0],
        facecolors='none', edgecolors='#95A5A6', s=28, linewidths=0.8,
        label=r"Frecuencia Discreta ($\Delta M = 0.1$)", zorder=2
    )
    ax.scatter(
        bin_centers[cum_counts > 0], cum_counts[cum_counts > 0],
        color="#1F618D", s=38, edgecolors="#154360", linewidths=0.6,
        label=r"Frecuencia Acumulada Observada $N(\geq M)$", zorder=3
    )

    m_fit = np.linspace(Mc, max_m, 100)
    N_fit = 10 ** (a_val - b_val * m_fit)
    ax.plot(
        m_fit, N_fit, color="#E74C3C", lw=2.4, linestyle="-",
        label=(
            f"Ajuste MLE ($M \\geq {Mc:.2f}$)\n"
            f"$\\log_{{10}} N = {a_val:.2f} - {b_val:.3f}M$\n"
            f"$b = {b_val:.3f} \\pm {b_std:.3f}$ ($N = {N_comp:,}$)"
        ),
        zorder=4
    )

    ax.axvline(
        x=Mc, color="#27AE60", linestyle="--", lw=1.8,
        label=rf"Magnitud de Completitud Calculada ($M_c = {Mc:.2f}$)", zorder=5
    )

    ax.set_yscale("log")
    ax.set_xlim(-1.05, max_m + 0.15)
    ax.set_ylim(0.8, max(cum_counts) * 2.5)
    ax.set_xlabel(r"Magnitud ($M$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Número Acumulado de Eventos $N(\geq M)$", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Distribución Magnitud-Frecuencia de Gutenberg-Richter\n"
        f"Catálogo SCEDC | $M_c = {Mc:.2f}$ (GFT) | $b = {b_val:.3f} \\pm {b_std:.3f}$",
        fontsize=12.5, fontweight="bold", color="#2C3E50", pad=12
    )
    ax.grid(True, which="both", linestyle=":", color="#BDC3C7", alpha=0.7)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"[ÉXITO] Gráfica guardada en '{output_fig}' | Mc calculada = {Mc:.2f} (Residuo R = {residuo_R:.2f}%), b = {b_val:.3f}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archivo = os.path.join(script_dir, "SearchResults.txt")
    if os.path.exists(archivo):
        mags = []
        with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#") or line.startswith("<") or line.startswith("SCSN") or line.startswith("YYYY"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        mags.append(float(parts[4]))
                    except ValueError:
                        pass
        graficar_gutenberg_richter(np.array(mags))