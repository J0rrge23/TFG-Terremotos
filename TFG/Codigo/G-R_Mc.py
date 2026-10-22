# =====================================================================
# CÁLCULO Y GRAFICACIÓN DE LA LEY DE GUTENBERG-RICHTER (G-R_Mc.py)
# =====================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# 1. ESTIMACIÓN AUTOMÁTICA Y DINÁMICA DE Mc
# =====================================================================
def estimar_mc_maxc(mags, bin_width=0.1, correction=0.0):
    """Estima Mc mediante el método de Máxima Curvatura (MAXC).

    Identifica el bin con la mayor frecuencia de eventos utilizando
    bins alineados en pasos de 0.1.
    """
    min_m = np.floor(mags.min() * 10) / 10
    max_m = np.ceil(mags.max() * 10) / 10
    bins = np.arange(min_m - bin_width / 2.0, max_m + 1.5 * bin_width, bin_width)

    counts, bin_edges = np.histogram(mags, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pico de frecuencia no acumulativa (MAXC)
    idx_max = np.argmax(counts)
    mc_estimada = bin_centers[idx_max] + correction

    return round(float(mc_estimada), 2)


def estimar_mc_gft(mags, bin_width=0.1, threshold=0.90):
    """Estima Mc mediante Goodness-of-Fit (GFT) evaluando sobre bins alineados."""
    min_m = np.floor(mags.min() * 10) / 10
    max_m = np.ceil(mags.max() * 10) / 10
    bins = np.arange(min_m - bin_width / 2.0, max_m + 1.5 * bin_width, bin_width)

    counts, bin_edges = np.histogram(mags, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    candidate_indices = np.where(counts >= 20)[0]
    best_mc = bin_centers[np.argmax(counts)]

    for idx in candidate_indices:
        mc_cand = bin_centers[idx]
        mags_sub = mags[mags >= mc_cand - bin_width / 2.0]
        n_total = len(mags_sub)

        if n_total < 30:
            continue

        # Estimación Aki-Utsu para el candidato
        mean_mag = np.mean(mags_sub)
        b_cand = np.log10(np.e) / (mean_mag - (mc_cand - bin_width / 2.0))
        a_cand = np.log10(n_total) + b_cand * mc_cand

        # Comparación acumulada observada vs sintética
        eval_m = bin_centers[bin_centers >= mc_cand]
        obs_cum = np.array([np.sum(mags >= m - bin_width / 2.0) for m in eval_m])
        syn_cum = 10 ** (a_cand - b_cand * eval_m)

        diff = np.sum(np.abs(obs_cum - syn_cum))
        sum_obs = np.sum(obs_cum)

        if sum_obs == 0:
            continue

        gft_value = 1.0 - (diff / sum_obs)
        if gft_value >= threshold:
            best_mc = mc_cand
            break

    return round(float(best_mc), 2)


# =====================================================================
# 2. AJUSTE DE GUTENBERG-RICHTER (Aki-Utsu MLE)
# =====================================================================
def fit_gutenberg_richter(
    magnitudes, Mc=None, bin_width=0.1, metodo_mc="maxc"
):
    """Calcula 'a' y 'b' de Gutenberg-Richter: log10(N) = a - b*M."""
    mags = np.asarray(magnitudes)
    mags = mags[~np.isnan(mags)]

    if len(mags) == 0:
        raise ValueError("El catálogo no contiene datos de magnitud válidos.")

    # 1. Estimación automática de Mc si es None
    if Mc is None:
        if str(metodo_mc).lower() == "gft":
            Mc = estimar_mc_gft(mags, bin_width=bin_width)
        else:
            Mc = estimar_mc_maxc(mags, bin_width=bin_width)

    # 2. Filtrado inclusivo del bin de completitud (M >= Mc - Delta_M / 2)
    mags_complete = mags[mags >= Mc - bin_width / 2.0]
    N_total = len(mags_complete)

    if N_total < 5:
        print(f"[Aviso] Muy pocos eventos ({N_total}) por encima de Mc = {Mc:.2f}.")
        return np.nan, np.nan, Mc, np.nan, np.nan

    # 3. Estimación del b-value (Aki-Utsu)
    mean_mag = np.mean(mags_complete)
    b_val = np.log10(np.e) / (mean_mag - (Mc - bin_width / 2.0))
    b_std = b_val / np.sqrt(N_total)

    # 4. Estimación del a-value
    a_val = np.log10(N_total) + b_val * Mc
    a_std = np.sqrt(((np.log10(np.e)) ** 2 / N_total) + (Mc * b_std) ** 2)

    return a_val, b_val, Mc, a_std, b_std


# =====================================================================
# 3. GRAFICACIÓN
# =====================================================================
def graficar_gutenberg_richter(
    magnitudes,
    Mc=None,
    bin_width=0.1,
    metodo_mc="maxc",
    output_fig="figura_gutenberg_richter.png",
):
    """Genera la gráfica de Gutenberg-Richter con alineación precisa."""
    mags = np.asarray(magnitudes)
    mags = mags[~np.isnan(mags)]

    a_val, b_val, Mc, a_std, b_std = fit_gutenberg_richter(
        mags, Mc=Mc, bin_width=bin_width, metodo_mc=metodo_mc
    )

    # Bins alineados exactamente en múltiplos de 0.1
    min_m = np.floor(mags.min() * 10) / 10
    max_m = np.ceil(mags.max() * 10) / 10
    bins = np.arange(min_m - bin_width / 2.0, max_m + 1.5 * bin_width, bin_width)

    counts, bin_edges = np.histogram(mags, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    cum_counts = np.array([np.sum(mags >= m - bin_width / 2.0) for m in bin_centers])

    valid_disc = counts > 0
    valid_cum = cum_counts > 0

    m_fit = np.linspace(Mc, max_m, 100)
    log_N_fit = a_val - b_val * m_fit
    N_fit = 10**log_N_fit

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("white")

    # Frecuencia discreta (Naranja)
    ax.scatter(
        bin_centers[valid_disc],
        counts[valid_disc],
        color="#E67E22",
        alpha=0.85,
        s=30,
        edgecolor="#D35400",
        lw=0.6,
        zorder=2,
        label=r"Frecuencia Discreta $n(M)$",
    )

    # Frecuencia acumulada (Azul)
    ax.scatter(
        bin_centers[valid_cum],
        cum_counts[valid_cum],
        color="#2980B9",
        s=35,
        edgecolor="#1B4F72",
        lw=0.7,
        zorder=3,
        label=r"Frecuencia Acumulada Observada $N(\geq M)$",
    )

    # Recta de ajuste Gutenberg-Richter
    label_fit = (
        r"Ajuste GR: $\log_{10} N = a - b M$" + "\n"
        rf"$a = {a_val:.2f} \pm {a_std:.2f}$, "
        rf"$b = {b_val:.2f} \pm {b_std:.2f}$" + "\n"
        rf"$M_c = {Mc:.2f}$ ($\Delta M = {bin_width}$)"
    )

    ax.plot(m_fit, N_fit, color="#C0392B", lw=2, linestyle="--", label=label_fit)

    # Línea vertical de Mc
    ax.axvline(
        x=Mc,
        color="#27AE60",
        linestyle=":",
        lw=1.5,
        label=rf"Completitud $M_c = {Mc:.2f}$",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"Magnitud ($M$)", fontsize=10.5, fontweight="bold", labelpad=6)
    ax.set_ylabel(
        r"Número Acumulado de Eventos $N(\geq M)$",
        fontsize=10.5,
        fontweight="bold",
        labelpad=6,
    )
    ax.set_title(
        r"Ley de Gutenberg-Richter",
        fontsize=12,
        fontweight="bold",
        color="#2C3E50",
    )

    ax.grid(True, which="both", linestyle=":", color="#BDC3C7", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"[ÉXITO] Gráfica guardada en '{output_fig}' | Mc detectada = {Mc:.2f}")
    plt.close()

    return a_val, b_val, Mc, a_std, b_std


# =====================================================================
# 4. EJECUCIÓN CON ARCHIVO DE CATÁLOGO (SearchResults.txt)
# =====================================================================
if __name__ == "__main__":
    mags = []
    archivo_txt = r"C:\Users\jorge\OneDrive\Desktop\TFG-Terremotos\TFG\Codigo\SearchResults.txt"
    # Cargar magnitudes del archivo de catálogo
    with open(archivo_txt, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("<") or line.startswith("SCSN"):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    mags.append(float(parts[4]))
                except ValueError:
                    pass

    mags = np.array(mags)

    # Ajuste mediante MAXC
    a, b, mc, a_err, b_err = graficar_gutenberg_richter(
        magnitudes=mags,
        Mc=None,
        bin_width=0.1,
        metodo_mc="maxc",
        output_fig="gutenberg_richter_corregido.png",
    )

    print(
        f"Resultados finales (MAXC): a = {a:.2f} ± {a_err:.2f} | b = {b:.2f} ± {b_err:.2f} | Mc = {mc:.2f}"
    )