# =====================================================================
# DEMOSTRACIÓN DE LA LEY DE OMORI EN 3 ÁRBOLES GENERADOS DE ZALIAPIN
# =====================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Importar funciones locales del proyecto
try:
    from clustering import identificar_clusters
    from main import buscar_archivo_catalogo, cargar_catalogo_scedc
    from seismology_laws import fit_gutenberg_richter
    from Zaliapin import calcular_zaliapin_nn
except ImportError:
    pass


# =====================================================================
# 1. MODELOS MATEMÁTICOS DE OMORI-UTSU
# =====================================================================
def omori_rate(t, K, c, p):
    """Tasa de réplicas n(t) = K / (t + c)^p"""
    return K / np.power(t + c, p)


def omori_cumulative(t, K, c, p):
    """Número acumulado de réplicas N(t) dentro del árbol"""
    if np.abs(p - 1.0) < 1e-3:
        return K * np.log(1.0 + t / c)
    else:
        return (K / (1.0 - p)) * (
            np.power(t + c, 1.0 - p) - np.power(c, 1.0 - p)
        )


# =====================================================================
# 2. EXTRACCIÓN Y AJUSTE DE LOS 3 ÁRBOLES DE ZALIAPIN
# =====================================================================
def analizar_omori_3_arboles_zaliapin(
    df_proc,
    top_n=3,
    t_max_days=120.0,
    output_fig='figura_omori_3_arboles.png',
):
    """Aísla los 3 árboles de desencadenamiento mayores en el grafo de Zaliapin,

    calcula la evolución temporal de sus nodos hijos y demuestra la Ley de
    Omori.
    """
    df_proc = df_proc.copy()

    # -----------------------------------------------------------------
    # ESTANDARIZACIÓN AUTOMÁTICA DE COLUMNAS (Mantiene compatibilidad)
    # -----------------------------------------------------------------
    # 1. Eliminar espacios invisibles en nombres de columnas
    df_proc.columns = df_proc.columns.str.strip()

    # 2. Si la fecha es el índice, resetearlo
    if (
        'time' not in df_proc.columns
        and df_proc.index.name
        and df_proc.index.name.lower() in ['time', 'date', 'datetime']
    ):
        df_proc = df_proc.reset_index()

    # 3. Mapear variantes comunes de nombres a 'time' y 'mag'
    renombres = {}
    for col in df_proc.columns:
        c_low = col.lower()
        if c_low in [
            'time',
            'date',
            'datetime',
            'origintime',
            'fecha',
            'tiempo',
            'yyy/mm/dd',
        ]:
            renombres[col] = 'time'
        elif c_low in ['mag', 'magnitude', 'mw', 'mb', 'ms', 'mag1']:
            renombres[col] = 'mag'

    df_proc = df_proc.rename(columns=renombres)

    # Validar que la columna 'time' exista tras el mapeo
    if 'time' not in df_proc.columns:
        raise KeyError(
            f"No se encontró columna de tiempo en el DataFrame. Columnas disponibles: {list(df_proc.columns)}"
        )

    # Asegurar formato datetime
    df_proc['time'] = pd.to_datetime(df_proc['time'])

    # -----------------------------------------------------------------
    # PROCESAMIENTO DE LOS ÁRBOLES
    # -----------------------------------------------------------------
    valid_clusters = df_proc[df_proc['cluster_id'] != -1]
    if valid_clusters.empty:
        print('[ERROR] No se encontraron clústeres válidos en el catálogo.')
        return pd.DataFrame()

    counts = valid_clusters['cluster_id'].value_counts()
    top_cluster_ids = counts.index[:top_n]
    num_arboles = len(top_cluster_ids)

    fig, axes = plt.subplots(
        num_arboles, 2, figsize=(12, 3.8 * num_arboles), dpi=300, squeeze=False
    )
    fig.patch.set_facecolor('white')

    print(
        '\n====================================================================='
    )
    print(
        f' DEMOSTRACIÓN DE OMORI-UTSU EN LOS {num_arboles} ÁRBOLES DE ZALIAPIN'
        ' MÁS GRANDES'
    )
    print(
        '====================================================================='
    )

    resultados_arboles = []

    for i, cid in enumerate(top_cluster_ids):
        df_tree = df_proc[df_proc['cluster_id'] == cid].copy()

        # Identificar el nodo raíz (Mainshock)
        df_ms = (
            df_tree[df_tree['event_type'] == 'mainshock']
            if 'event_type' in df_tree.columns
            else pd.DataFrame()
        )
        if not df_ms.empty:
            ms_row = df_ms.iloc[0]
        else:
            idx_max = df_tree['mag'].idxmax()
            ms_row = df_tree.loc[idx_max]

        # Acceso seguro a 'time' y 'mag'
        t_root = ms_row['time']
        mag_root = ms_row['mag']

        # Filtrar réplicas
        if 'event_type' in df_tree.columns:
            df_children = df_tree[
                (df_tree['event_type'] == 'aftershock')
                & (df_tree['time'] > t_root)
            ].copy()
        else:
            df_children = df_tree[df_tree['time'] > t_root].copy()

        df_children['dt_days'] = (
            df_children['time'] - t_root
        ).dt.total_seconds() / 86400.0

        # Ventana de tiempo
        df_children = df_children[
            (df_children['dt_days'] > 1e-4)
            & (df_children['dt_days'] <= t_max_days)
        ]

        times = np.sort(df_children['dt_days'].values)
        N_total = len(times)

        if N_total == 0:
            print(
                f'[Aviso] Árbol #{i+1} (Cluster ID: {cid}) sin réplicas'
                ' suficientes en la ventana de tiempo.'
            )
            continue

        N_cum_obs = np.arange(1, N_total + 1)

        p0 = [max(N_total / 5.0, 1.0), 0.01, 1.0]
        bounds = ([0.1, 1e-4, 0.2], [1e6, 5.0, 2.5])

        try:
            popt, pcov = curve_fit(
                omori_cumulative,
                times,
                N_cum_obs,
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
            K_fit, c_fit, p_fit = popt
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            print(f'[Aviso] Fallo en convergencia para el Árbol #{i+1}: {e}')
            K_fit, c_fit, p_fit = p0
            perr = [0.0, 0.0, 0.0]

        resultados_arboles.append({
            'arbol_id': i + 1,
            'cluster_id': cid,
            'mainshock_mag': mag_root,
            'n_nodos': N_total,
            'K': K_fit,
            'c': c_fit,
            'p': p_fit,
            'p_err': perr[2],
        })

        t_grid = np.logspace(-3, np.log10(t_max_days), 400)
        N_cum_model = [
            omori_cumulative(tg, K_fit, c_fit, p_fit) for tg in t_grid
        ]

        # -----------------------------------------------------------------
        # DIBUJAR PANELES
        # -----------------------------------------------------------------
        ax_cum = axes[i, 0]
        ax_cum.step(
            times,
            N_cum_obs,
            where='post',
            color='#2980B9',
            lw=1.5,
            label=f'Nodos del Árbol #{i+1} ($N={N_total}$)',
        )
        ax_cum.plot(
            t_grid,
            N_cum_model,
            color='#C0392B',
            lw=1.8,
            linestyle='--',
            label=(
                'Modelo Omori-Utsu\n'
                f'$K={K_fit:.1f}$, $c={c_fit:.3f}$ d,'
                f' $p={p_fit:.2f}\\pm{perr[2]:.2f}$'
            ),
        )

        ax_cum.set_xscale('log')
        ax_cum.set_xlabel(
            'Tiempo transcurrido $t - t_{root}$ (días)',
            fontsize=9.5,
            fontweight='bold',
        )
        ax_cum.set_ylabel(
            'Número Acumulado $N(t)$', fontsize=9.5, fontweight='bold'
        )
        ax_cum.set_title(
            f'Árbol #{i+1}: Raíz M{mag_root:.1f} (Cluster ID: {cid})',
            fontsize=10.5,
            fontweight='bold',
            color='#2C3E50',
        )
        ax_cum.grid(
            True, which='both', linestyle=':', color='#BDC3C7', alpha=0.7
        )
        ax_cum.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

        ax_rate = axes[i, 1]
        bins = np.logspace(-3, np.log10(t_max_days), 16)
        counts, bin_edges = np.histogram(times, bins=bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        rates_obs = counts / bin_widths

        valid = rates_obs > 0
        ax_rate.scatter(
            bin_centers[valid],
            rates_obs[valid],
            color='#2980B9',
            s=35,
            zorder=3,
            edgecolors='#1B4F72',
            lw=0.7,
            label='Tasa por Bin $n(t)$',
        )
        ax_rate.plot(
            t_grid,
            omori_rate(t_grid, K_fit, c_fit, p_fit),
            color='#C0392B',
            lw=1.8,
            label=(
                f'$n(t) = \\frac{{{K_fit:.1f}}}{{(t +'
                f' {c_fit:.3f})^{{{p_fit:.2f}}}}}$'
            ),
        )

        ax_rate.set_xscale('log')
        ax_rate.set_yscale('log')
        ax_rate.set_xlabel(
            'Tiempo transcurrido $t - t_{root}$ (días)',
            fontsize=9.5,
            fontweight='bold',
        )
        ax_rate.set_ylabel(
            'Tasa de Réplicas (eventos/día)', fontsize=9.5, fontweight='bold'
        )
        ax_rate.set_title(
            f'Decaimiento Temporal del Árbol #{i+1} ($p = {p_fit:.2f}$)',
            fontsize=10.5,
            fontweight='bold',
            color='#2C3E50',
        )
        ax_rate.grid(
            True, which='both', linestyle=':', color='#BDC3C7', alpha=0.7
        )
        ax_rate.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

        print(
            f'Árbol #{i+1} (Mainshock M{mag_root:.1f}): N = {N_total} | K ='
            f' {K_fit:.2f} | c = {c_fit:.4f} d | p = {p_fit:.2f} ± {perr[2]:.2f}'
        )

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(
        f"\n[ÉXITO] Figura de Omori para {num_arboles} árboles guardada como"
        f" '{output_fig}'"
    )
    plt.close()

    return pd.DataFrame(resultados_arboles)


# =====================================================================
# 3. EJECUCIÓN DEL FLUJO COMPLETO
# =====================================================================
if __name__ == '__main__':
    path = buscar_archivo_catalogo()
    if os.path.exists(path):
        df_raw = cargar_catalogo_scedc(path)
        Mc = 3.0

        print('\n[1] Filtrando catálogo y estimando b-value inicial...')
        df_proc = df_raw[df_raw['mag'] >= Mc].copy().reset_index(drop=True)
        a_cat, b_cat = fit_gutenberg_richter(df_raw['mag'].values, Mc=Mc)
        b_val = b_cat if not np.isnan(b_cat) else 1.0

        print('[2] Construyendo árbol de Zaliapin y clasificando clústeres...')
        df_proc = calcular_zaliapin_nn(df_proc, b=b_val, d=1.6, m0=Mc)
        df_proc = identificar_clusters(df_proc, log10_eta_0=0.0)

        print(
            '[3] Generando demostración de la Ley de Omori en los 3 árboles'
            ' mayores...'
        )
        df_res = analizar_omori_3_arboles_zaliapin(
            df_proc, top_n=3, t_max_days=120.0
        )
    else:
        print(f"[ERROR] No se encontró el catálogo en '{path}'.")