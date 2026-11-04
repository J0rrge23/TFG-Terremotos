import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cargar_magnitudes_fechas(filepath):
    """Carga Fechas y Magnitudes del catálogo SCEDC."""
    records = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('YYYY'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    dt = pd.to_datetime(f"{parts[0]} {parts[1]}")
                    mag = float(parts[4])
                    records.append({'datetime': dt, 'mag': mag})
                except ValueError:
                    continue
    df = pd.DataFrame(records).sort_values('datetime').reset_index(drop=True)
    return df

def estimar_mc_maxc_corregido(mags, bin_width=0.1, correction=0.2):
    """
    Estima Mc usando Máxima Curvatura (MAXC) con corrección empírica (+0.2).
    Evita la subestimación sistemática del punto de completitud lineal.
    """
    if len(mags) < 100: 
        return np.nan
    bins = np.arange(0.0, 8.0, bin_width)
    counts, bin_edges = np.histogram(mags, bins=bins)
    max_idx = np.argmax(counts)
    return bin_edges[max_idx] + (bin_width / 2.0) + correction

def estimar_mc_gft(mags, bin_width=0.1, R_target=0.90):
    """
    Estima Mc mediante el Test de Bondad de Ajuste (Goodness-of-Fit Test, GFT).
    Evalúa el ajuste R^2 del modelo Gutenberg-Richter acumulado.
    """
    if len(mags) < 100:
        return np.nan
        
    m_cands = np.arange(0.5, 3.5, bin_width)
    for mc in m_cands:
        m_sub = mags[mags >= mc]
        if len(m_sub) < 50:
            break
            
        # Estimador de Máxima Verosimilitud de b (Aki, 1965)
        b_val = (1.0 / (np.mean(m_sub) - (mc - bin_width/2.0))) * np.log10(np.e)
        a_val = np.log10(len(m_sub)) + b_val * mc
        
        # Frecuencias observadas vs teóricas
        bins = np.arange(mc, 8.0, bin_width)
        obs_counts, _ = np.histogram(m_sub, bins=bins)
        cum_obs = np.cumsum(obs_counts[::-1])[::-1]
        
        m_centers = bins[:-1] + bin_width/2.0
        cum_pred = 10**(a_val - b_val * m_centers)
        
        # Coeficiente R^2 de ajuste
        res = np.sum((cum_obs - cum_pred)**2)
        tot = np.sum((cum_obs - np.mean(cum_obs))**2)
        r2 = 1.0 - (res / (tot + 1e-10))
        
        if r2 >= R_target:
            return round(mc, 2)
            
    return estimar_mc_maxc_corregido(mags, bin_width, correction=0.2)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    ruta = os.path.join(script_dir, "SearchResults.txt")
    
    if not os.path.exists(ruta):
        print(f"[ERROR] Archivo no encontrado en: {ruta}")
        return
        
    print("Cargando catálogo sísmico...")
    df = cargar_magnitudes_fechas(ruta)

    # Parámetros de la ventana deslizante
    window_size = 2000  # Tamaño aumentado para mayor estabilidad estadística
    step = 400
    
    fechas_centro = []
    mc_maxc_corr = []
    mc_gft = []

    print(f"Calculando evolución de Mc (Ventana = {window_size} eventos)...")
    for i in range(0, len(df) - window_size, step):
        ventana = df.iloc[i:i+window_size]
        mags_v = ventana['mag'].values
        
        mc_m = estimar_mc_maxc_corregido(mags_v, correction=0.2)
        mc_g = estimar_mc_gft(mags_v, R_target=0.90)
        
        fechas_centro.append(ventana['datetime'].iloc[int(window_size/2)])
        mc_maxc_corr.append(mc_m)
        mc_gft.append(mc_g)

    df_res = pd.DataFrame({
        'fecha': fechas_centro, 
        'mc_maxc': mc_maxc_corr, 
        'mc_gft': mc_gft
    }).dropna()

    # Suavizado por mediana móvil
    df_res['mc_smooth'] = df_res['mc_maxc'].rolling(window=5, center=True, min_periods=1).median()

    # Visualización profesional
    plt.figure(figsize=(11, 5.5), dpi=300)
    
    # Valores MAXC Corregido
    plt.plot(df_res['fecha'], df_res['mc_maxc'], color='#BDC3C7', linestyle='--', 
             linewidth=0.8, alpha=0.7, label=r'MAXC bruto corregido ($M_{MAXC} + 0.2$)')
    
    # Tendencia Suavizada (Mediana)
    plt.plot(df_res['fecha'], df_res['mc_smooth'], color='#2980B9', linestyle='-', 
             linewidth=2.0, label=r'Evolución $M_c(t)$ (Mediana móvil)')

    # Línea de Referencia Consistente para Gutenberg-Richter
    mc_global_recomendado = 1.3
    plt.axhline(mc_global_recomendado, color='#E74C3C', linestyle='--', linewidth=1.8, 
                label=f'Umbral global de completitud ($M_c = {mc_global_recomendado}$)')

    plt.xlabel('Año', fontsize=11, fontweight='bold')
    plt.ylabel(r'Magnitud de Completitud ($M_c$)', fontsize=11, fontweight='bold')
    plt.title('Evolución Temporal de la Magnitud de Completitud $M_c(t)$\nRed Sísmica del Sur de California (SCEDC 1984–2026)', 
              fontsize=12, fontweight='bold', pad=10)
    
    plt.ylim(0.4, 2.8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    plt.tight_layout()
    
    output_filename = 'figura_Mc_vs_tiempo_corregida.png'
    plt.savefig(output_filename, dpi=300)
    print(f"[ÉXITO] Gráfica guardada como '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()