import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cargar_magnitudes_fechas(filepath):
    """Carga solo Fechas y Magnitudes del catálogo para análisis de Mc temporal."""
    records = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('YYYY'):
                continue
            parts = line.split()
            if len(parts) >= 11:
                try:
                    dt = pd.to_datetime(f"{parts[0]} {parts[1]}")
                    mag = float(parts[4])
                    records.append({'datetime': dt, 'mag': mag})
                except ValueError:
                    continue
    df = pd.DataFrame(records).sort_values('datetime').reset_index(drop=True)
    return df

def estimar_mc_maxc(mags, bin_width=0.1):
    """Estima Mc usando el método de Máxima Curvatura (MAXC)."""
    if len(mags) == 0: return np.nan
    bins = np.arange(0.0, 8.0, bin_width)
    counts, bin_edges = np.histogram(mags, bins=bins)
    return bin_edges[np.argmax(counts)] + (bin_width / 2.0)

def main():
    # Detectar el archivo automáticamente al igual que los demás scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(script_dir, "SearchResults.txt")
    
    if not os.path.exists(ruta):
        print(f"ERROR: Archivo {ruta} no encontrado.")
        return
        
    print("Cargando datos para el análisis temporal...")
    df = cargar_magnitudes_fechas(ruta)

    # Calcular Mc usando una ventana deslizante de sismos
    window_size = 1500  # Número de sismos para cada estimación (aumenta para suavizar)
    step = 500          # Solapamiento
    
    fechas_centro = []
    mc_values = []

    print(f"Calculando evolución de Mc en ventanas de {window_size} eventos...")
    for i in range(0, len(df) - window_size, step):
        ventana = df.iloc[i:i+window_size]
        mags_ventana = ventana['mag'].values
        
        mc_estimado = estimar_mc_maxc(mags_ventana)
        fecha_media = ventana['datetime'].iloc[int(window_size/2)]
        
        mc_values.append(mc_estimado)
        fechas_centro.append(fecha_media)

    # 3. Graficar
    plt.figure(figsize=(11, 5), dpi=300)
    plt.plot(fechas_centro, mc_values, color='#2980B9', marker='o', markersize=3, 
             linestyle='-', linewidth=1.2, alpha=0.8, label=r'Estimación $M_c$ (Método MAXC)')

    plt.axhline(1.5, color='#E74C3C', linestyle='--', linewidth=2, 
                label=r'Referencia general esperada ($M_c = 1.5$)')

    plt.xlabel('Año', fontsize=12, fontweight='bold')
    plt.ylabel(r'Magnitud de Completitud ($M_c$)', fontsize=12, fontweight='bold')
    plt.title('Evolución Temporal de la Calidad de la Red Sísmica (SCEDC)', fontsize=13, fontweight='bold')
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_filename = 'figura_Mc_vs_tiempo.png'
    plt.savefig(output_filename)
    print(f"Gráfica guardada como '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()