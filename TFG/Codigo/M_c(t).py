import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generar_mapa_scedc_mc(archivo_cat, mc_cutoff=3.0):
    data = []
    with open(archivo_cat, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    dt = pd.to_datetime(f"{parts[0]} {parts[1]}")
                    mag, lat, lon = float(parts[4]), float(parts[6]), float(parts[7])
                    data.append({'datetime': dt, 'lat': lat, 'lon': lon, 'mag': mag})
                except Exception:
                    continue
                    
    df = pd.DataFrame(data).dropna()
    
    # Filter strictly for M >= Mc = 3.0 and SCEDC region
    df_filtered = df[(df['mag'] >= mc_cutoff) & 
                     (df['lat'] >= 32.0) & (df['lat'] <= 37.0) & 
                     (df['lon'] >= -122.0) & (df['lon'] <= -114.0)].reset_index(drop=True)

    plt.figure(figsize=(9, 9))
    
    # Background Seismicity
    plt.scatter(df_filtered['lon'], df_filtered['lat'], c='#888888', s=6, alpha=0.4, label=f'Sismicidad de Fondo ($M \ge {mc_cutoff:.1f}$)')
    
    # Major Historical Mainshocks
    mainshocks = [
        {'name': 'Landers M7.3 (1992)', 'lat': 34.20, 'lon': -116.44, 'mag': 7.3},
        {'name': 'Big Bear M6.4 (1992)', 'lat': 34.26, 'lon': -116.83, 'mag': 6.4},
        {'name': 'Northridge M6.7 (1994)', 'lat': 34.21, 'lon': -118.54, 'mag': 6.7},
        {'name': 'Hector Mine M7.1 (1999)', 'lat': 34.59, 'lon': -116.27, 'mag': 7.1},
        {'name': 'Ridgecrest M7.1 (2019)', 'lat': 35.77, 'lon': -117.60, 'mag': 7.1}
    ]
    
    for ms in mainshocks:
        plt.scatter(ms['lon'], ms['lat'], c='red', s=ms['mag']*25, edgecolors='black', linewidths=1.2, zorder=5)
        plt.annotate(ms['name'], (ms['lon'], ms['lat']), xytext=(6, 6), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.xlim(-122.0, -114.0)
    plt.ylim(32.0, 37.0)
    plt.xlabel('Longitud (°O)', fontweight='bold')
    plt.ylabel('Latitud (°N)', fontweight='bold')
    plt.title(f'Mapa de Sismicidad en California del Sur ($M \ge M_c = {mc_cutoff:.1f}$)\nCatálogo SCEDC Ajustado a Completitud', fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig("figura_5_mapa_corregido_mc.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    archivo = r"C:\Users\jorge\OneDrive\Desktop\TFG-Terremotos\TFG\Codigo\SearchResults.txt"
    if os.path.exists(archivo):
        generar_mapa_scedc_mc(archivo, mc_cutoff=3.0)