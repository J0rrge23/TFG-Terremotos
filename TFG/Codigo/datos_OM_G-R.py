import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar y preparar el catálogo de datos
# Asegúrate de colocar el nombre correcto del archivo CSV descargado
df = pd.read_csv("terremotos_baja_california.csv") 
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# -------------------------------------------------------------------
# OBSERVABLE 1: LEY DE GUTENBERG-RICHTER (MLE & Distribución Acumulada)
# -------------------------------------------------------------------
Mc = 2.5  # Magnitud de completitud elegida
df_gr = df[df['mag'] >= Mc].copy()

# Estimación del valor b por Máxima Verosimilitud (Fórmula de Aki-Utsu)
delta_m = 0.1
m_mean = df_gr['mag'].mean()
b_value = (np.log10(np.exp(1))) / (m_mean - (Mc - (delta_m / 2)))

print("--- ANÁLISIS GUTENBERG-RICHTER ---")
print(f"Número de sismos (M >= {Mc}): {len(df_gr)}")
print(f"Valor b estimado (MLE): {b_value:.3f}\n")

# Graficar la distribución Exceedance N(>=M)
m_values, counts = np.unique(df_gr['mag'], return_counts=True)
cum_counts = np.cumsum(counts[::-1])[::-1]  # Exceedance / Acumulada

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(m_values, cum_counts, color='navy', alpha=0.7, label='Datos reales')

# Ajuste teórico log10(N) = a - b*M
a_value = np.log10(len(df_gr)) + b_value * Mc
fit_N = 10**(a_value - b_value * m_values)
plt.plot(m_values, fit_N, 'r--', label=rf'Ajuste MLE ($b = {b_value:.2f}$)')

plt.yscale('log')
plt.xlabel(r'Magnitud ($M$)')
plt.ylabel(r'$N(\geq M)$')  # Corregido: Raw string y \geq para LaTeX
plt.title('Ley de Gutenberg-Richter (Exceedance)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

# -------------------------------------------------------------------
# OBSERVABLE 2: LEY DE OMORI (Decaimiento de réplicas)
# -------------------------------------------------------------------
# Seleccionar el Terremoto Principal (Mainshock)
mainshock = df.loc[df['mag'].idxmax()]
t_main = mainshock['time']

print("--- ANÁLISIS LEY DE OMORI ---")
print(f"Mainshock detectado: Magnitud {mainshock['mag']} el {t_main}\n")

# Filtrar ventana temporal posterior (ej. 60 días tras el sismo principal)
dias_ventana = 60
df_replica = df[(df['time'] > t_main) & (df['time'] <= t_main + pd.Timedelta(days=dias_ventana))].copy()
df_replica['dt_days'] = (df_replica['time'] - t_main).dt.total_seconds() / 86400.0

# Binning logarítmico del tiempo transcurrido (t - t_0)
bins = np.logspace(-2, np.log10(dias_ventana), 20)
counts_omori, bin_edges = np.histogram(df_replica['dt_days'], bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths = np.diff(bin_edges)

# Filtrar bines sin terremotos para evitar divisiones por cero en escala logarítmica
mask = counts_omori > 0
bin_centers = bin_centers[mask]
rate = counts_omori[mask] / bin_widths[mask]  # Tasa n(t) = sismos por día

plt.subplot(1, 2, 2)
plt.plot(bin_centers, rate, 'o-', color='darkgreen', label=r'Tasa de réplicas $n(t)$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Tiempo desde el sismo principal $t - t_0$ (días)')
plt.ylabel(r'Tasa de réplicas $n(t)$ [eventos/día]')
plt.title(rf'Ley de Omori (Mainshock $M={mainshock["mag"]}$)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()