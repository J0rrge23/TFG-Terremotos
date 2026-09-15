# =====================================================================
# DIAGRAMA DE ERROR DE MOLCHAN PARA DATOS SÍSMICOS REALES (USGS)
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CARGA Y PREPARACIÓN DEL CATÁLOGO
df = pd.read_csv("terremotos_baja_california.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 2. CONFIGURACIÓN DE PARÁMETROS FÍSICOS Y TEMPORALES
Mc = 2.5                   # Magnitud de completitud
M_target = 4.2             # Magnitud del evento objetivo (Mainshocks a predecir)
ventana_dias = 7           # Ventana temporal previa para la tasa de sismicidad R(t)

# Filtrar catálogo por magnitud de completitud
df_filt = df[df['mag'] >= Mc].copy()
df_filt['fecha'] = df_filt['time'].dt.floor('D')

# Crear línea temporal continua discreta por días
t_inicio = df_filt['fecha'].min()
t_fin = df_filt['fecha'].max()
eje_dias = pd.date_range(start=t_inicio, end=t_fin, freq='D')

# Conteo diario de sismos (M >= Mc)
conteo_diario = df_filt.groupby('fecha').size().reindex(eje_dias, fill_value=0)

# Indicador Precursor R(t): Tasa media móvil de sismos en los últimos 'ventana_dias'
R_t = conteo_diario.rolling(window=ventana_dias, min_periods=1).mean()

# Identificar días en los que ocurrió al menos un evento objetivo (M >= M_target)
fechas_target = df_filt[df_filt['mag'] >= M_target]['fecha'].unique()
dias_objetivo = pd.Series(eje_dias.isin(fechas_target), index=eje_dias)

# 3. CÁLCULO CUALITATIVO DEL DIAGRAMA DE MOLCHAN
umbrales_R = np.linspace(R_t.min(), R_t.max(), 250)
tau_alarma = []  # Volumen de alarma (fracción de tiempo en alerta)
nu_fallos = []   # Tasa de fallos (fracción de sismos no predichos)

total_dias = len(eje_dias)
total_eventos_target = dias_objetivo.sum()

for U in umbrales_R:
    # IMPORTANTE: shift(1) garantiza predicción estricta (usar solo datos del pasado)
    alarma_activa = (R_t.shift(1) > U).fillna(False)
    
    # Fracción del tiempo total en estado de alarma
    volumen = alarma_activa.sum() / total_dias
    
    # Cálculo de eventos no detectados
    if total_eventos_target > 0:
        eventos_detectados = (dias_objetivo & alarma_activa).sum()
        fallos = 1.0 - (eventos_detectados / total_eventos_target)
    else:
        fallos = 1.0
        
    tau_alarma.append(volumen)
    nu_fallos.append(fallos)

# 4. REPRESENTACIÓN GRÁFICA
plt.figure(figsize=(7, 6))

# Línea de predicción aleatoria (sin capacidad predictiva)
plt.plot([0, 1], [1, 0], 'k--', linewidth=1.5, label='Estrategia Aleatoria (Azar)')

# Curva empírica de Molchan
plt.plot(tau_alarma, nu_fallos, color='navy', linewidth=2.5, 
         label=rf'Datos Reales ($M_c={Mc}$, $M_{{target}} \geq {M_target}$)')

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xlabel(r'Volumen de Alarma ($\tau_a$)', fontsize=11)
plt.ylabel(r'Tasa de Fallos ($\nu$)', fontsize=11)
plt.title(f'Diagrama de Molchan — Baja California\nPrecursor: Tasa de Sismicidad ({ventana_dias} días)', 
          fontsize=12, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("--- RESUMEN DEL ANÁLISIS DE MOLCHAN ---")
print(f"Días totales analizados: {total_dias}")
print(f"Eventos objetivo (M >= {M_target}): {total_eventos_target}")