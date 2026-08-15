# =====================================================================
# BARRIDO DEL PARÁMETRO ALPHA EN EL MODELO OFC (10 VALORES)
# Estudio de Gutenberg-Richter, Ley de Omori y Predictibilidad
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import warnings

warnings.filterwarnings('ignore') # Ignorar avisos de convergencia de powerlaw

# --- 1. CONFIGURACIÓN DEL BARRIDO ---
N = 100                         # Tamaña de la red NxN
pasos = 25000                    # Pasos de tiempo por simulación
F_th = 1.0                       # Umbral de ruptura
alphas = np.linspace(0.15, 0.24, 10) # 10 valores de alpha de 0.15 a 0.24

# Estructuras para almacenar resultados globales
resultados_tau = []
resultados_b = []
intensidad_omori = []
estres_critico = []

print(f"Iniciando barrido de {len(alphas)} valores de alpha para N={N}...")

# --- 2. BUCLE PRINCIPAL DE BARRIDO DE ALPHA ---
for i, alpha in enumerate(alphas):
    cuadricula = np.random.uniform(0, F_th, (N, N))
    sismos = []
    estres_medio = []
    
    for t in range(pasos):
        # Carga tectónica
        F_max = np.max(cuadricula)
        delta_F = F_th - F_max
        cuadricula += delta_F
        estres_medio.append(np.mean(cuadricula))
        
        # Casillas inestables
        inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
        pila = list(zip(inestables_x, inestables_y))
        sismo_tamaño = 0
        
        while pila:
            cx, cy = pila.pop()
            if cuadricula[cx, cy] >= F_th:
                estres_acum = cuadricula[cx, cy]
                cuadricula[cx, cy] = 0.0
                sismo_tamaño += 1
                
                estres_trans = alpha * estres_acum
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        cuadricula[nx, ny] += estres_trans
                        if cuadricula[nx, ny] >= F_th:
                            pila.append((nx, ny))
                            
        sismos.append(sismo_tamaño)
        
    # --- PROCESAMIENTO DE DATOS ---
    trans = int(pasos * 0.3)
    sismos_est = np.array(sismos[trans:])
    estres_est = np.array(estres_medio[trans:])
    sismos_filt = sismos_est[sismos_est > 0]
    
    # 1. Fit MLE con powerlaw
    ajuste = powerlaw.Fit(sismos_filt, discrete=True, xmin=3, verbose=False)
    tau = ajuste.power_law.alpha
    b_val = tau - 1.0
    
    resultados_tau.append(tau)
    resultados_b.append(b_val)
    estres_critico.append(np.mean(estres_est))
    
    # 2. Análisis de Omori (Señal de réplicas post-mainshock en dt = 1..5)
    umbral_main = np.percentile(sismos_est, 98)
    idx_main = np.where(sismos_est >= umbral_main)[0]
    
    ventana = 20
    replicas_post = np.zeros(ventana)
    n_mains = 0
    for idx in idx_main:
        if idx + ventana < len(sismos_est):
            replicas_post += sismos_est[idx+1 : idx+1+ventana]
            n_mains += 1
            
    if n_mains > 0:
        replicas_post /= n_mains
        # Intensidad de Omori = Exceso del pico inicial (dt=1..3) sobre el ruido de fondo
        ruido_fondo = np.mean(sismos_est)
        pico_replicas = np.mean(replicas_post[0:3])
        exceso_omori = max(0, (pico_replicas - ruido_fondo) / ruido_fondo)
    else:
        exceso_omori = 0
        
    intensidad_omori.append(exceso_omori)
    
    print(f"  [{i+1}/10] alpha={alpha:.2f} | tau={tau:.2f} | b={b_val:.2f} | Omori_Signal={exceso_omori:.2f}")

print("\nBarrido completado exitosamente. Generando gráficos...")

# --- 3. FIGURA MULTIPANEL DE RESULTADOS ---
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(fr'Estudio del Parámetro de Disipación $\alpha$ en el Modelo OFC ($N={N}$)', fontsize=14, fontweight='bold')

# Panel 1: Exponente crítico tau y valor b de G-R frente a alpha
axs[0, 0].plot(alphas, resultados_tau, 'o-', color='crimson', lw=2, label=r'Exponente $\tau$ (PDF)')
axs[0, 0].plot(alphas, resultados_b, 's--', color='navy', lw=2, label=r'Valor $b$ de G-R ($b = \tau - 1$)')
axs[0, 0].set_xlabel(r'Parámetro de Elasticidad $\alpha$', fontsize=11)
axs[0, 0].set_ylabel('Exponente Estructura', fontsize=11)
axs[0, 0].set_title(r'1. Variación de $\tau$ y $b$ con la Disipación', fontsize=11)
axs[0, 0].grid(True, linestyle=':', alpha=0.6)
axs[0, 0].legend(fontsize=10)

# Panel 2: Intensidad de la Ley de Omori frente a alpha
axs[0, 1].plot(alphas, intensidad_omori, '^-', color='purple', lw=2.5, ms=7)
axs[0, 1].set_xlabel(r'Parámetro de Elasticidad $\alpha$', fontsize=11)
axs[0, 1].set_ylabel('Señal de Réplicas (Exceso Relativo)', fontsize=11)
axs[0, 1].set_title(r'2. Emergencia de la Ley de Omori ($\alpha \to 0.25$)', fontsize=11)
axs[0, 1].grid(True, linestyle=':', alpha=0.6)

# Panel 3: Estrés Crítico Estacionario frente a alpha
axs[1, 0].plot(alphas, estres_critico, 'd-', color='darkgreen', lw=2)
axs[1, 0].set_xlabel(r'Parámetro de Elasticidad $\alpha$', fontsize=11)
axs[1, 0].set_ylabel(r'Estrés Crítico Medios $\langle F \rangle_c$', fontsize=11)
axs[1, 0].set_title(r'3. Nivel de Estrés Estacionario del Atractor SOC', fontsize=11)
axs[1, 0].grid(True, linestyle=':', alpha=0.6)

# Panel 4: Mapa de resumen de propiedades
axs[1, 1].axis('off')
texto_resumen = (
    "SÍNTESIS DE PROPIEDADES EN OFC:\n\n"
    r"• Carácter Abeliano: NO ABELIANO $[T_i, T_j] \neq 0$" + "\n\n"
    r"• Régimen Conservativo ($\alpha \to 0.25$):" + "\n"
    r"  - Menor exponente $b$ (terremotos más grandes)." + "\n"
    r"  - Aparición clara de Réplicas (Ley de Omori)." + "\n\n"
    r"• Régimen Disipativo ($\alpha < 0.18$):" + "\n"
    r"  - Mayor exponente $b$ (eventos pequeños dominan)." + "\n"
    r"  - Desaparición de Réplicas (Ruido plano)."
)
axs[1, 1].text(0.05, 0.2, texto_resumen, fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.show()