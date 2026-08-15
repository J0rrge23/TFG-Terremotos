# =====================================================================
# MODELO OFC MULTI-ALPHA — G-R, ESTRÉS, MOLCHAN Y LEY DE OMORI
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

# 1. CONFIGURACIÓN GENERAL
N = 150                # Cuadrícula de 200x200
pasos = 30000          # Pasos por simulación
F_th = 1.0             # Umbral de ruptura
lista_alphas = [0.15, 0.20, 0.23, 0.245]  # Lista de alphas a evaluar (< 0.25)

resultados = {}

# 2. FUNCIÓN DE SIMULACIÓN DE UN SOLO ALPHA
def simular_ofc(alpha_val):
    print(f"Simulando alpha = {alpha_val}...")
    np.random.seed(42)  # Semilla para comparabilidad justa
    cuadricula = np.random.uniform(0, F_th, (N, N))
    lista_tamaños = []
    lista_estres = []

    for t in range(pasos):
        # Carga tectónica
        F_max = np.max(cuadricula)
        cuadricula += (F_th - F_max)
        lista_estres.append(np.mean(cuadricula))
        
        inestables_x, inestables_y = np.where(cuadricula >= F_th - 1e-9)
        pila = list(zip(inestables_x, inestables_y))
        terremoto_tamaño = 0
        
        while pila:
            cx, cy = pila.pop()
            if cuadricula[cx, cy] >= F_th:
                estres_acumulado = cuadricula[cx, cy]
                cuadricula[cx, cy] = 0.0
                terremoto_tamaño += 1
                
                estres_transferido = alpha_val * estres_acumulado
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        cuadricula[nx, ny] += estres_transferido
                        if cuadricula[nx, ny] >= F_th:
                            pila.append((nx, ny))
                            
        lista_tamaños.append(terremoto_tamaño)
        
    return np.array(lista_tamaños), np.array(lista_estres)

# 3. BUCLE SOBRE TODOS LOS ALPHAS Y CÁLCULO DE MÉTRICAS
for alpha in lista_alphas:
    sismos, estres = simular_ofc(alpha)
    transitorio = int(pasos * 0.3)
    
    s_est = sismos[transitorio:]
    e_est = estres[transitorio:]
    s_filt = s_est[s_est > 0]
    
    # Ajuste MLE Powerlaw
    ajuste = powerlaw.Fit(s_filt, discrete=True, xmin=3, verbose=False)
    beta = ajuste.power_law.alpha - 1.0
    
    # Diagrama de Molchan
    umbral_sismo = np.percentile(s_filt, 90)
    eventos_obj = s_est >= umbral_sismo
    umbrales_e = np.linspace(min(e_est), max(e_est), 100)
    tau_a, nu_f = [], []
    for u in umbrales_e:
        alerta = e_est > u
        tot = np.sum(eventos_obj)
        tau_a.append(np.sum(alerta) / len(s_est))
        nu_f.append(1.0 - (np.sum(eventos_obj & alerta) / tot) if tot > 0 else 1.0)
        
    # Ley de Omori
    umbral_main = np.percentile(s_est, 98)
    indices_main = np.where(s_est >= umbral_main)[0]
    ventana = 50
    actividad = np.zeros(ventana)
    conteo = 0
    for idx in indices_main:
        if idx + ventana < len(s_est):
            actividad += s_est[idx + 1 : idx + 1 + ventana]
            conteo += 1
    tasa_omori = actividad / conteo if conteo > 0 else np.zeros(ventana)
    
    resultados[alpha] = {
        'sismos': s_filt,
        'beta': beta,
        'tau_molchan': tau_a,
        'nu_molchan': nu_f,
        'omori': tasa_omori
    }

# 4. GRÁFICA COMPARATIVA 1: CCDF DE GUTENBERG-RICHTER
plt.figure(figsize=(8, 6))
colores = plt.cm.viridis(np.linspace(0.1, 0.9, len(lista_alphas)))

for (alpha, res), color in zip(resultados.items(), colores):
    s_ord = np.sort(res['sismos'])
    ccdf = 1.0 - (np.arange(len(s_ord)) / len(s_ord))
    plt.plot(s_ord, ccdf, label=fr'$\alpha={alpha}$ ($b={res["beta"]:.2f}$)', lw=2, color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tamaño del Terremoto (S)', fontsize=11)
plt.ylabel(r'Probabilidad Acumulada $P(S \geq s)$', fontsize=11)
plt.title('Gutenberg-Richter / CCDF para distintos $\\alpha$', fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# 5. GRÁFICA COMPARATIVA 2: MOLCHAN Y OMORI
fig, (ax_mol, ax_omo) = plt.subplots(1, 2, figsize=(13, 5))

ax_mol.plot([0, 1], [1, 0], 'k--', lw=1.5, label='Azar')

for (alpha, res), color in zip(resultados.items(), colores):
    # Molchan
    ax_mol.plot(res['tau_molchan'], res['nu_molchan'], label=fr'$\alpha={alpha}$', lw=2, color=color)
    
    # Omori
    ax_omo.plot(range(1, 51), res['omori'], marker='o', ms=3, label=fr'$\alpha={alpha}$', lw=1.5, color=color)

ax_mol.set_xlabel(r'Volumen de Alarma ($\tau_a$)', fontsize=11)
ax_mol.set_ylabel(r'Tasa de Fallos ($\nu$)', fontsize=11)
ax_mol.set_title('Diagrama de Error de Molchan', fontsize=12, fontweight='bold')
ax_mol.grid(True, linestyle=':', alpha=0.6)
ax_mol.legend(fontsize=10)

ax_omo.set_xlabel(r'Tiempo tras el sismo ($\Delta t$)', fontsize=11)
ax_omo.set_ylabel('Tamaño medio de réplica', fontsize=11)
ax_omo.set_title('Respuesta Post-Mainshock (Omori)', fontsize=12, fontweight='bold')
ax_omo.grid(True, linestyle=':', alpha=0.6)
ax_omo.legend(fontsize=10)

plt.tight_layout()
plt.show()