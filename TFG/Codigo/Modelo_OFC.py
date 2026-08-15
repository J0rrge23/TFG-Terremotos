# =====================================================================
# MODELO OFC MULTI-N Y MULTI-ALPHA — DIAGRAMA DE MOLCHAN Y LEY DE OMORI
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN GENERAL
lista_N = [150,200, 250,300,350,400]                     # Tamaños de cuadrícula a evaluar
pasos = 20000                               # Pasos por simulación
F_th = 1.0                                  # Umbral de ruptura
lista_alphas = np.linspace(0.05, 0.245, 10) # 10 valores de alpha (< 0.25)

# 2. FUNCIÓN DE SIMULACIÓN DE UN SOLO ALPHA Y TAMAÑO N
def simular_ofc(alpha_val, N_val):
    print(f"  -> Simulando N = {N_val}x{N_val} | alpha = {alpha_val:.3f}...")
    np.random.seed(42)  # Semilla para comparabilidad justa
    cuadricula = np.random.uniform(0, F_th, (N_val, N_val))
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
                    if 0 <= nx < N_val and 0 <= ny < N_val:
                        cuadricula[nx, ny] += estres_transferido
                        if cuadricula[nx, ny] >= F_th:
                            pila.append((nx, ny))
                            
        lista_tamaños.append(terremoto_tamaño)
        
    return np.array(lista_tamaños), np.array(lista_estres)

# 3. BUCLE PRINCIPAL SOBRE TAMAÑOS DE RED N
for N in lista_N:
    print(f"\n==========================================")
    print(f"   EJECUTANDO SIMULACIONES PARA N = {N}x{N}")
    print(f"==========================================")
    
    resultados = {}
    
    # BUCLE SOBRE LOS 10 ALPHAS PARA EL N ACTUAL
    for alpha in lista_alphas:
        sismos, estres = simular_ofc(alpha, N)
        transitorio = int(pasos * 0.3)
        
        s_est = sismos[transitorio:]
        e_est = estres[transitorio:]
        s_filt = s_est[s_est > 0]
        
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
            'tau_molchan': tau_a,
            'nu_molchan': nu_f,
            'omori': tasa_omori
        }

    # 4. GRÁFICA COMPARATIVA POR CADA N: MOLCHAN Y OMORI
    fig, (ax_mol, ax_omo) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Propiedades Dinámicas en OFC ($N = {N}\\times{N}$)', fontsize=13, fontweight='bold')

    colores = plt.cm.inferno(np.linspace(0.15, 0.85, len(lista_alphas)))

    ax_mol.plot([0, 1], [1, 0], 'k--', lw=1.5, label='Azar')

    for (alpha, res), color in zip(resultados.items(), colores):
        # Panel Molchan
        ax_mol.plot(res['tau_molchan'], res['nu_molchan'], label=fr'$\alpha={alpha:.3f}$', lw=1.5, color=color)
        
        # Panel Omori
        ax_omo.plot(range(1, 51), res['omori'], marker='o', ms=2, label=fr'$\alpha={alpha:.3f}$', lw=1.2, color=color)

    ax_mol.set_xlabel(r'Volumen de Alarma ($\tau_a$)', fontsize=11)
    ax_mol.set_ylabel(r'Tasa de Fallos ($\nu$)', fontsize=11)
    ax_mol.set_title('Diagrama de Error de Molchan', fontsize=12, fontweight='bold')
    ax_mol.grid(True, linestyle=':', alpha=0.6)
    ax_mol.legend(fontsize=8)

    ax_omo.set_xlabel(r'Tiempo tras el sismo ($\Delta t$)', fontsize=11)
    ax_omo.set_ylabel('Tamaño medio de réplica', fontsize=11)
    ax_omo.set_yscale('log')
    ax_omo.set_title('Respuesta Post-Mainshock (Omori)', fontsize=12, fontweight='bold')
    ax_omo.grid(True, which="both", linestyle=':', alpha=0.6)
    ax_omo.legend(fontsize=8)

    plt.tight_layout()
    plt.show()