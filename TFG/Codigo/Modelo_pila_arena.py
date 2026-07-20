#Codigo de la pila de arena de una cuadricula 30x30, donde incluyo una partícula en una de las casillas a cada paso de tiempo
#las variables son discretas y el tiempo es discreto, la simulación se realiza en un bucle for que recorre un número determinado de pasos de tiempo. En cada paso de tiempo, se selecciona una casilla aleatoria de la cuadricula y se incrementa el valor de esa casilla en 1, representando la adición de una partícula a esa posición.
#las condiciones de frontera son disipativas, es decir, si una casilla alcanza un valor mayor o igual a 4, se "desborda" y distribuye sus partículas a las casillas vecinas (arriba, abajo, izquierda, derecha). Si una casilla está en el borde de la cuadricula y se desborda, las partículas que se desbordan se pierden.

#MEdir la carga media de la cuadricula a lo largo del tiempo, es decir, el promedio de partículas por casilla en cada paso de tiempo. Esto se puede hacer sumando todos los valores de la cuadricula y dividiéndolos por el número total de casillas (900 en este caso). Almacenar estos valores en una lista para poder analizarlos posteriormente.
#Tambien quiero medir la criticidad de la cuadricula, es decir, el número de casillas que tienen un valor mayor o igual a 4 en cada paso de tiempo. Esto se puede hacer recorriendo la cuadricula y contando cuántas casillas cumplen esta condición. Almacenar estos valores en una lista para poder analizarlos posteriormente.
import random
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN DE LA SIMULACIÓN ---
N = 30                 # Tamaño de la cuadrícula (30x30 = 900 casillas)
pasos = 10000          # Número de pasos de tiempo
cuadricula = [[0 for _ in range(N)] for _ in range(N)]  # Inicializar con ceros

# --- VARIABLES DE CONTROL OPTIMIZADAS ---
total_particulas = 0   # Rastreador O(1) del número total de granos en la red
sitios_criticos = 0    # Rastreador O(1) de las casillas con exactamente 3 granos

# --- LISTAS PARA ALMACENAR MÉTRICAS ---
lista_carga_media = []
lista_tamaño_avalancha = []  # Mide la criticidad dinámica (desbordes por paso)
lista_sitios_criticos = []   # Mide cuántas casillas quedaron con exactamente 3 granos

# --- BUCLE PRINCIPAL DE TIEMPO ---
for t in range(pasos):
    # 1. Seleccionar una casilla aleatoria e incrementar su valor
    x = random.randint(0, N-1)
    y = random.randint(0, N-1)
    
    val_antiguo = cuadricula[x][y]
    cuadricula[x][y] += 1
    total_particulas += 1     # Incremento en el contador global
    
    # Actualización O(1) de sitios críticos (valor igual a 3)
    if val_antiguo == 3:
        sitios_criticos -= 1
    elif cuadricula[x][y] == 3:
        sitios_criticos += 1
    
    # 2. Resolver desbordes (avalanchas) en cadena
    pila_inestables = []
    if cuadricula[x][y] >= 4:
        pila_inestables.append((x, y))
        
    desbordes_este_paso = 0
    
    # Mientras existan casillas inestables, la avalancha continúa
    while pila_inestables:
        cx, cy = pila_inestables.pop()
        
        # Verificamos de nuevo que siga inestable
        if cuadricula[cx][cy] >= 4:
            # La casilla se desborda: pierde 4 granos
            cuadricula[cx][cy] -= 4
            desbordes_este_paso += 1
            
            # Si su nuevo valor tras perder 4 granos es 3, sumamos uno
            if cuadricula[cx][cy] == 3:
                sitios_criticos += 1
            
            # Distribuir 1 grano a los 4 vecinos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                
                # Si está dentro de los límites, recibe el grano
                if 0 <= nx < N and 0 <= ny < N:
                    val_antiguo_vecino = cuadricula[nx][ny]
                    cuadricula[nx][ny] += 1
                    
                    # Actualización O(1) de sitios críticos para el vecino
                    if val_antiguo_vecino == 3:
                        sitios_criticos -= 1
                    elif cuadricula[nx][ny] == 3:
                        sitios_criticos += 1
                    
                    # Si este vecino ahora es inestable, se añade a la pila para colapsar
                    if cuadricula[nx][ny] >= 4:
                        pila_inestables.append((nx, ny))
                else:
                    # Si se sale del borde, se disipa (se pierde) y se resta del total
                    total_particulas -= 1

    # 3. MEDICIÓN DE MÉTRICAS (al finalizar la avalancha del paso 't')
    carga_media = total_particulas / (N * N)
    lista_carga_media.append(carga_media)
    lista_tamaño_avalancha.append(desbordes_este_paso)
    lista_sitios_criticos.append(sitios_criticos)

# --- FIN DE LA SIMULACIÓN ---
print(
    f"Simulación finalizada tras {pasos} pasos.\n"
    f"Carga media final: {lista_carga_media[-1]:.4f}\n"
    f"Sitios críticos al final: {lista_sitios_criticos[-1]} de {N*N} ({(lista_sitios_criticos[-1]/(N*N))*100:.1f}%)\n"
    f"Número de desbordes totales: {sum(lista_tamaño_avalancha)}\n"
)


# =====================================================================
# RECORTE DE ZONAS DE INTERÉS
# =====================================================================

# Definimos el inicio del estado estacionario (descartamos el primer 30% del transitorio)
inicio_estacionario = int(pasos * 0.3)

# 1. Ajuste de datos para la serie temporal
tiempo_interes = np.arange(inicio_estacionario, pasos)
carga_interes = lista_carga_media[inicio_estacionario:]
criticos_interes = lista_sitios_criticos[inicio_estacionario:]

# 2. Ajuste de datos para la Ley de Potencias
avalanchas_estacionarias = lista_tamaño_avalancha[inicio_estacionario:]
avalanchas_filtradas = [s for s in avalanchas_estacionarias if s > 0]


# =====================================================================
# GRÁFICOS 1 & 2: EVOLUCIÓN EN EL ESTADO ESTACIONARIO (SOC)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle(f"Análisis en el Estado Estacionario (SOC) — Pasos {inicio_estacionario} a {pasos}", 
             fontsize=14, fontweight='bold')

# Gráfico 1: Carga Media en la zona de interés
ax1.plot(tiempo_interes, carga_interes, color='teal', lw=1.2, label='Carga Media Activa')
ax1.axhline(17/8, color='crimson', linestyle='--', lw=1.5, label='Límite Teórico Infinito (2.125)')
ax1.set_ylabel('Carga Media\n(partículas / casilla)', fontsize=10)
ax1.set_title('1. Fluctuaciones de la Carga Media en Equilibrio', fontsize=11, loc='left')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Optimizar escala vertical para apreciar las microfluctuaciones de la carga
ax1.set_ylim(min(carga_interes) * 0.98, max(carga_interes) * 1.02)

# Gráfico 2: Sitios Críticos en la zona de interés
ax2.plot(tiempo_interes, criticos_interes, color='darkorange', lw=1, alpha=0.8, label='Sitios Críticos (h = 3)')
ax2.set_xlabel('Paso de tiempo (t)', fontsize=11)
ax2.set_ylabel('Nº de Sitios Críticos', fontsize=10)
ax2.set_title('2. Densidad de Sitios Críticos (Estado de Alerta)', fontsize=11, loc='left')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# Optimizar escala vertical para los sitios críticos
ax2.set_ylim(min(criticos_interes) * 0.9, max(criticos_interes) * 1.1)

plt.tight_layout()
plt.show()


# =====================================================================
# GRÁFICO 3: LEY DE POTENCIAS EN LA ZONA DE ESCALA (SCALING REGION)
# =====================================================================
if len(avalanchas_filtradas) > 10:
    max_s = max(avalanchas_filtradas)
    num_bins = 20  
    bins_log = np.logspace(0, np.log10(max_s), num_bins)
    
    # Calcular Histograma y Densidad de Probabilidad (PDF)
    conteos, bordes_bins = np.histogram(avalanchas_filtradas, bins=bins_log)
    centros_bins = np.sqrt(bordes_bins[:-1] * bordes_bins[1:])
    anchos_bins = np.diff(bordes_bins)
    pdf = conteos / (sum(conteos) * anchos_bins)
    
    mascara = pdf > 0
    x_datos = centros_bins[mascara]
    y_datos = pdf[mascara]
    
    # Límites rigurosos de la "Zona de Interés de Escala" (Scaling Region)
    # Evitamos S = 1 (discreto) y S > N*N/4 (efecto de borde / tamaño finito)
    x_min_interes = 2
    x_max_interes = (N * N) / 4
    
    rango_ajuste = (x_datos >= x_min_interes) & (x_datos <= x_max_interes)
    
    if sum(rango_ajuste) > 2:
        log_x = np.log10(x_datos)
        log_y = np.log10(y_datos)
        pendiente, interseccion = np.polyfit(log_x[rango_ajuste], log_y[rango_ajuste], 1)
        tau = -pendiente
        print(f"--> Exponente crítico hallado en la zona de escala (tau): {tau:.3f}")
    else:
        pendiente, interseccion = None, None
        print("No hay suficientes datos en la zona de escala para ajustar.")
    
    # --- Gráfico de la Ley de Potencias enfocado en la Zona de Interés ---
    plt.figure(figsize=(8, 6))
    
    # Pintar todos los datos con opacidad baja
    plt.scatter(x_datos, y_datos, color='gray', edgecolor='none', s=40, alpha=0.3, label='Fuera de Escala')
    
    # Destacar los puntos que pertenecen estrictamente a la Zona de Interés
    plt.scatter(x_datos[rango_ajuste], y_datos[rango_ajuste], color='darkorange', 
                edgecolor='black', s=65, zorder=3, label='Zona de Interés (Escala Limpia)')
    
    # Pintar la recta de ajuste
    if pendiente is not None:
        x_ajuste = np.logspace(np.log10(x_min_interes), np.log10(x_max_interes), 100)
        y_ajuste = 10**interseccion * x_ajuste**pendiente
        plt.plot(x_ajuste, y_ajuste, color='navy', linestyle='--', lw=2.5, 
                 label=f'Ajuste Teórico ($\\tau$ = {tau:.2f})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamaño de la Avalancha ($S$)', fontsize=12)
    plt.ylabel('Densidad de Probabilidad $P(S)$', fontsize=12)
    plt.title('Distribución de Avalanchas: Foco en la Zona de Escala', fontsize=13, fontweight='bold')
    
    # --- ACERCAMIENTO VISUAL (ZOOM) A LA ZONA DE INTERÉS ---
    # Centramos el eje X desde 1 hasta el límite superior de escala (con un margen dinámico)
    plt.xlim(0.9, x_max_interes * 1.5)
    
    # Ajustamos el eje Y para omitir el vacío de los datos dispersos del final
    y_interes_valores = y_datos[rango_ajuste]
    if len(y_interes_valores) > 0:
        plt.ylim(min(y_interes_valores) * 0.5, max(y_datos) * 2)
        
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.show()

else:
    print("No se registraron suficientes avalanchas para realizar la estadística.")