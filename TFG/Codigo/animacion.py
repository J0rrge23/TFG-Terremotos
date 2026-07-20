import pygame
import numpy as np
import random
import sys

# --- CONFIGURACIÓN DE LA SIMULACIÓN ---
N = 30                 # Tamaño de la cuadrícula (NxN)
PASOS_TIEMPO = 10000   # Cuántos granos añadir en total
VELOCIDAD_AVALANCHA = 30 # (ms) Tiempo de espera entre pasos de la avalancha (más bajo = más rápido)

# --- CONFIGURACIÓN GRÁFICA (PROFESIONAL) ---
TAM_CASILLA = 20        # Píxeles por lado de la casilla
MARGEN = 2              # Espacio entre casillas
ANCHO_PANELES = 300     # Ancho del panel de estadísticas derecho

# Calcular dimensiones de la ventana
ANCHO_GRID = N * TAM_CASILLA + (N + 1) * MARGEN
ALTO_VENTANA = N * TAM_CASILLA + (N + 1) * MARGEN
ANCHO_WINDOW = ANCHO_GRID + ANCHO_PANELES

# Colores (Paleta Moderna/Cyberpunk)
COLOR_FONDO = (10, 15, 25)      # Azul muy oscuro
COLOR_GRID = (30, 40, 60)       # Azul grisáceo oscuro
COLOR_PARTICULA = (0, 255, 200) # Turquesa neón brillante
COLOR_TEXTO = (200, 220, 240)   # Gris azulado claro
COLOR_TEORICO = (255, 100, 100) # Rojo pastel para la línea teórica

# Configuración de partículas gráficas por casilla (coordenadas locales relativas a la casilla)
# 1 grano: Centro, 2 granos: Diagonal, 3 granos: Triángulo, 4 granos: Cuadrado
RADIO_PARTICULA = (TAM_CASILLA - MARGEN * 2) // 5 # Pequeñas para que quepan
offset_c = TAM_CASILLA // 2
offset_d = TAM_CASILLA // 4
POSICIONES_GRAFICAS = {
    1: [(offset_c, offset_c)],
    2: [(offset_d, offset_d), (TAM_CASILLA - offset_d, TAM_CASILLA - offset_d)],
    3: [(offset_c, offset_d), (offset_d, TAM_CASILLA - offset_d), (TAM_CASILLA - offset_d, TAM_CASILLA - offset_d)],
    4: [(offset_d, offset_d), (TAM_CASILLA - offset_d, offset_d), (offset_d, TAM_CASILLA - offset_d), (TAM_CASILLA - offset_d, TAM_CASILLA - offset_d)]
}

# --- INICIALIZACIÓN ---
pygame.init()
pygame.display.set_caption("Modelo Pila de Arena 2D - Visualización Profesional")
pantalla = pygame.display.set_mode((ANCHO_WINDOW, ALTO_VENTANA))
reloj = pygame.time.Clock()
fuente_stats = pygame.font.SysFont("Consolas", 18)
fuente_titulo = pygame.font.SysFont("Arial", 22, bold=True)

# Inicializar cuadrícula con NumPy
cuadrgrid = np.zeros((N, N), dtype=int)
historial_carga = []
paso_actual = 0

# --- FUNCIONES DE DIBUJO ---

def dibujar_particulas_casilla(superficie, x_grid, y_grid, num_granos):
    """Dibuja las bolitas discretas dentro de una casilla específica."""
    if num_granos == 0:
        return

    # Calcular coordenadas de la esquina superior izquierda de la casilla
    posX = MARGEN + x_grid * (TAM_CASILLA + MARGEN)
    posY = MARGEN + y_grid * (TAM_CASILLA + MARGEN)

    # Dibujar borde de casilla (opcional para estructura)
    # rect_casilla = pygame.Rect(posX, posY, TAM_CASILLA, TAM_CASILLA)
    # pygame.draw.rect(superficie, COLOR_GRID, rect_casilla, 1)

    # Dibujar las bolitas según la cantidad
    particulas_a_dibujar = num_granos if num_granos < 4 else 4
    for px, py in POSICIONES_GRAFICAS[particulas_a_dibujar]:
        pygame.draw.circle(superficie, COLOR_PARTICULA, (posX + px, posY + py), RADIO_PARTICULA)

    # Efecto visual si está desbordándose (opcional)
    if num_granos >= 4:
        pygame.draw.circle(superficie, (255, 255, 255), (posX + offset_c, posY + offset_c), RADIO_PARTICULA + 3, 2)


def dibujar_grid_completo(superficie, grid):
    """Renderiza toda la cuadrícula de una vez."""
    superficie.fill(COLOR_FONDO)
    for i in range(N):
        for j in range(N):
            dibujar_particulas_casilla(superficie, i, j, grid[i, j])

def dibujar_panel_estadisticas(superficie, grid, tiempo, historial):
    """Dibuja el panel de información de la derecha."""
    panel_rect = pygame.Rect(ANCHO_GRID, 0, ANCHO_PANELES, ALTO_VENTANA)
    pygame.draw.rect(superficie, (15, 20, 30), panel_rect)
    
    # Textos fijos
    y_offset = 20
    superficie.blit(fuente_titulo.render("ESTADÍSTICAS", True, COLOR_TEXTO), (ANCHO_GRID + 20, y_offset))
    y_offset += 40
    
    # Datos dinámicos
    carga_actual = np.mean(grid)
    info = [
        f"Paso de tiempo (t): {tiempo}",
        f"Granos totales: {np.sum(grid)}",
        f"Carga Media: {carga_actual:.4f}",
        f"Valor Crítico Teór: {2.1250}"
    ]
    
    for linea in info:
        superficie.blit(fuente_stats.render(linea, True, COLOR_TEXTO), (ANCHO_GRID + 20, y_offset))
        y_offset += 25

    # Pequeño gráfico de carga media (simplificado)
    if len(historial) > 10:
        y_offset += 30
        superficie.blit(fuente_titulo.render("EVOLUCIÓN CARGA", True, COLOR_TEXTO), (ANCHO_GRID + 20, y_offset))
        y_offset += 30
        
        graf_alto = 150
        graf_ancho = ANCHO_PANELES - 40
        rect_graf = pygame.Rect(ANCHO_GRID + 20, y_offset, graf_ancho, graf_alto)
        pygame.draw.rect(superficie, COLOR_GRID, rect_graf, 1)
        
        # Línea crítica teórica
        crit_y = rect_graf.bottom - (2.125 / 2.5) * graf_alto
        pygame.draw.line(superficie, COLOR_TEORICO, (rect_graf.left, crit_y), (rect_graf.right, crit_y), 2)

        # Dibujar historial (cada 50 pasos para velocidad)
        if len(historial) > 1:
            puntos = []
            max_t = len(historial)
            for t in range(0, max_t, max(1, max_t // graf_ancho)):
                px = rect_graf.left + (t / max_t) * graf_ancho
                py = rect_graf.bottom - (historial[t] / 2.5) * graf_alto
                puntos.append((px, py))
            if len(puntos) > 1:
                pygame.draw.lines(superficie, COLOR_PARTICULA, False, puntos, 2)

# --- LÓGICA DE ACTUALIZACIÓN DEL SISTEMA ---

pila_avalancha = []

def ejecutar_paso_tiempo():
    """Añade un grano y gestiona desbordes."""
    global cuadrgrid, paso_actual, pila_avalancha
    
    # 1. Añadir una partícula
    x, y = np.random.randint(0, N, size=2)
    cuadrgrid[x, y] += 1
    paso_actual += 1
    
    # 2. Verificar si empieza avalancha
    if cuadrgrid[x, y] >= 4:
        pila_avalancha.append((x, y))

def procesar_paso_avalancha():
    """Ejecuta UN SOLO paso de desborde de la avalancha actual."""
    global cuadrgrid, pila_avalancha
    
    if not pila_avalancha:
        return False # No hay avalancha activa

    cx, cy = pila_avalancha.pop()
    
    if cuadrgrid[cx, cy] >= 4:
        # Colapso de la casilla
        cuadrgrid[cx, cy] -= 4
        
        # Repartir a vecinos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < N and 0 <= ny < N:
                cuadrgrid[nx, ny] += 1
                if cuadrgrid[nx, ny] >= 4:
                    pila_avalancha.append((nx, ny))
        return True # Hubo un desborde visible

    return False # La casilla ya se había estabilizado por otro desborde

# --- BUCLE PRINCIPAL (MAIN LOOP) ---

corriendo = True
pausado = False
en_avalancha_visual = False
ultimo_reparto_time = pygame.time.get_ticks()

while corriendo:
    # Gestión de eventos (Cerrar, Pausa, Velocidad)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            corriendo = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pausado = not pausado
            if event.key == pygame.K_UP:
                VELOCIDAD_AVALANCHA = max(1, VELOCIDAD_AVALANCHA - 10)
            if event.key == pygame.K_DOWN:
                VELOCIDAD_AVALANCHA += 10

    if not pausado:
        current_time = pygame.time.get_ticks()

        # Si hay avalancha activa, procesamos el desborde poco a poco para que se vea el reparto
        if pila_avalancha:
            if current_time - ultimo_reparto_time > VELOCIDAD_AVALANCHA:
                hubo_reparto = procesar_paso_avalancha()
                ultimo_reparto_time = current_time
        # Si no hay avalancha, añadimos el siguiente grano de arena (paso de tiempo normal)
        elif paso_actual < PASOS_TIEMPO:
            ejecutar_paso_tiempo()
            # Guardar estadística solo cuando el sistema está estable
            historial_carga.append(np.mean(cuadrgrid))
        
        # Si ya terminamos todos los pasos, pausar
        if paso_actual >= PASOS_TIEMPO and not pila_avalancha:
            pausado = True

    # --- RENDERIZADO (DIBUJO) ---
    # 1. Dibujar el grid completo con las bolitas
    dibujar_grid_completo(pantalla, cuadrgrid)
    
    # 2. Dibujar el panel de estadísticas y gráfica
    dibujar_panel_estadisticas(pantalla, cuadrgrid, paso_actual, historial_carga)
    
    # 3. Actualizar la pantalla (Flip buffers)
    pygame.display.flip()
    
    # Controlar FPS (para que no consuma 100% CPU innecesariamente)
    reloj.tick(60)

# Salida limpia
pygame.quit()
sys.exit()