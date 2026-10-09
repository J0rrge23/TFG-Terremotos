import numpy as np
import matplotlib.pyplot as plt

# Estilo gráfico profesional para el TFG
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# =============================================================================
# 1. DEFINICIÓN DE LOS MODELOS DE AUTÓMATAS CELULARES
# =============================================================================

def btw_relax(grid, z_c=4):
    """Relajación del modelo de Pila de Arena BTW (Variables discretas)."""
    grid = grid.copy()
    L = grid.shape[0]
    while True:
        unstable = grid >= z_c
        if not np.any(unstable):
            break
        for i in range(L):
            for j in range(L):
                if grid[i, j] >= z_c:
                    k = grid[i, j] // z_c
                    grid[i, j] %= z_c
                    if i > 0: grid[i-1, j] += k
                    if i < L-1: grid[i+1, j] += k
                    if j > 0: grid[i, j-1] += k
                    if j < L-1: grid[i, j+1] += k
    return grid

def ofc_relax(grid, F_c=1.0, alpha=0.20):
    """Relajación del modelo de Bloques y Muelles OFC (Variables continuas disipativas)."""
    grid = grid.copy()
    L = grid.shape[0]
    while True:
        unstable = grid >= F_c
        if not np.any(unstable):
            break
        new_grid = grid.copy()
        unstable_idx = np.argwhere(unstable)
        for i, j in unstable_idx:
            val = grid[i, j]
            new_grid[i, j] -= val
            transferred = alpha * val
            if i > 0: new_grid[i-1, j] += transferred
            if i < L-1: new_grid[i+1, j] += transferred
            if j > 0: new_grid[i, j-1] += transferred
            if j < L-1: new_grid[i, j+1] += transferred
        grid = new_grid
    return grid

# =============================================================================
# 2. CONFIGURACIÓN DEL EXPERIMENTO Y ESTADOS INICIALES
# =============================================================================

L = 25              # Tamaño de la red (25x25)
z_c = 4             # Pendiente crítica BTW
F_c = 1.0           # Esfuerzo crítico OFC
alpha = 0.20        # Parámetro de acoplamiento elástico OFC (< 0.25 = disipativo)

np.random.seed(42)

# Generar estado crítico estable inicial para BTW
grid_btw_0 = btw_relax(np.random.randint(0, z_c, size=(L, L)) + 15)

# Generar estado de esfuerzo alto inicial para OFC
grid_ofc_0 = np.random.uniform(0.70, 0.98, size=(L, L))

# Coordenadas de las dos perturbaciones A y B
pos_A = (6, 6)
pos_B = (18, 18)

# =============================================================================
# 3. EXPERIMENTO EN MODELO BTW (PILA DE ARENA)
# =============================================================================

# Secuencia A -> B
state_btw_A = grid_btw_0.copy()
state_btw_A[pos_A] += 1
state_btw_A = btw_relax(state_btw_A)
state_btw_AB = state_btw_A.copy()
state_btw_AB[pos_B] += 1
state_btw_AB = btw_relax(state_btw_AB)

# Secuencia B -> A
state_btw_B = grid_btw_0.copy()
state_btw_B[pos_B] += 1
state_btw_B = btw_relax(state_btw_B)
state_btw_BA = state_btw_B.copy()
state_btw_BA[pos_A] += 1
state_btw_BA = btw_relax(state_btw_BA)

diff_btw = np.abs(state_btw_AB - state_btw_BA)

# =============================================================================
# 4. EXPERIMENTO EN MODELO OFC (BLOQUES Y MUELLES)
# =============================================================================

# Secuencia A -> B
state_ofc_A = grid_ofc_0.copy()
state_ofc_A[pos_A] += 0.30
state_ofc_A = ofc_relax(state_ofc_A)
state_ofc_AB = state_ofc_A.copy()
state_ofc_AB[pos_B] += 0.30
state_ofc_AB = ofc_relax(state_ofc_AB)

# Secuencia B -> A
state_ofc_B = grid_ofc_0.copy()
state_ofc_B[pos_B] += 0.30
state_ofc_B = ofc_relax(state_ofc_B)
state_ofc_BA = state_ofc_B.copy()
state_ofc_BA[pos_A] += 0.30
state_ofc_BA = ofc_relax(state_ofc_BA)

diff_ofc = np.abs(state_ofc_AB - state_ofc_BA)

# =============================================================================
# 5. IMPRESIÓN DE RESULTADOS EN CONSOLA
# =============================================================================

print("=========================================================")
print("  RESULTADOS DEL EXPERIMENTO DE CONMUTATIVIDAD")
print("=========================================================")
print(f"Diferencia Máxima en BTW  |S_AB - S_BA|: {np.max(diff_btw):.6f}")
print(f"Diferencia Máxima en OFC  |S_AB - S_BA|: {np.max(diff_ofc):.6f}")
print("---------------------------------------------------------")
if np.max(diff_btw) == 0:
    print("=> MODELO BTW: ABELIANO (El orden de perturbación NO altera el estado).")
if np.max(diff_ofc) > 0:
    print("=> MODELO OFC: NO ABELIANO (El orden de perturbación SÍ altera el estado).")
print("=========================================================")

# =============================================================================
# 6. GENERACIÓN DE LA FIGURA PUBLICABLE
# =============================================================================

fig, axs = plt.subplots(2, 3, figsize=(14, 9), dpi=300)

# --- Fila 1: Modelo BTW ---
im1 = axs[0, 0].imshow(state_btw_AB, cmap='YlOrRd', vmin=0, vmax=3)
axs[0, 0].set_title(r"BTW: Estado Final Secuencia $A \to B$", fontsize=11, fontweight='bold')
plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)

im2 = axs[0, 1].imshow(state_btw_BA, cmap='YlOrRd', vmin=0, vmax=3)
axs[0, 1].set_title(r"BTW: Estado Final Secuencia $B \to A$", fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)

im3 = axs[0, 2].imshow(diff_btw, cmap='Blues', vmin=0, vmax=1)
axs[0, 2].set_title(r"Diferencia $|S_{AB} - S_{BA}| = 0$" + "\n(CARÁCTER ABELIANO)", fontsize=11, fontweight='bold', color='darkgreen')
plt.colorbar(im3, ax=axs[0, 2], fraction=0.046, pad=0.04)

# --- Fila 2: Modelo OFC ---
im4 = axs[1, 0].imshow(state_ofc_AB, cmap='magma', vmin=0, vmax=1.0)
axs[1, 0].set_title(r"OFC: Estado Final Secuencia $A \to B$", fontsize=11, fontweight='bold')
plt.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)

im5 = axs[1, 1].imshow(state_ofc_BA, cmap='magma', vmin=0, vmax=1.0)
axs[1, 1].set_title(r"OFC: Estado Final Secuencia $B \to A$", fontsize=11, fontweight='bold')
plt.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04)

im6 = axs[1, 2].imshow(diff_ofc, cmap='Reds', vmin=0, vmax=np.max(diff_ofc))
# Solución al bug: duplicar llaves {{AB}} y {{BA}} en f-strings de LaTeX
axs[1, 2].set_title(f"Diferencia $|S_{{AB}} - S_{{BA}}| \\neq 0$\n(CARÁCTER NO ABELIANO, Máx = {np.max(diff_ofc):.2f})", fontsize=11, fontweight='bold', color='darkred')
plt.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04)

# Ajustes estéticos finales
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Demostración Numérica: Carácter Abeliano (BTW) vs. No Abeliano (OFC)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("demostracion_abeliano_vs_no_abeliano.pdf", dpi=300, bbox_inches='tight')
plt.savefig("demostracion_abeliano_vs_no_abeliano.png", dpi=300, bbox_inches='tight')
plt.show()