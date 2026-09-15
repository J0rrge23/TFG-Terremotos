import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calcular_zaliapin_nn(df, b=1.0, d=1.6, m0=2.5):
    """
    df: DataFrame con columnas ['t', 'x', 'y', 'mag'] ordenado por tiempo t
    b: Exponente b de Gutenberg-Richter
    d: Dimensión fractal espacial
    """
    N = len(df)
    t = df['t'].to_numpy()
    coords = df[['x', 'y']].to_numpy()
    mag = df['mag'].to_numpy()
    
    T_star = np.zeros(N)
    R_star = np.zeros(N)
    eta_star = np.zeros(N)
    parent_idx = np.zeros(N, dtype=int)
    
    # El primer evento es la raíz del árbol
    eta_star[0], T_star[0], R_star[0], parent_idx[0] = np.nan, np.nan, np.nan, -1
    
    for j in range(1, N):
        dt = t[j] - t[:j] # Solo eventos del pasado i < j
        dr = np.linalg.norm(coords[j] - coords[:j], axis=1)
        
        # Métrica normalizada
        T_ij = dt * (10 ** (-0.5 * b * (mag[:j] - m0)))
        R_ij = (dr ** d) * (10 ** (-0.5 * b * (mag[:j] - m0)))
        eta_ij = T_ij * R_ij
        
        # Selección del vecino más cercano
        i_min = np.argmin(eta_ij)
        eta_star[j] = eta_ij[i_min]
        T_star[j] = T_ij[i_min]
        R_star[j] = R_ij[i_min]
        parent_idx[j] = i_min
        
    df['log10_eta'] = np.log10(eta_star)
    df['log10_T'] = np.log10(T_star)
    df['log10_R'] = np.log10(R_star)
    df['parent_idx'] = parent_idx
    
    return df