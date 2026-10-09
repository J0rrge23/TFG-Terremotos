import numpy as np
import pandas as pd
import networkx as nx

def identificar_clusters(df, log10_eta_0=-5.0):
    """
    Agrupa eventos en familias (árboles) mediante el umbral log10_eta_0.
    Clasifica cada evento en: 'mainshock', 'aftershock', 'foreshock' o 'background'.
    """
    N = len(df)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    
    mask = (df['log10_eta'] <= log10_eta_0) & (df['parent_idx'] >= 0)
    hijos = df.index[mask].values
    padres = df['parent_idx'][mask].values
    
    for h, p in zip(hijos, padres):
        G.add_edge(p, h)
        
    G_undirected = G.to_undirected()
    componentes = list(nx.connected_components(G_undirected))
    
    event_type = np.full(N, 'background', dtype=object)
    cluster_id = np.full(N, -1, dtype=int)
    mainshock_idx_col = np.full(N, -1, dtype=int)
    
    c_id = 0
    for comp in componentes:
        nodes = list(comp)
        if len(nodes) == 1:
            # Evento independiente de fondo
            continue
            
        sub_mags = df.loc[nodes, 'mag'].values
        sub_times = df.loc[nodes, 'datetime'].values
        
        max_mag_local_idx = np.argmax(sub_mags)
        m_idx = nodes[max_mag_local_idx]
        m_time = sub_times[max_mag_local_idx]
        
        for node in nodes:
            cluster_id[node] = c_id
            mainshock_idx_col[node] = m_idx
            
            if node == m_idx:
                event_type[node] = 'mainshock'
            elif df.loc[node, 'datetime'] > m_time:
                event_type[node] = 'aftershock'
            else:
                event_type[node] = 'foreshock'
                
        c_id += 1
        
    df['event_type'] = event_type
    df['cluster_id'] = cluster_id
    df['mainshock_idx'] = mainshock_idx_col
    return df