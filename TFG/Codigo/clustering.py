import numpy as np

def identificar_clusters(df, log10_eta_0=-5.0):
    """Clasifica eventos en 'aftershock' o 'background'."""
    df['event_type'] = np.where(df['log10_eta'] <= log10_eta_0, 'aftershock', 'background')
    return df