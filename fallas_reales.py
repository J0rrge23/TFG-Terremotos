import urllib.request
import os

# Base de datos global de fallas activas de la Fundación GEM (Global Earthquake Model)
url_gem = "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson"
archivo_destino = "qfaults.geojson"

print("Descargando base de datos oficial de fallas activas (GEM)...")

try:
    # Descarga directa desde el repositorio oficial de GEM
    urllib.request.urlretrieve(url_gem, archivo_destino)
    
    if os.path.exists(archivo_destino):
        size_mb = os.path.getsize(archivo_destino) / (1024 * 1024)
        print(f"¡Éxito! Archivo descargado correctamente: '{archivo_destino}' ({size_mb:.2f} MB)")
    else:
        print("[ERROR] No se pudo guardar el archivo.")

except Exception as e:
    print(f"[ERROR] Ocurrió un fallo durante la descarga: {e}")