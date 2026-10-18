import os
import urllib.request

url_usgs = "https://earthquake.usgs.gov/static/lrgis/kml/USA_Quaternary_Faults.geojson"
nombre_archivo = "qfaults.geojson"

print("Descargando mapa de fallas oficial de la USGS...")

try:
    # Añadimos una cabecera User-Agent para evitar el bloqueo 403
    req = urllib.request.Request(
        url_usgs, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    with urllib.request.urlopen(req) as response, open(nombre_archivo, 'wb') as out_file:
        out_file.write(response.read())

    if os.path.exists(nombre_archivo):
        tamano_mb = os.path.getsize(nombre_archivo) / (1024 * 1024)
        print(f"[ÉXITO] Archivo descargado correctamente: '{nombre_archivo}' ({tamano_mb:.2f} MB)")

except Exception as e:
    print(f"[ERROR] Ocurrió un problema durante la descarga: {e}")