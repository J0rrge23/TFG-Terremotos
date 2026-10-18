import json
import folium

# 1. Cargar el archivo GeoJSON
file_path = "fallas.geojson"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
except Exception as e:
    print(f"Error al cargar '{file_path}': {e}")
    exit()

# 2. Función para clasificar colores según movimiento
def get_color(slip_type):
    if not slip_type:
        return "#999999"
    s = str(slip_type).lower()
    if any(k in s for k in ["dextral", "sinistral", "strike"]):
        return "#e41a1c"  # Rojo: Lateral / Strike-Slip
    elif any(k in s for k in ["reverse", "thrust"]):
        return "#377eb8"  # Azul: Inversa / Cabalgamiento
    elif "normal" in s:
        return "#4daf4a"  # Verde: Normal
    return "#984ea3"      # Morado: Otro

def style_function(feature):
    props = feature.get("properties", {})
    return {
        "color": get_color(props.get("slip_type", "")),
        "weight": 3,
        "opacity": 0.85
    }

# 3. Inicializar el mapa con Esri Dark Gray (100% Libre, sin API Key ni marcas de agua)
m = folium.Map(
    location=[34.5, -117.0],
    zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, NGA, USGS",
    name="Fondo Oscuro (Esri Dark)"
)

# 4. Capas secundarias libres (OpenStreetMap y Esri Satélite)
folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap (Estándar)"
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, Maxar, Earthstar Geographics",
    name="Satélite (Esri World Imagery)"
).add_to(m)

# 5. Agregar la capa GeoJSON con datos reales
geojson_layer = folium.GeoJson(
    geojson_data,
    name="Fallas Geológicas",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "slip_type", "net_slip_rate"],
        aliases=["Falla:", "Movimiento:", "Tasa (mm/año):"],
        localize=True,
        sticky=True
    ),
    popup=folium.GeoJsonPopup(
        fields=["name", "catalog_name", "slip_type", "net_slip_rate", "upper_seis_depth", "lower_seis_depth", "average_dip"],
        aliases=["Nombre:", "Catálogo:", "Tipo de Movimiento:", "Tasa Deslizamiento (mm/año):", "Prof. Sup (km):", "Prof. Inf (km):", "Buzamiento (°):"]
    )
).add_to(m)

# 6. Leyenda flotante para el TFG
legend_html = '''
<div style="
    position: fixed; 
    bottom: 30px; left: 30px; width: 230px; height: 145px; 
    background-color: rgba(20, 20, 20, 0.9);
    border:2px solid #555; z-index:9999; font-size:12px;
    color: white; padding: 10px; border-radius: 8px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.5);
    font-family: sans-serif;
    ">
    <b style="font-size: 13px;">Mecanismo de Falla</b><br><br>
    <i style="background: #e41a1c; width: 18px; height: 4px; display: inline-block; margin-right: 8px; vertical-align: middle;"></i> Lateral / Strike-Slip<br>
    <i style="background: #377eb8; width: 18px; height: 4px; display: inline-block; margin-right: 8px; vertical-align: middle;"></i> Inversa / Cabalgamiento<br>
    <i style="background: #4daf4a; width: 18px; height: 4px; display: inline-block; margin-right: 8px; vertical-align: middle;"></i> Normal<br>
    <i style="background: #984ea3; width: 18px; height: 4px; display: inline-block; margin-right: 8px; vertical-align: middle;"></i> Otro / Indefinido<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 7. Encuadrar y exportar
m.fit_bounds(geojson_layer.get_bounds())
folium.LayerControl().add_to(m)

output_file = "mapa_fallas_interactivo.html"
m.save(output_file)
print(f"Mapa limpio y sin marcas de agua generado en '{output_file}'")