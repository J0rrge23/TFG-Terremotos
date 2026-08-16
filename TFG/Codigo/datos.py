import pandas as pd

# -------------------------------------------------------------
# 1. DATOS DE BAJA CALIFORNIA (MÉXICO)
# -------------------------------------------------------------
url_baja = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?"
    "format=csv&"
    "starttime=2020-01-01&"
    "endtime=2024-01-01&"
    "minmagnitude=2.0&"
    "minlatitude=22.5&maxlatitude=32.5&"
    "minlongitude=-118.0&maxlongitude=-109.0"
)

df_baja = pd.read_csv(url_baja)
print(f"Baja California: {len(df_baja)} terremotos cargados.")
df_baja.to_csv("terremotos_baja_california.csv", index=False)

# -------------------------------------------------------------
# 2. DATOS DEL SUR DE ESPAÑA
# -------------------------------------------------------------
url_espana = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?"
    "format=csv&"
    "starttime=2020-01-01&"
    "endtime=2024-01-01&"
    "minmagnitude=2.0&"
    "minlatitude=35.0&maxlatitude=39.0&"
    "minlongitude=-10.0&maxlongitude=0.0"
)

df_espana = pd.read_csv(url_espana)
print(f"Sur de España: {len(df_espana)} terremotos cargados.")
df_espana.to_csv("terremotos_sur_espana.csv", index=False)