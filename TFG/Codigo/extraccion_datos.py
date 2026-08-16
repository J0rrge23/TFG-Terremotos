import pandas as pd

# Descarga directa del catálogo del USGS en un DataFrame
url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?"
    "format=csv&starttime=2010-01-01&endtime=2023-01-01&"
    "minmagnitude=2.0&minlatitude=32.5&maxlatitude=36.5&"
    "minlongitude=-121.0&maxlongitude=-114.0"
)

df = pd.read_csv(url)

# Convertir tiempo a datetime
df['time'] = pd.to_datetime(df['time'])
print(df[['time', 'latitude', 'longitude', 'depth', 'mag']].head())