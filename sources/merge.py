import requests
import pandas as pd
import geopandas as gpd

from shapely import wkt
from pyproj import Transformer


location_to_details = {}

source = 'osm'
for _, row in gpd.read_file(r'osm-garages.gpkg', layer='garages').iterrows():
    geometry = row['geometry']
    geometry = list(geometry.coords[0])[::-1]

    transformer = Transformer.from_crs("epsg:4326", "epsg:28992")
    point = transformer.transform(*geometry)

    location_to_details[point] = {
        'source': source,
        'x': point[0],
        'y': point[1],
        'type': 'garage',
        'name': row.get('alt_name'),
        'city': row.get('addr:city'),
        'country': row.get('addr:country'),
        'housenumber': row.get('addr:housenumber'),
        'postcode': row.get('addr:postcode'),
        'street': row.get('addr:street'),
    }

for _, row in gpd.read_file(r'osm-garages.gpkg', layer='garage-entrances').iterrows():
    geometry = row['geometry']
    geometry = list(geometry.coords[0])[::-1]

    transformer = Transformer.from_crs("epsg:4326", "epsg:28992")
    point = transformer.transform(*geometry)

    location_to_details[point] = {
        'source': source,
        'x': point[0],
        'y': point[1],
        'type': 'garage-ingang',
        'name': row.get('alt_name'),
        'city': row.get('addr:city'),
        'country': row.get('addr:country'),
        'housenumber': row.get('addr:housenumber'),
        'postcode': row.get('addr:postcode'),
        'street': row.get('addr:street'),
    }

for _, row in pd.read_excel('Parkeergarages_BRT.xlsx').iterrows():
    location_to_details[(row['x'], row['y'])] = {
        'source': 'brt',
        'x': row['x'],
        'y': row['y'],
    }

LOOKUP_URL = 'http://geodata.nationaalgeoregister.nl/locatieserver/v3/lookup?id={}'
SUGGEST_URL = 'http://geodata.nationaalgeoregister.nl/locatieserver/v3/suggest?q={} and type:adres'

p200 = pd.read_excel('meldingenparkeergarages uniek. ingevuld.xlsx')
for arguments in p200['Adres'][:75].values:
    try:
        url = SUGGEST_URL.format(arguments)
        object_id = requests.get(url).json()['response']['docs'][0]['id']

        url = LOOKUP_URL.format(object_id)
        geometry = requests.get(url).json()['response']['docs'][0]['centroide_rd']
        geometry = wkt.loads(geometry).coords[0]

        location_to_details[geometry] = {
            'source': 'p2000',
            'x': geometry[0],
            'y': geometry[1],
        }
    except Exception as e:
        print(e)
        geometry = None

print(location_to_details)

data = list(location_to_details.values())
df = pd.DataFrame(data)
df.to_csv('garages.csv', index=False)

df = gpd.GeoDataFrame(data)
