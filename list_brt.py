import os
import geopandas as gpd

from pyproj import Transformer


df = gpd.read_file(os.path.join('data', 'Ondergrondseparkeergarages.gdb.zip'), layer='gebouwpunt')

for x in df['geometry']:
    value = list(x.coords[0])

    transformer = Transformer.from_crs("epsg:28992", "epsg:4326")
    out = transformer.transform(*value)

    print(out)
