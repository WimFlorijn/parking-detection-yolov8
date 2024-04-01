import json
import os
import geopandas as gpd

from shapely.geometry import shape

filter_underground = True

with open(os.path.join('data', 'amsterdam.geojson')) as fh:
    bbox = shape(json.load(fh))

parking_garages = gpd.read_file(os.path.join('data', 'parking.geojson'))
outside_bbox = parking_garages.index[parking_garages['geometry'].apply(lambda x: x.within(bbox)) == False]
parking_garages.drop(outside_bbox, inplace=True)

if filter_underground:
    parking_garages.drop(parking_garages.index[parking_garages['parking'] != 'underground'], inplace=True)

parking_entrances = gpd.read_file(os.path.join('data', 'parking_entrance.geojson'))
outside_bbox = parking_entrances.index[parking_entrances['geometry'].apply(lambda x: x.within(bbox)) == False]
parking_entrances.drop(outside_bbox, inplace=True)

if filter_underground:
    parking_entrances.drop(parking_entrances.index[parking_entrances['parking'] != 'underground'], inplace=True)

print(f'Writing {len(parking_garages)} garages, and {len(parking_entrances)} entrances.')

parking_garages.to_file("package.gpkg", layer='garages', driver="GPKG")
parking_entrances.to_file("package.gpkg", layer='garage-entrances', driver="GPKG")

for _, x in parking_entrances.iterrows():
    print(list(x.geometry.coords[0])[::-1])
