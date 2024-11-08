# https://open-data-ls-osp-sdg.hub.arcgis.com/pages/suraym-duomenys
# https://open-data-ls-osp-sdg.hub.arcgis.com/search?collection=Dataset&tags=census2021
# https://open-data-ls-osp-sdg.hub.arcgis.com/search?collection=Dataset&q=ezerai

import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import math
from pyproj import Transformer

def read_geojson_as_df(path):
    with open(path, 'r', encoding='utf-8') as file: data = json.load(file)
    data = data['features']
    data = [{'POP': d['properties']['POP'], 'polygon': d['geometry']['coordinates'][0]} for d in data]
    df = pd.DataFrame(data)
    return df

def read_geojson_as_df_only_geo(path):
    with open(path, 'r', encoding='utf-8') as file: data = json.load(file)
    data = data['features']
    data = [{'polygon': d['geometry']['coordinates'][0]} for d in data]
    df = pd.DataFrame(data)
    return df

def to_polygon(polygon):
    try:
        return Polygon(polygon)
    except Exception:
        return unary_union([Polygon(p) for p in polygon])

def as_gpd(df):
    df['id'] = df.index
    geometry = [to_polygon(polygon) for polygon in df['polygon']]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf = gdf.set_crs(epsg=4326, allow_override=True)
    gdf = gdf.to_crs(epsg=3346)
    gdf['polygon'] = gdf['geometry']
    gdf['area'] = gdf['polygon'].apply(lambda x: x.area)
    return gdf

def gdf_get_polygon(df, polygon):
    polygon = Polygon(polygon)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True)
    polygon = Polygon([transformer.transform(x, y) for x, y in polygon.exterior.coords])
    return df[df.geometry.within(polygon)].reset_index(drop=True)

def gdf_plot(df, column=None, polygon=None, add_0=True, as_log=True):
    df = df.copy()
    if polygon:
        df = gdf_get_polygon(df, polygon)
    if polygon and add_0:
        empty_data = as_gpd(pd.DataFrame({column: [0], 'polygon': [polygon]}))
        df = pd.concat([empty_data, df], ignore_index=True).reset_index(drop=True)
    if as_log and column:
        df[column] = np.log1p(df[column])
    if column:
        df.plot(column=column)
    else: df.plot()
    return None

def polygon_overlap(a, b):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)): return Polygon()
    res = a.intersection(b)
    return res
    

gyv_1000 = as_gpd(read_geojson_as_df('gyventojų_ir_būstų_surašymas_2021_gyventojų_skaičius_gardelės_1_km_x_1_km.geojson'))
gyv_0500 = as_gpd(read_geojson_as_df('gyventojų_ir_būstų_surašymas_2021_gyventojų_skaičius_gardelės_500_m_x_500_m.geojson'))
gyv_0250 = as_gpd(read_geojson_as_df('gyventojų_ir_būstų_surašymas_2021_gyventojų_skaičius_gardelės_250_m_x_250_m.geojson'))
gyv_0100 = as_gpd(read_geojson_as_df('gyventojų_ir_būstų_surašymas_2021_gyventojų_skaičius_gardelės_100_m_x_100_m.geojson'))
water = as_gpd(read_geojson_as_df_only_geo('ežerų_tvenkinių_ir_dirbtinių_nepratekamų_paviršinio_vandens_telkinių_duomenys.geojson'))

gdf_plot(gyv_1000, "POP")
gdf_plot(gyv_0500, "POP")
gdf_plot(gyv_0250, "POP")
gdf_plot(gyv_0100, "POP")
gdf_plot(water)


vilnius = [[25.170424, 54.752536], [25.162179, 54.618375], [25.387549, 54.621953], [25.371746, 54.783433]]

gdf_plot(gyv_1000, "POP", vilnius)
gdf_plot(gyv_0500, "POP", vilnius)
gdf_plot(gyv_0250, "POP", vilnius)
gdf_plot(gyv_0100, "POP", vilnius)
gdf_plot(water, None, vilnius, False)

kaunas = [[23.738269, 54.971641], [23.724527, 54.793102], [24.107931, 54.786768], [24.124422, 54.974005]]

gdf_plot(gyv_1000, "POP", kaunas)
gdf_plot(gyv_0500, "POP", kaunas)
gdf_plot(gyv_0250, "POP", kaunas)
gdf_plot(gyv_0100, "POP", kaunas)
gdf_plot(water, None, kaunas, False)


def gdf_diff_hellper(group):
    group['polygon_a_b'] = group['polygon_a'].intersection(group['polygon_b'])
    group['area_a_b'] = group['polygon_a_b'].area
    group['area_p_a_b'] = group['area_a_b'] / group['area_a']
    group['POP_a_b'] = group['POP_a'] * group['area_p_a_b']
    
    intersection_pop = group['POP_a_b'].sum()
    small_geo = unary_union(group['polygon_a'])
    big_geo = group['polygon_b'].to_numpy()[0]
    big_pop = group['POP_b'].to_numpy()[0]
    
    new_geo = big_geo.difference(small_geo)
    new_pop = big_pop - intersection_pop
    new_pop = 0 if new_pop < 0 else new_pop
    
    res = {'POP': new_pop, 'polygon': new_geo, 'area': new_geo.area, 'id': group['id_b'].to_numpy()[0]}
    
    return res

def gdf_diff(b, a):
    merged = gpd.sjoin(b, a, how='left', predicate='intersects', lsuffix = 'b', rsuffix='a')
    diff = merged.groupby('id_b').apply(lambda group: gdf_diff_hellper(group)).tolist()
    diff = pd.DataFrame(diff)
    diff = gpd.GeoDataFrame(diff, geometry=diff['polygon'])
    
    diff = diff[diff['area'] >= 10**2].reset_index(drop=True)
    return diff

def scale_pop_to_100(df):
    df = df.copy()
    df['POP'] = df['POP'] / (df['area'] / (100**2))
    return(df)
    
diff_0250_0100 = gdf_diff(gyv_0250, gyv_0100)
diff_0500_0250 = gdf_diff(gyv_0500, gyv_0250)
diff_1000_0500 = gdf_diff(gyv_1000, gyv_0500)

full_df =  pd.concat([scale_pop_to_100(gyv_0100), scale_pop_to_100(diff_0250_0100), scale_pop_to_100(diff_0500_0250), scale_pop_to_100(diff_1000_0500)], ignore_index=True).reset_index(drop=True)

gdf_plot(full_df, "POP")
gdf_plot(full_df, "POP", kaunas)
gdf_plot(full_df, "POP", vilnius)

full_df.to_file('pop_density_per_100_m_x_100_m.geojson', driver="GeoJSON")
full_df[['POP', 'geometry']].to_file('pop_density_per_100_m_x_100_m_shp/pop_density_per_100_m_x_100_m.shp', driver='ESRI Shapefile')
