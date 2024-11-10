from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import requests
import io
import numpy as np
from tqdm import tqdm
import ee
import os

ee.Authenticate()
ee.Initialize()

def ee_image_to_epsg3346(ee_image, scale):
    res = ee_image.reproject(crs='EPSG:3346', scale=scale)
    return res

def gee_image_to_np_iamge(ee_image, buffer, scale, bands=None):
    ee_image = ee_image_to_epsg3346(ee_image, scale)
    if bands:
        url = ee_image.getDownloadURL({
            'scale': scale,
            'region': buffer.getInfo(),
            'format': 'NPY',
            'bands': bands
        })
    else:
        url = ee_image.getDownloadURL({
            'scale': scale,
            'region': buffer.getInfo(),
            'format': 'NPY'
        })
    response = requests.get(url)
    gee_arr = np.load(io.BytesIO(response.content), allow_pickle=True)
    return gee_arr

def get_ee_sentinel_1(from_date, to_date, ee_geometry):
    res = ee.ImageCollection('COPERNICUS/S1_GRD')
    res = res.filterDate(from_date, to_date)
    res = res.filterBounds(ee_geometry).mean()
    return res

def mask_s2_clouds(image):
  qa = image.select('QA60')
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )
  return image.updateMask(mask)

def get_ee_sentinel_2(from_date, to_date, ee_geometry):
    res = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    res = res.filterDate(from_date, to_date)
    res = res.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
    res = res.map(mask_s2_clouds)
    res = res.filterBounds(ee_geometry).mean()
    return res

def mask_night_clouds(image):
    # Select the cloud-free coverage band
    qa = image.select('cf_cvg')
    # Define a threshold for minimum cloud-free observations
    cloud_free_threshold = 50
    # Mask out pixels with low cloud-free observation counts
    cloud_free_mask = qa.lt(cloud_free_threshold)
    return image.updateMask(cloud_free_mask)

def get_ee_night(from_date, to_date, ee_geometry):
    res = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
    res = res.filterDate(from_date, to_date)
    res = res.map(mask_night_clouds)
    res = res.filterBounds(ee_geometry).mean()
    return res

def get_ee_pop(ee_geometry):
    res = ee.FeatureCollection('projects/ginsed2019/assets/pop_density_per_100_m_x_100_m')
    res = res.reduceToImage(properties = ['POP'], reducer = ee.Reducer.mean())
    return res

def normalize_2d(array):
    max_ = array.max()
    min_ = array.min()
    array = (array - min_) / (max_ - min_)
    return array

def take_500(matrix):
    x, y = matrix.shape
    x = (x - 500) // 2
    y = (y - 500) // 2
    matrix = matrix[x:x+500,y:y+500]
    return matrix

with open('areas_of_interest.json', 'r', encoding='utf-8') as file: areass = json.load(file)
radius = 2600
scale = 10
areas = [ee.Geometry.Point([area[0], area[1]]).buffer(radius) for area in areass]

pop = []
vv = []
vh = []
b1 = []
b2 = []
b3 = []
b4 = []
b5 = []
b6 = []
b7 = []
b8 = []
b8a = []
b9 = []
b11 = []
b12 = []
night = []

i = 0
for area in areas:
    print(i)
    i = i + 1
    popp = gee_image_to_np_iamge(get_ee_pop(area), area, scale)
    # plt.imshow(popp['mean'])
    se1 = gee_image_to_np_iamge(get_ee_sentinel_1('2021-06-01', '2021-09-01', area), area, scale)
    # plt.imshow(se1['VV'])
    # plt.imshow(se1['VH'])
    se2 = gee_image_to_np_iamge(get_ee_sentinel_2('2021-06-01', '2021-09-01', area), area, scale, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
    # plt.imshow(se2['B1'])
    # plt.imshow(se2['B2'])
    # plt.imshow(se2['B3'])
    # plt.imshow(se2['B4'])
    # plt.imshow(se2['B5'])
    # plt.imshow(se2['B6'])
    # plt.imshow(se2['B7'])
    # plt.imshow(se2['B8'])
    # plt.imshow(se2['B8A'])
    # plt.imshow(se2['B9'])
    # plt.imshow(se2['B11'])
    # plt.imshow(se2['B12'])
    nigh = gee_image_to_np_iamge(get_ee_night('2021-12-01', '2022-03-01', area), area, scale)
    # plt.imshow(np.log1p(nigh['avg_rad']))
    
    pop.append(take_500(popp['mean']))
    vv.append(take_500(se1['VV']))
    vh.append(take_500(se1['VH']))
    b1.append(take_500(se2['B1']))
    b2.append(take_500(se2['B2']))
    b3.append(take_500(se2['B3']))
    b4.append(take_500(se2['B4']))
    b5.append(take_500(se2['B5']))
    b6.append(take_500(se2['B6']))
    b7.append(take_500(se2['B7']))
    b8.append(take_500(se2['B8']))
    b8a.append(take_500(se2['B8A']))
    b9.append(take_500(se2['B9']))
    b11.append(take_500(se2['B11']))
    b12.append(take_500(se2['B12']))
    night.append(take_500(nigh['avg_rad']))

np.save('data/pop.npy', np.array(pop))
np.save('data/vv.npy', np.array(vv))
np.save('data/vh.npy', np.array(vh))
np.save('data/b1.npy', np.array(b1))
np.save('data/b2.npy', np.array(b2))
np.save('data/b3.npy', np.array(b3))
np.save('data/b4.npy', np.array(b4))
np.save('data/b5.npy', np.array(b5))
np.save('data/b6.npy', np.array(b6))
np.save('data/b7.npy', np.array(b7))
np.save('data/b8.npy', np.array(b8))
np.save('data/b8a.npy', np.array(b8a))
np.save('data/b9.npy', np.array(b9))
np.save('data/b11.npy', np.array(b11))
np.save('data/b12.npy', np.array(b12))
np.save('data/night.npy', np.array(night))

np.save('data/areas.npy', np.array(areass))

import requests

def get_city_name(latitude, longitude):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        'lat': latitude,
        'lon': longitude,
        'format': 'json',
        'addressdetails': 1,
        'zoom': 10  # Adjust as needed
    }
    headers = {
        'User-Agent': 'YourAppName/1.0 (your-email@example.com)'  # Replace with your info
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        city_name = address.get("city") or address.get("town") or address.get("village") or "Unknown"
        return city_name
    else:
        print(f"Error: {response.status_code}")
        return None

names = [get_city_name(a[1], a[0]) for a in areass]
np.save('data/names.npy', np.array(names))
