# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:27:16 2018

@author: pp
"""

import pandas as pd
from time import sleep

from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pymongo import MongoClient

country=[]
name=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweet_with_geo.find({},{"place":1}):
    country.append(doc['place']['country'])
    name.append(doc['place']['name'])

places = pd.DataFrame({'city': name, 'country': country})

places['lat'] = None
places['lon'] = None

geolocator = Nominatim()
for index, row in places.iterrows():
    if index > 1347:
            counter = counter + 1
            city = row['city']
            location = geolocator.geocode(city)
            if hasattr(location, 'latitude') and hasattr(location, 'longitude'):
                row['lat'] = location.latitude
                row['lon'] = location.longitude
                print location.latitude, location.longitude, index
            else:
                row['lat'] = None
                row['lon'] = None
                print "Invalid Address", index
            sleep(0.8)
    else:
        pass

places = places.dropna(how='any')
places.to_csv('geo.csv')

fix, ax = plt.subplots(figsize=(14, 8))
earth = Basemap()
earth.bluemarble(alpha = 0.3)
earth.drawcoastlines(color ='#555566', linewidth = 1)
ax.scatter(places['lon'], places['lat'], 8, c = 'red', alpha = 0.5, zorder = 10)
ax.set_xlabel("Tweets about #Facebook")
plt.savefig('tweets2.png', dpi = 350, bbox_inches = 'tight')
