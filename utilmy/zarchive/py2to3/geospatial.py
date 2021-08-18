# -*- coding: utf-8 -*-
"""GEO MAP API """
###########################################################################################








####### Google Distance ####################################################################################
# https://iliauk.com/2015/12/18/data-mining-google-places-cafe-nero-example

import urllib
import csv
import time
import math
import requests
import itertools
 
 
API_KEY = [
]
shops_list = []
debug_list = []
SAVE_PATH = 'H:/'
COMPANY_SEARCH = 'Cafe Nero'
RADIUS_KM = 0.5
LIMIT = 60
 
 
class coordinates_box(object):
    """
    Initialise a coordinates_box class which will hold the produced coordinates and
    output a html map of the search area
    """
    def __init__(self):
        self.coordset = []
 
    def createcoordinates(self,
                          southwest_lat,
                          southwest_lng,
                          northeast_lat,
                          northeast_lng):
        """
        Based on the input radius this tesselates a 2D space with circles in
        a hexagonal structure
        """
        earth_radius_km = 6371
        lat_start = math.radians(southwest_lat)
        lon_start = math.radians(southwest_lng)
        lat = lat_start
        lon = lon_start
        lat_level = 1
        while True:
            if (math.degrees(lat) <= northeast_lat) & (math.degrees(lon) <= northeast_lng):
                self.coordset.append([math.degrees(lat), math.degrees(lon)])
            parallel_radius = earth_radius_km * math.cos(lat)
            if math.degrees(lat) > northeast_lat:
                break
            elif math.degrees(lon) > northeast_lng:
                lat_level += 1
                lat += (RADIUS_KM / earth_radius_km) + (RADIUS_KM / earth_radius_km) * math.sin(math.radians(30))
                if lat_level % 2 != 0:
                    lon = lon_start
                else:
                    lon = lon_start + (RADIUS_KM / parallel_radius) * math.cos(math.radians(30))
            else:
                lon += 2 * (RADIUS_KM / parallel_radius) * math.cos(math.radians(30))
 
        print('Coordinates-set contains %d coordinates' % len(self.coordset))
        # Save coordinates:
        f = open(SAVE_PATH + 'circles_' + COMPANY_SEARCH + '_python_mined.csv', 'w', newline='')
        w = csv.writer(f)
        for coord in self.coordset:
            w.writerow(coord)
        f.close()
        # LOG MAP
        self.htmlmaplog(SAVE_PATH + 'htmlmaplog_' + COMPANY_SEARCH + '.html')
 
    def htmlmaplog(self,
                   map_save_path):
        """
        Outputs a HTML map
        """
        htmltext = """
        <!DOCTYPE html >
          <style type="text/css">
                    html, body {
                        height: 100%;
                        width: 100%;
                        padding: 0px;
                        margin: 0px;
                    }
        </style>
        <head>
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
        <meta http-equiv="content-type" content="text/html; charset=UTF-8"/>
        <title>Boundary Partitioning</title>
        <xml id="myxml">
        <markers>
        """
        # Content
        for coord in self.coordset:
            rowcord = '<marker name = "' + COMPANY_SEARCH + '" lat = "' + \
                      '%.5f' % coord[0] + '" lng = "' + '%.5f' % coord[1] + '"/>\n'
            htmltext += rowcord
        # Bottom
        htmltext += """
        </markers>
        </xml>
        <script type="text/javascript" src="https://maps.googleapis.com/maps/api/js?&sensor=false&libraries=geometry"></script>
        <script type="text/javascript">
        var XML = document.getElementById("myxml");
        if(XML.documentElement == null)
        XML.documentElement = XML.firstChild;
        var MARKERS = XML.getElementsByTagName("marker");
        """
        htmltext += "var RADIUS_KM = " + str(RADIUS_KM) + ";"
        htmltext += """
        var map;
        var geocoder = new google.maps.Geocoder();
        var counter = 0
        function load() {
            // Initialize around City, London
            var my_lat = 51.518175;
            var my_lng = -0.129064;
            var mapOptions = {
                    center: new google.maps.LatLng(my_lat, my_lng),
                    zoom: 12
            };
            map = new google.maps.Map(document.getElementById('map'),
                mapOptions);
            var bounds = new google.maps.LatLngBounds();
            for (var i = 0; i < MARKERS.length; i++) {
                var name = MARKERS[i].getAttribute("name");
                var point_i = new google.maps.LatLng(
                    parseFloat(MARKERS[i].getAttribute("lat")),
                    parseFloat(MARKERS[i].getAttribute("lng")));
                var icon = {icon: 'http://labs.google.com/ridefinder/images/mm_20_gray.png'};
                var col = '#0033CC';
                var draw_circle = new google.maps.Circle({
                    center: point_i,
                    radius: RADIUS_KM*1000,
                    strokeColor: col,
                    strokeOpacity: 0.15,
                    strokeWeight: 2,
                    fillColor: col,
                    fillOpacity: 0.15,
                    map: map
                });
                var marker = new google.maps.Marker({
                    position: point_i,
                    map: map,
                    icon: 'https://maps.gstatic.com/intl/en_us/mapfiles/markers2/measle_blue.png'
                })
                bounds.extend(point_i);
            };
            map.fitBounds(bounds);
        }
        </script>
        </head>
        <body onload="load()">
        <center>
        <div style="padding-top: 20px; padding-bottom: 20px;">
        <div id="map" style="width:90%; height:1024px;"></div>
        </center>
        </body>
        </html>
        """
        with open(map_save_path, 'w') as f:
            f.write(htmltext)
        f.close()
 
 
class counter(object):
    """
    Counter class to keep track of the requests usage
    """
    def __init__(self):
        self.keynum = 0
        self.partition_num = 0
        self.detailnum = 0
 
    def increment_key(self):
        self.keynum += 1
 
    def increment_partition(self):
        self.partition_num += 1
 
    def increment_detail(self):
        self.detailnum += 1
 
 
def googleplaces(lat,
                 lng,
                 radius_metres,
                 search_term,
                 key,
                 pagetoken=None,
                 nmbr_returned=0):
    """
    Function uses the 'nearbysearch', however it is possible to use the radar-search and others
    located here: https://developers.google.com/places/web-service/search
    The API call returns a page_token for the next page up to a total of 60 results
    """
    location = urllib.parse.quote("%.5f,%.5f" % (lat,lng))
    radius = float(radius_metres)
    name = urllib.parse.quote(str(search_term))
 
    search_url = ('https://maps.googleapis.com/maps/api/place/' + 'nearbysearch' +
                  '/json?location=%s&radius=%d&keyword=%s&key=%s') % (location, radius, name, key)
    if pagetoken is not None:
        search_url += '&pagetoken=%s' % pagetoken
        # SLEEP so that request is generated
        time.sleep(2)
 
    time.sleep(0.1)
    req_count.increment_key()
    print("Search number %d: %s" % (req_count.keynum, search_url))
    google_search_request = requests.get(search_url)
    search_json_data = google_search_request.json()
 
    print(search_json_data['status'])
    if search_json_data['status'] == 'OK':
        nmbr_returned += len(search_json_data['results'])
        for place in search_json_data['results']:
            shop = [place['name'].encode('ascii', 'ignore').decode('ascii'),
                    place['vicinity'].encode('ascii', 'ignore').decode('ascii'),
                    place['geometry']['location']['lat'],
                    place['geometry']['location']['lng'],
                    place['types'],
                    place['place_id']]
            if shop not in shops_list:
                shops_list.append(shop)
        # Possible to get up to 60 results
        # from one search by passing next_page_token
        try:
            next_token = search_json_data['next_page_token']
            googleplaces(lat=lat,
                         lng=lng,
                         radius_metres=radius_metres,
                         search_term=search_term,
                         key=key,
                         pagetoken=next_token,
                         nmbr_returned=nmbr_returned)
            return
        except KeyError:
            pass
    elif search_json_data['status'] == 'ZERO_RESULTS':
        pass
    else:
        try:
            print('Error: %s' % search_json_data['error_message'])
        except KeyError:
            print('Unknown error message - check URL')
 
    debug_list.append([lat, lng, nmbr_returned])
    print('Partition %s no. %d/%d - found %d stores' %
          (location, req_count.partition_num, len(coord.coordset), nmbr_returned))
    if nmbr_returned >= LIMIT:
        print('Warning possible cut-off')
    print('List contains %d stores with key number: %d' % (len(shops_list), (req_count.keynum // 900)))
 
 
def googledetails(place_id,
                  key):
    """
    Function uses the miend place_ids to get further data from the details API
    """
    detail_url = ('https://maps.googleapis.com/maps/api/place/' + 'details' +
                  '/json?placeid=%s&key=%s') % (place_id, key)
    print(detail_url)
    google_detail_request = requests.get(detail_url)
    detail_json_data = google_detail_request.json()
    time.sleep(0.1)
 
    if detail_json_data['status'] == 'OK':
        try:
            address_components = detail_json_data['result']['address_components']
            print(address_components)
            # At the moment care only about extracting postcode, however possible to get:
            # Street number, Town, etc.
            for x in address_components:
                if x['types'] == ["postal_code"]:
                    postcode = x['long_name'].encode('ascii', 'ignore').decode('ascii')
                    break
                postcode = 'Nan'
        except KeyError:
            postcode = 'NaN'
        try:
            formatted_address = detail_json_data['result']['formatted_address'].encode('ascii', 'ignore').decode('ascii')
        except KeyError:
            formatted_address = 'NaN'
        try:
            website = detail_json_data['result']['website'].encode('ascii', 'ignore').decode('ascii')
        except KeyError:
            website = 'NaN'
        detail = [postcode, formatted_address, website]
    else:
        detail = detail_json_data['status'].encode('ascii', 'ignore').decode('ascii')
 
    print(detail)
    return detail
 
 
def fillindetails(f=SAVE_PATH + COMPANY_SEARCH + '_python_mined.csv'):
    """
    Opens the produced CSV and extracts the place ID for querying 
    """
    detailed_stores_out = []
    simple_stores_out = []
    with open(f, 'r') as csvin:
        reader = csv.reader(csvin)
 
        for store in reader:
            req_count.increment_detail()
            key_number = (req_count.keynum // 950)
 
            detailed_store = googledetails(store[5],  API_KEY[key_number])
            print('Row number %d/%d, store info: %s' % (req_count.detailnum, len(shops_list), detailed_store))
            detailed_stores_out.append(detailed_store)
            simple_stores_out.append(store)
 
    # OUTPUT to CSV
    f = open(SAVE_PATH + 'detailed_' + COMPANY_SEARCH + '_python_mined.csv', 'w', newline='')
    w = csv.writer(f)
 
    # Combine both lists into one
    combined_list = [list(itertools.chain(*a)) for a in zip(simple_stores_out, detailed_stores_out)]
 
    for one_store in combined_list:
        try:
            w.writerow(one_store)
        except Exception as err:
            print("Something went wrong: %s" % err)
            w.writerow("Error")
    f.close()
     
     
def runsearch():
    """
    Initialises the searches for each partition produced
    """
    print("%d Keys Remaining" % (len(API_KEY)-1))
    for partition in coord.coordset:
        # Keys have a life-span of 1000 requests
        key_number = (req_count.keynum // 1000)
        req_count.increment_partition()
 
        googleplaces(lat=partition[0],
                     lng=partition[1],
                     radius_metres=RADIUS_KM*1000,
                     search_term=COMPANY_SEARCH,
                     key=API_KEY[key_number])
 
    # OUTPUT to CSV
    f = open(SAVE_PATH + COMPANY_SEARCH + '_python_mined.csv', 'w', newline='')
    w = csv.writer(f)
    for one_store in shops_list:
        w.writerow(one_store)
    f.close()
 
    # OUTPUT LOG to CSV
    f = open(SAVE_PATH + 'log_' + COMPANY_SEARCH + '_python_mined.csv', 'w', newline='')
    w = csv.writer(f)
    for debug_result in debug_list:
        w.writerow(debug_result)
    f.close()
 
    # DETAIL SEARCH
    fillindetails()
 
 
if __name__ == "__main__":
    # 1. CREATE PARTITIONS
    # Setup coordinates
    coord = coordinates_box()
    coord.createcoordinates(51.471834, -0.204326, 51.542672, -0.049488)
    # 2. SEARCH PARTITIONS
    # Setup counter
    req_count = counter()
    runsearch()




















###########################################################################################
###########################################################################################
So we see that the geopy.distance.vincenty calculation takes 866 seconds, geopy.distance.great_circle takes 486 seconds, and our vectorised-numpy haversine formula takes 2.85 seconds (for 30 million distances). Hence, if accuracy is not so important (or we have hundreds of millions of distances to calculate) it may be better to use this over the vincenty formula – however, what about the projected-coordinates? It is not fair to compare the calculations on the 30 million distances to cKDTree.sparse_distance_matrix because although that calculates 30 million distances (in 152 seconds) it considers 10 billion potential distances!

So it may be interesting to use our vectorised function to create a distance matrix. With a slightly re-write we can now input a list of co-ordinates (rather than a list of pairs) and get back a full distance-matrix. However, this is wasteful as we have duplicates so we can modify the function to only return the upper triangle (“broadcasting_based_trima“). This functions calculates 199 million distances in 23.3 seconds.

Can we get even faster? If we project our co-ordinates then we can use the super-fast scipy.spatial.distance.cdist function (using the minkowski distance). It takes 0.45 seconds to project the points and 2.52 seconds to get the full distance matrix -> less than 3 seconds for 200 million distances (and as we saw more the projection-based approach was more accurate than the haversine approximation)!


"""
Part C - Distance Matricies
 
Lets find a way to compare the numpy vectorised code; instead of
using the pairings generated by cKDTree's distance matrix let's create our
own, by slightly revising the code of the vectorised functions:
"""
import numpy as np
 
path_to_csv = '.../huge_points_test.csv'
points = np.genfromtxt(path_to_csv,
                       delimiter=',',
                       skip_header=1,
                       usecols=(0,1),
                       dtype=(float, float))
 
def broadcasting_based_ma(df):
    """
    Cross every co-ordinate with every co-ordinate.
    lat, lon
    """
    data = np.deg2rad(df)
    lat = data[:,0]
    lng = data[:,1]
    diff_lat = lat[:,None] - lat
    diff_lng = lng[:,None] - lng
    d = np.sin(diff_lat/2)**2 + np.cos(lat[:,None])*np.cos(lat) * np.sin(diff_lng/2)**2
    return 2 * 6371 * 1000 * np.arcsin(np.sqrt(d))
 
def broadcasting_based_trima(df):
    """
    Create an upper triangular distance matrix to
    save resources.
    lat, lon
    """
    data = np.deg2rad(df)
    lat = data[:,0]
    lng = data[:,1]
    idx1,idx2 = np.triu_indices(lat.size,1)
    diff_lat = lat[idx2] - lat[idx1]
    diff_lng = lng[idx2] - lng[idx1]
    d = np.sin(diff_lat/2)**2 + np.cos(lat[idx2])*np.cos(lat[idx1]) * np.sin(diff_lng/2)**2
    return 2 * 6371 * 1000 * np.arcsin(np.sqrt(d))    
 
# Try creating a distance matrix with 20,000 points
 
%time out = broadcasting_based_ma(points[:20000])
del out
# Calculations: 400,000,000
# Wall time: 24.6 s
 
%time out_v2 = broadcasting_based_trima(points[:20000])
del out_v2
# Calculations: 199,990,000
# Wall time: 23.3 s
 
"""
E.g with 4 rows the dense matrix - broadcasting_based_ma:
[[      0.          284016.09682169  278297.26191605  359212.49587497]
 [ 284016.09682169       0.          342440.79369869  445836.60288353]
 [ 278297.26191605  342440.79369869       0.          104156.69522161]
 [ 359212.49587497  445836.60288353  104156.69522161       0.        ]]
 
The upper triangle - broadcasting_based_trima:
[ 284016.09682169  278297.26191605  359212.49587497  342440.79369869
  445836.60288353  104156.69522161]
"""
 
"""
Let's try the cdist function (on the projected points)
"""
from pyproj import Proj, transform
from scipy.spatial.distance import cdist
from scipy import sparse
 
def proj_arr(points,proj_to):
    """
    Project geographic co-ordinates to get cartesian x,y
    Transform(origin|destination|lon|lat)
    """
    inproj = Proj(init='epsg:4326')
    outproj = Proj(proj_to)
    func = lambda x: transform(inproj,outproj,x[1],x[0])
    return np.array(list(map(func, points)))
 
uk = '+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 \
+x_0=400000 +y_0=-100000 +ellps=airy \
+towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489 +units=m +no_defs'
 
%time proj_pnts = proj_arr(points, uk)
# Wall time: 454 ms
%time out_v3 = cdist(proj_pnts[:20000],proj_pnts[:20000],p=2)
# Wall time: 2.52 s
max_dist = np.amax(sparse.tril(out_v3, k=-1).data)
print(max_dist)
# 1,201,694 (length of UK is 1,407,000)
del points
del out_v3






############################################################################
#---------------------             --------------------







############################################################################











############################################################################
#---------------------             --------------------




############################################################################





















############################################################################
#---------------------             --------------------






























