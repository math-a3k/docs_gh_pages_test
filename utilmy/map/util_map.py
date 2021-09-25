HELP = """
  pip install folium geopandas

# should be used in a web run env as jupyter notebook to see the actual map. 


# remember, you can't re-render folium map, if you have already plotted map
# you can't add json to it, https://github.com/python-visualization/folium/issues/906 




"""

import json,  requests
from utilmy import pd_read_file

try:
    import folium
    import geopandas as gdp
except:
    print('''pip install folium geopandas
        ''')


def log(*s):
    print(*s, flush=True)

def help():
    ss = ""
    ss += HELP
    print(ss)
          


################################################################################
##################### Use cases ################################################
# plot_map
def test_plot_map_use_case():
    plot_map([15,34],12)


# plot_geojson
def test_plot_geojson_use_case():
    url = 'http://enjalot.github.io/wwsd/data/world/world-110m.geojson'
    plot_geojson(url)


# plot_topojson
def test_plot_topojson_use_case():
    url = 'https://unpkg.com/world-atlas@1.1.4/world/110m.json'
    plot_topojson(url,'countries')

# plot_choropleth_map
def test_plot_choropleth_map_use_case():
    csv=  'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/US_Unemployment_Oct2012.csv'
    geojson = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json'
    plot_choropleth_map([48, -102],3,geojson,csv,'id','State','Unemployment')




#################### Main ########################################################
def plot_map(center:list=[18,32],zoom:int=2)->folium.Map:

    #folium will catch value errors for invalid numbers, no try block needed
    m = folium.Map(location=center, zoom_start=zoom)
    folium.LayerControl().add_to(m)
    return m
    

def plot_geojson(geojson_path:str,center:list=[18,32],zoom:int=2):
    m = folium.Map(location=center,zoom_start=zoom)
    folium.GeoJson(geojson_path, name="geojson").add_to(m)
    return m 


def plot_topojson(topojson_path:str,json_object_name:str,center:list=[18,32],zoom:int=2):
    m = folium.Map(location=center,zoom_start=zoom)
    folium.TopoJson(
    json.loads(requests.get(topojson_path).text),
    f"objects.{json_object_name}",
    name="topojson",
    ).add_to(m)
    return m
    
# not completed yet
# def plot_shp(shapefile_path,center:list=[18,32],zoom:int=2):
#     shp = gpd.read_file(shapefile_path)
#     geojson = shp.to_file("file.json", driver="GeoJSON")
#     plot_geojson(geojson,center,zoom)


def plot_choropleth_map(center:list, zoom:int, geojson:str, csv:str, geojson_mutual_column:str, csv_mutual_column:str, 
                        choropleth_data_column:str, legend_title:str='Legend' ,color:str='PuBu'):
    if geojson=='' :
        print('empty path for geojson data')
        return
    elif csv=='':
        print('empty path for geojson data') 
        return
    #m = plot_map(center,zoom)
    data_read = pd_read_file(csv)
    m = folium.Map(location=center, zoom_start=zoom)
    c = folium.Choropleth(
                geo_data     = geojson,
                name         = "choropleth",
                data         = data_read,
                columns      = [f"{csv_mutual_column}", f"{choropleth_data_column}"],
                key_on       = f"feature.{geojson_mutual_column}",
                fill_color   = f"{color}",
                fill_opacity = 0.7,
                line_opacity = 0.2,
                legend_name  = f"{legend_title}",
    )
    c.add_to(m)
    return m



###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
    # test2()






# todo 
#   adding tests and docs to plot_choropleth_map
#   adding Vector Data (shp)
#   adding raster data. 
#   adding plot_geocoded_adderess 









##################### Tests #####################    

def ztestz_plot_map()->None:
    # folium will catch value errors for invalid numbers, no try block needed
    assert type(plot_map([35,139])) == folium.Map

def ztestz_plot_geojson()->None:
    assert type(plot_geojson) == folium.Map

def ztestz_plot_topojson()->None:
    assert type(plot_topojson) == folium.Map

def ztestz_plot_choropleth_map()->None:
    assert type(plot_choropleth_map) == folium.Map
