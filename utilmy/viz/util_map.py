HELP = """
  pip install folium geopandas

# should be used in a web run env as jupyter notebook to see the actual map. 


# remember, you can't re-render folium map, if you have already plotted map
# you can't add json to it, https://github.com/python-visualization/folium/issues/906 




"""

import json,  requests
from os import link
import os
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
          


# to kevin, runing tests requires these classes to be defined first, thus they are defined before tests





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


def test_create_webmap():
    return create_webmap()

def test_add_geojson_to_map(geojson_map:str='http://enjalot.github.io/wwsd/data/world/world-110m.geojson'):
    map = webMap()
    map.create_map(open_new_tab=False)
    add_geojson_to_webmap(geojson_map,map)

def test_add_topojson_to_map(topojson_path:str="https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json"):
    map = webMap()
    map.create_map(open_new_tab=False)
    add_topojson_to_webmap(topojson_path,map)

    

#################### Main ######################  # 
# remember, you can't re-render folium map, if you have already plotted map
# you can't add json to it, https://github.com/python-visualization/folium/issues/906 

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


def create_webmap(dir:str=''):
    """
        creates a new webmap as html file, if directory is not given, the current working diretory is used.

        Args:
        ----
            dir: str
                the directory to which the file is saved.
    """
    map = webMap()
    map.create_map(dir=dir)
    return map


def add_geojson_to_webmap(geojson_path:str,map):
    """
        adds geojson data to existing map object.

        Args:
        ----
            geojson_path: str
                url to geojson data
            map: webMap
                the map object to which the data is added.
    """
    map.add_geojson(geojson_path) 

def add_topojson_to_webmap(topojson_path,map):
    """
        adds geojson data to existing map object.

        Args:
        ----
            topojson_path: str
                url to topojson data
            map: webMap 
                the map object to which the data is added
    """
    map.add_topojson(topojson_path)

def plot_choropleth_webmap():
    pass

def plot_heatWebMap():
    pass

class HTMLDoc(object):
    def __init__(self,title:str='index.html') -> None:
        super().__init__()
        self.out_url=""
        self.doc = "<!DOCTYPE html> \n"
        self.title = title
        self.head = f"<html> \n<head>\n"
        self.body = "\n</head> \n<body>"
        self.tail = "\n</body> \n</html>"
        self.added_tags=""
        self.added_js=''

        # leaflet links
        links = """ <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"
            integrity="sha512-xodZBNTC5n17Xt2atTPuE1HxjVMSvLVW9ocqUKLsCC5CXdbqCmblAshOMAS6/keqq/sMZMZ19scR4PsZChSR7A=="
            crossorigin=""/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"
            integrity="sha512-XQoYMqMTK8LvdxXYG3nZ448hOEQiglfqkJs1NOQV44cWnUrBc8PkAOcXy20w0vlaXaVUearIOBhiXZ5V3ynxwA=="
            crossorigin=""></script>
            <script src="https://unpkg.com/topojson@3"></script>
            """

        self.head+=f"<title>{self.title}</title>\n"+links

    def get_document(self):
        """
            get the whole html documents reconstructed from it's parts
        """
        doc = self.doc + self.head + self.body + self.tail
        return doc

    def print_document(self):
        """
            print the html document in the terminal
        """
        doc = self.doc + self.head + self.body + self.tail
        print(doc)


    def tag(self,tag_name:str,content:str='',styles:dict={},css_class:str='',css_id=''): 
        """
            adds a tag with a tag_name to the <body> of the html document
            styles added as a dictionary, e.g == {'color':'blue'}

            Args:
            ----
            tag_name: str
                valid name of HTML tag
            content: str
                content of the html tag, if <div> tag is used, pass inner html elements as a string
            styels: dict
                dictionary holding css styles, pass the same valid css attribute names and values 
        """

        css = self.get_styles(styles)
        self.body+= '\n'+f"<{tag_name} style='{css}' class='{css_class}' id='{css_id}'>{content}</{tag_name}>"
        if "<script>" not in content: self.added_tags+= '\n'+f"<{tag_name} style='{css}' class='{css_class}' id='{css_id}'>{content}</{tag_name}>"


    def br(self,styles:dict={}): 
        css = self.get_styles(styles)
        self.body+= '\n'+f"<br style='{css}'/>"

    def hr(self,styles:dict={}): 
        css = self.get_styles(styles)
        self.body+= '\n'+f"<hr style='{css}'/>"
    
    def get_styles(self,styles):
        """
            loops styles dict and returns a css string 

            Args
            ----
            styles: dict
                css styles for the html element 
        """

        css = ''
        for item in styles.items():
            css+=f'{item[0]}:{item[1]};'
        return css

    def add_css(self,styles:dict):
        """
            add css styles to the html document

            Args:
            ----
                styles:dict
                    a dictionary holding styles to be added, styles has the same attributes and values as normal css file
        """
        styles = self.get_styles(styles)
        css = f"\n<style>{styles}</style>"
        self.head+=css
    
    def add_js(self,js_script):
        """
            add javascript tag with it's code to the html document

            Args:
            ----
                js_script: str
                    the javascript code to be added to the html document.
        """
        self.added_js = f"\n<script>\n{js_script}</script>"
        self.body+=self.added_js

    def clear_body(self):
        """
            removes all tags inside the html document body tag.
        """
        self.body=''

    def rewrite_js(self,js_script):
        """
            remove all the existing javascript tags along with code and adds a new one. 

            Args:
            ----
                js_script: str
                    new javascrpit code to be added to a script tag in the body tag of the html.
        """
        self.clear_body()
        self.added_js = f"\n<script>\n{js_script}</script>"
        self.body+=self.added_tags+self.added_js

    def save(self,dir:str=None):
        # note to Kevin: this is a simpler implementation, when a directory isn't given just print the file to the current working direcotry.
        """
            saves the html file to a specific directory, default file name is index.html, default directory is current working directory  

            Args:
            ----
            file_name: str
                name of the html document
            dir: str
                directory to which the file will be saved.
        """

        dir = dir.replace('\\','/')+"/" if dir is not None and dir.startswith('/') else os.getcwd()+'/'
        dir = dir.replace('\\','/') # in case the cwd has a \\
        path = dir+self.title
        self.out_url = path
        htmlDoc = self.get_document()
        with open(path,'w') as f:
            f.write(htmlDoc)

    def browse(self):
        import webbrowser as wb  # this is a built-in module
        browsed_file = f"file:///{self.out_url}"
        wb.open(browsed_file, new=0) # cannot control fully the behavior of the browser, see:https://stackoverflow.com/questions/1997327/python-webbrowser-open-setting-new-0-to-open-in-the-same-browser-window-does

class webMap():
    def __init__(self,title:str='map.html') -> None:
        self.title = title
        self.htmlDoc = HTMLDoc(title=title) 
        self.out_dir = ''
        self.map_js = ''

    def load_document(self,rewrite=False,open_new_tab=True):
        self.htmlDoc.add_js(self.map_js) if rewrite==False else self.htmlDoc.rewrite_js(self.map_js)
        self.htmlDoc.save(self.out_dir)
        if open_new_tab:self.htmlDoc.browse()

    
    def create_map(self,center:list=[18,32],zoom:int=3,tile:str='',dir:str='',open_new_tab=True):
        """
            creates a new web map stored in html file with all it's configurations

            Args:
            ----
                center: list
                    the center of the map, defaults: [18,32], lat lng EPSG:,
                zoom: int
                    initial zoom level of the map, default 3.
                tile: str
                    the tile of the map, if empty it defaults to "https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"
        """

        
        self.htmlDoc.tag('div',styles={'width':'100vw','height':'100vh'},css_id='map')
        tile = tile if tile != "" else "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        
        js = """let map = L.map('map').setView({center},{zoom});
                    var OpenStreetMap_Mapnik = L.tileLayer('{tile}', {{
                    maxZoom: 19,
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    }});
                    OpenStreetMap_Mapnik.addTo(map);
                """.format(center=center,zoom=zoom,tile=tile)
        
        self.out_dir = dir
        self.map_js = js
        self.load_document(open_new_tab=open_new_tab)

    def add_geojson(self,geojson_path):
        """
            adds a geojson to an existing map object, and modify it's html code.
            
            Args:
            ----
                geojson_path: str
                    geojson_path to add to the map 
        """
        geojson_js = """
                        let grab_geojson = async function(){{
                        let geo_path = '{geojson_path}'
                        let geojson_promise = await fetch(geo_path).then(res=>res.json()).then(data=>json_data=data)
                        return geojson_promise

                        }}
                        let geojson_loader = async function(){{
                        let geojson = await grab_geojson()
                        L.geoJSON(json_data).addTo(map);	
                        }}
                        geojson_loader()
        
        """.format(geojson_path=geojson_path)
        self.map_js+=geojson_js        
        self.load_document(rewrite=True)
        
    def add_topojson(self,topojson_path):
        """
            adds a topojson to an existing map object, and modify it's html code.
            
            Args:
            ----
            geojson_path: str
                geojson_path to add to the map 
        """
        
        topojson_js = """
        L.TopoJSON = L.GeoJSON.extend({{
                addData: function (data) {{
                    var geojson, key;
                    if (data.type === "Topology") {{
                        for (key in data.objects) {{
                            if (data.objects.hasOwnProperty(key)) {{
                                geojson = topojson.feature(data, data.objects[key]);
                                L.GeoJSON.prototype.addData.call(this, geojson);
                            }}
                        }}
                    return this;
                }}
                L.GeoJSON.prototype.addData.call(this, data);
                return this;
                }}
            }});
            L.topoJson = function (data, options) {{
            return new L.TopoJSON(data, options);
            }};
            
            var geojson = L.topoJson(null, {{
                style: function(feature){{
                return {{
                    color: "#000",
                    opacity: 1,
                    weight: 1,
                    fillColor: '#35495d',
                    fillOpacity: 0.8
                }}
            }},
            onEachFeature: function(feature, layer) {{
            layer.bindPopup('<p>'+feature.properties.name+'</p>')
            }}
            }}).addTo(map);
        
      async function getGeoData(url) {{
        let response = await fetch(url);
        let data = await response.json();
        console.log(data)
        return data;
      }}
      
      //fetch the geojson and add it to our geojson layer
      getGeoData('{topojson_path}').then(data => geojson.addData(data));
    """.format(topojson_path=topojson_path)
        self.map_js+=topojson_js        
        self.load_document(rewrite=True)
        
# ###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()





# todo 
#   adding web choropleth_map and heat map
#   remove folium and it's dependencies
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
