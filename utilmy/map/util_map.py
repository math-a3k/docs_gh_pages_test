import folium
from utilmy import pd_read_file

def plot_map(center:list=[18,32],zoom:int=2)->folium.folium.Map:
    #folium will catch value errors for invalid numbers, no try block needed
    m = folium.Map(location=center, zoom_start=zoom)
    folium.LayerControl().add_to(m)
    return m
    
# todo 
#   adding plot_choropleth_map
#   adding plot_geocoded_adderess 