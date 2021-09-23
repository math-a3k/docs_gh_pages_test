from utilmy import pd_read_file
try:
    import folium
except:
    print('pip install folium')

HELP = """pip install folium"""

def help():
    ss = ""
    ss += HELP
    print(ss)
          
    
#################### Main ######################    

def plot_map(center:list=[18,32],zoom:int=2)->folium.Map:

    #folium will catch value errors for invalid numbers, no try block needed
    m = folium.Map(location=center, zoom_start=zoom)
    folium.LayerControl().add_to(m)
    return m
    

##################### Tests #####################    

def test_plot_map()->None:
    # folium will catch value errors for invalid numbers, no try block needed
    assert type(plot_map([35,139])) == folium.Map


# todo 
#   adding plot_choropleth_map
#   adding plot_geocoded_adderess 


