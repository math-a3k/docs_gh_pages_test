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
    

def plot_choropleth_map(center:list, zoom:int, geojson:str, csv:str, geojson_mutual_column:str, csv_mutual_column:str, choropleth_data_column:str, legend_title:str='Legend' ,color:str='PuBu'):
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
    geo_data=geojson,
    name="choropleth",
    data=data_read,
    columns=[f"{csv_mutual_column}", f"{choropleth_data_column}"],
    key_on=f"feature.{geojson_mutual_column}",
    fill_color=f"{color}",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f"{legend_title}",
)
    c.add_to(m)
    return m



##################### Tests #####################    

def test_plot_map()->None:
    # folium will catch value errors for invalid numbers, no try block needed
    assert type(plot_map([35,139])) == folium.Map


# todo 
#   adding tests and docs to plot_choropleth_map
#   adding plot_geocoded_adderess 


