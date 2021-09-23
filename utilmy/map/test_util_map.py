import folium
from .util_map import plot_map

def test_plot_map()->None:
    # folium will catch value errors for invalid numbers, no try block needed
    assert type(plot_map([35,139])) == folium.folium.Map


