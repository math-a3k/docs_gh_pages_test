default = """
              body{margin:25px;font-family: 'Open Sans', sans-serif;}
              h1,h2,h3,h4,h5,h6{margin-bottom: 0.5rem;font-family: 'Arvo', serif;line-height: 1.5;color: #32325d;}
              .dataTables_wrapper{overflow-x: auto;}
              hr{border-top: dotted 4px rgba(26, 47, 51, 0.7);opacity:0.3 ;}
              div{margin-top: 5px;margin-bottom: 5px;}
              table {border-collapse: collapse;}
              table th,table td {border: 1px solid lightgrey;}
              #mynetwork{float: none !important;}
              #config{float: none !important;height: auto !important;}              
"""

a4 = default + """
            body {background: rgb(204,204,204); }
            page {
              background: white;display: block;padding:15px;margin: 0 auto;margin-bottom: 0.5cm;
              box-shadow: 0 0 0.5cm rgba(0,0,0,0.5);
            }
            page[size="A4"] {width: 21cm; }
            @media print {body, page {margin: 0;box-shadow: 0;}}
"""

border = default + """
            .highcharts-container {border: 3px dotted grey;}
            .mpld3-figure {border: 3px dotted grey;}
"""

a3d = default + """
            div {
            background: white;display: block;margin: 0 auto;
            margin-bottom: 0.5cm;box-shadow: 0 0 0.5cm rgba(0,0,0,0.5);}
            h1,h2,h3,h4,h5,h6 {box-shadow: 0 0 0.5cm rgba(0,0,0,0.5);
            padding: 5px;} 
"""



grey = """
        .body {
          font: 90%/1.45em "Helvetica Neue", HelveticaNeue, Verdana, Arial, Helvetica, sans-serif;
          margin: 0;
          padding: 0;
          color: #333;
          background-color: #fff;
        }
"""
def fontsize_css(size):
    """function fontsize_css
    Args:
        size:   
    Returns:
        
    """
    css_8px = base + """
            body,table th,table td,text,
            .highcharts-title,.highcharts-axis-title,
            #mynetwork,#config{
               font-size:"""+size+"""px !important;
            }
    """
    return css_8px

def getcss(css_name):
    if css_name == "grey":
        return grey
    elif css_name == "a3d":
        return a3d
    elif css_name == "default":
        return default
    elif css_name == "border":
        return border
    elif css_name == "a4":
        return a4
    elif css_name.split("_")[0] == "css" and len(css_name.split("_")) == 2 and css_name.split("_")[1][:-2].isdigit():
        return fontsize_css(css_name.split("_")[1][:-2])
    else:
        return ""
