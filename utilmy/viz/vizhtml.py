""" Converter python ---> HTML
https://try2explore.com/questions/10109123
https://mpld3.github.io/examples/index.html
https://notebook.community/johnnycakes79/pyops/dashboard/pandas-highcharts-examples


https://datatables.net/


"""
import random
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
from tqdm import tqdm

import fire

from box import Box

# Converting python --> HTML
import matplotlib.pyplot as plt
import mpld3


##################################################################################################################
def log(*s):
    print(*s, flush=True)


###################################################################################
#### Example usage ################################################################
def test_getdata():
    """
    data = test_get_data()
    df   = data['housing.csv']
    df.head(3)

    https://github.com/szrlee/Stock-Time-Series-Analysis/tree/master/data
    """
    import pandas as pd
    flist = [
       'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',

       'https://raw.githubusercontent.com/szrlee/Stock-Time-Series-Analysis/master/data/AAPL_2006-01-01_to_2018-01-01.csv',

       'https://github.com/subhadipml/California-Housing-Price-Prediction/raw/master/housing.csv',


    ]

    data = {}
    for url in flist :
       fname =  url.split("/")[-1]
       print( "\n", "\n", url, )
       df = pd.read_csv(url)
       data[fname] = df
       print(df)
       # df.to_csv(fname , index=False)
    print(data.keys() )
    return data






def test_usage():
    # pip install box-python    can use .key or ["mykey"]  for dict
    cfg = Box({})
    cfg.tseries = {"title": 'ok'}
    cfg.scatter = {"title": 'ok'}
    cfg.histo = {"title": 'ok'}
    cfg.use_datatable = True

    df = pd.DataFrame([[1, 2]])
    df2_list = [df, df, df]

    doc = htmlDoc(dir_out="", title="hello", format='myxxxx', cfg=cfg)

    doc.h1('My title')  # h1
    doc.sep()
    doc.br()  # <br>


    # list_info = []
    # for i in range(100):
    #     info = {}
    #     info ['x'] = i
    #     info ['y'] = i
    #     info ['label'] = i
    #     info ['class1'] = i
    #     info ['class2'] = i
    #     info ['class2_size'] = i
    #     info ['class1_color'] = i
    #     list_info.append(info)
    # df = pd.DataFrame.from_records(list_info)

    # doc.tag('<h2> My graph title </h2>')
    # doc.plot_scatter(df, cfg.scatter, mode='mpld3', save_img=False)
    # doc.hr()  # doc.sep() line separator



    # for df2_i in df2_list:
    #     print(df2_i)
    #     # doc.h3(f" plot title: {df2_i['category'].values[0]}")
    #     doc.plot_tseries(df2_i, cfg.tseries, mode='mpld3', save_img="")

    # print(doc.get_html())


    # df2 = pd.DataFrame({
    #     'col1': [1.5, 0.5, 1.2, 0.9, 3],
    #     'col2': [0.7, 0.2, 0.15, 0.2, 1.1]
    #     })

    # doc.tag('<h2> My histo title </h2>')
    # print(df2)
    # # doc.plot_histogram(df2['col1', 'col2'], cfg.histo, mode='mpld3', save_img="")
    # doc.plot_histogram(df2, cfg.histo, mode='mpld3', save_img="")

    # print(doc.get_html())



    # test create table
    list_info = []
    for i in range(1000):
        info = {}
        info ['x'] = i
        info ['y'] = i
        info ['label'] = i
        info ['class1'] = i
        info ['class2'] = i
        info ['class2_size'] = i
        info ['class1_color'] = i
        list_info.append(info)
    df = pd.DataFrame.from_records(list_info)
    doc.table(df, use_datatable=cfg.use_datatable, table_id="test")


    doc.tag("""<p>    My mutilines whatever I want to write
      ok</p>
    """)
    print(doc.get_html())

    doc.save(dir_out="myfile.html")
    doc.open_browser()  # Open myfile.html


#####################################################################################
#### Class ##########################################################################
class htmlDoc(object):
    def __init__(self, dir_out=None, mode="", title="", format: str = None, cfg: dict = None):
        """
           Generate HTML page to display graph/Table.
           Combine pages together.

        """
        self.cc = Box(cfg)  # Config dict
        self.dir_out = dir_out

        self.cc.use_datatable = self.cc.get('use_datatable', False)  # Default val

        self.head = "<html>\n    <head>"
        self.html = "<body>"

        if self.cc.use_datatable:
            self.head = self.head + """
            <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/v/dt/dt-1.10.25/datatables.min.css"/>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
            <script type="text/javascript" src="https://cdn.datatables.net/v/dt/dt-1.10.25/datatables.min.js"></script>"""
            # https://datatables.net/manual/installation
            # add $(document).ready( function () {    $('#table_id').DataTable(); } );

    def get_html(self):
        return self.html

    def tag(self, x):  self.html += "\n" + x
    def h1(self, x)  : self.html += "\n" + f"<h1>{x}</h1>"
    def h2(self, x)  : self.html += "\n" + f"<h2>{x}</h2>"
    def h3(self, x)  : self.html += "\n" + f"<h3>{x}</h3>"
    def h4(self, x)  : self.html += "\n" + f"<h4>{x}</h4>"
    def p(self, x)   : self.html += "\n" + f"<p>{x}</p>"
    def hr(self)     : self.html += "\n" + f"</hr>"
    def sep(self)    : self.html += "\n" + f"</hr>"
    def br(self)     : self.html += "\n" + f"</br>"


    def hidden(self, x):
        # Hidden paragraph
        self.html += "\n" + f"<div id='hidden_section_id'>{x}</div>"
        self.head += "\n" + js_code.js_hidden  # Hidden  javascript


    def save(self, dir_out=None):
        self.dir_out = dir_out if dir_out is not None else self.dir_out
        full = self.head + "\n    </head>\n" + self.html + "\n    </body>\n</html>"
        with open(self.dir_out, mode='w') as fp:
            fp.write(full)


    def open_browser(self):
        if os.name == 'nt':
            os.system(f'start chrome "{self.dir_out}" ')


    def table(self, df, format='blue_light', use_datatable=False, table_id=None, **kw):
        """
        ## show table in HTML : https://pypi.org/project/pretty-html-table/
        """
        import pretty_html_table
        html_code = pretty_html_table.build_table(df, format)

        table_id = random.randint(9999,999999) if table_id is None else table_id  #### Unique ID
        if use_datatable:
            html_code = html_code.replace('<table', f'<table id="{table_id}"')
            html_code += """\n<script>$(document).ready( function () {    $('#{mytable_id}').DataTable(); } );</script>\n""".replace(
                '{mytable_id}', str(table_id))
        self.html += "\n\n" + html_code


    def plot_tseries(self, df, coldate, cols_axe1, cols_axe2=None,  cfg: dict = None, mode='mpld3', save_img="",  **kw):
        html_code = ''
        if mode == 'mpld3':
            fig       = pd_plot_tseries_matplot(df)
            html_code = mpld3.fig_to_html(fig)

        elif mode == 'highcharts':
            html_code = pd_plot_tseries_highcharts(df, coldate, cols_axe1=cols_axe1, cols_axe2=cols_axe2, cfg=cfg)

        self.html += "\n\n" + html_code


    def plot_histogram(self, df,  cfg: dict = None, mode='mpld3', save_img="",  **kw):
        html_code = ''
        if mode == 'mpld3':
            fig       = pd_plot_histogram_matplot(df)
            html_code = mpld3.fig_to_html(fig)
        self.html += "\n\n" + html_code


    def plot_scatter(self, df,  cfg: dict = None, mode='mpld3', save_img=False,  **kw):
        html_code = ''
        if mode == 'mpld3':
            html_code = pd_plot_scatter_matplot(df,  cfg, mode, save_img,)
        self.html += "\n\n" + html_code








##################################################################################################################
######### MLPD3 Display ##########################################################################################
mpld3_CSS = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
"""


class mpld3_TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();
      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);
      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


def mlpd3_add_tooltip(fig, points, labels):
    # set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(
        points[0], labels, voffset=10, hoffset=10, css=mpld3_CSS)
    # connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, mpld3_TopToolbar())



def pd_plot_scatter_get_data(df0,colx=None, coly=None, collabel=None,
                            colclass1=None, colclass2=None, nmax=20000):
    import copy
    nmax = min(nmax, len(df0))
    df   = df0.sample(nmax)

    colx      = 'x'      if colx is None else colx
    coly      = 'y'      if coly is None else coly
    collabel  = 'label'  if collabel is None else collabel    ### label per point
    colclass1 = 'class1' if colclass1 is None else colclass1  ### Color per point class1
    colclass2 = 'class2' if colclass2 is None else colclass2  ### Size per point class2

    #######################################################################################
    for ci in [ collabel, colclass1, colclass2 ] :
       if ci  not in df.columns : df[ci]  = ''
       df[ci]  = df[ci].fillna('')

    #######################################################################################
    xx = df[colx].values
    yy = df[coly].values

    label_list = df[collabel].values

    ### Using Class 1 ---> Color
    color_scheme = [ 0,1,2,3]
    n_colors     = len(color_scheme)
    color_list   = [  color_scheme[ hash(str( x)) % n_colors ] for x in df[colclass1].values     ]


    ### Using Class 2  ---> Color
    n_size      = len(df['class2'].unique())
    smin, smax  = 1.0, 15.0
    size_scheme = np.arange(smin, smax, (smax-smin)/n_size)
    n_colors    = len(size_scheme)
    size_list   = [  size_scheme[ hash(str( x)) % n_colors ] for x in df[colclass2].values     ]


    ###
    ptype_list = []

    return xx, yy, label_list, color_list, size_list, ptype_list




def pd_plot_scatter_matplot(df, colx=None, coly=None, collabel=None,
                            colclass1=None, colclass2=None, cfg: dict = None, mode='d3', save_path='',  **kw):
    """
    """
    cc           = Box(cfg)
    cc.figsize   = cc.get('figsize', (25, 15))  # Dict type default values
    cc.title     = cc.get('title', 'scatter title' )

    #######################################################################################
    xx, yy, label_list, color_list, size_list, ptype_list = pd_plot_scatter_get_data(df,colx, coly, collabel,
                                                            colclass1, colclass2)

    # set up plot
    fig, ax = plt.subplots(figsize= cc.figsize)  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to returnthe appropriate color/label
    ax.plot(xx, yy, marker='o', linestyle='', ms= size_list, label=label_list,
            color=color_list,
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',  # changes apply to the x-axis
                   which='both',  # both major and minor ticks are affected
                   bottom='off',  # ticks along the bottom edge are off
                   top='off',  # ticks along the top edge are off
                   labelbottom='off')
    ax.tick_params(axis='y',  # changes apply to the y-axis
                   which='both',  # both major and minor ticks are affected
                   left='off',  # ticks along the bottom edge are off
                   top='off',  # ticks along the top edge are off
                   labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label
    for i in range(len(df)):
        ax.text(xx[i], yy[i], label_list[i], size=8)


    if len(save_path) > 1 :
        plt.savefig(f'{cc.save_path}-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.png', dpi=200)

    # Plot
    fig, ax = plt.subplots(figsize=cc.figsize)  # set plot size
    ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    #for name, group in groups_clusters:
    points = ax.plot(xx, yy, marker='o', linestyle='',
                     ms    = size_list,
                     label = label_list, mec='none',
                     color = color_list)
    ax.set_aspect('auto')

    # set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    mlpd3_add_tooltip(fig, points, label_list)
    # set tooltip using points, labels and the already defined 'css'
    # tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=mpld3_CSS)
    # connect tooltip to fig
    # mpld3.plugins.connect(fig, tooltip, mpld3_TopToolbar())

    ax.legend(numpoints=1)  # show legend with only one dot

    return fig
    ##### Export ############################################################
    #mpld3.fig_to_html(fig, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False, **kwargs)[source]
    ## html_code = mpld3.fig_to_html(fig,  **kw)
    ## return html_code



def pd_plot_histogram_matplot(dfi, path_save=None, nbin=20.0, q5=0.005, q95=0.995, nsample=-1, show=False, clear=True):
    """
       fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(df[config['x']].values,
            bins=config['bins'], color='red', alpha=0.5)
    ax.set_xlabel(config['x'])
    ax.set_ylabel(config['y'])
    ax.set_title(config['title'])
    ax.set_xlim(config['xlim'])
    ax.set_ylim(config['ylim'])
    return fig

    """
    q0 = dfi.quantile(q5)
    q1 = dfi.quantile(q95)

    fig = plt.figure()

    if nsample < 0:
        dfi.hist(bins=2)
        # dfi.hist(bins=np.arange(q0, q1,  (q1 - q0) / nbin))
    else:
        dfi.sample(n=nsample, replace=True).hist(
            bins=np.arange(q0, q1,  (q1 - q0) / nbin))
    plt.title(path_save.split("/")[-1] if path_save else 'None')

    if path_save is not None:
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        plt.savefig(path_save)
        print(path_save)

    # plt.close(fig)
    return fig


def pd_plot_tseries_matplot(df, plot_type=None, cols_axe1: list = [], cols_axe2: list = [],
                            figsize=(8, 4), spacing=0.1, **kw):
    """


    """
    from pandas import plotting
    from pandas.plotting import _matplotlib
    from matplotlib import pyplot as plt

    plt.figure(figsize=figsize)
    # Get default color style from pandas - can be changed to any other color list
    if cols_axe1 is None:
        cols_axe1 = df.columns
    if len(cols_axe1) == 0:
        return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(
        num_colors=len(cols_axe1 + cols_axe2))

    # Displays subplot's pair in case of plot_type defined as `pair`
    if plot_type == 'pair':
        ax = df.plot(subplots=True, figsize=figsize, **kw)
        # plt.show()
        html_code = mpld3.fig_to_html(ax,  **kw)
        return html_code

    # First axis
    ax = df.loc[:, cols_axe1[0]].plot(
        label=cols_axe1[0], color=colors[0], **kw)
    ax.set_ylabel(ylabel=cols_axe1[0])
    ##  lines, labels = ax.get_legend_handles_labels()
    lines, labels = [], []

    i1 = len(cols_axe1)
    for n in range(1, len(cols_axe1)):
        df.loc[:, cols_axe1[n]].plot(
            ax=ax, label=cols_axe1[n], color=colors[(n) % len(colors)], **kw)
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label

    for n in range(0, len(cols_axe2)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        df.loc[:, cols_axe2[n]].plot(
            ax=ax_new, label=cols_axe2[n], color=colors[(i1 + n) % len(colors)], **kw)
        ax_new.set_ylabel(ylabel=cols_axe2[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    #plt.show()
    return ax
    # html_code = mpld3.fig_to_html(ax,  **kw)
    # return html_code



def mpld3_server_start():
    # Windows specifc
    # if os.name == 'nt': os.system(f'start chrome "{dir_out}/embeds.html" ')
    # mpld3.show(fig=None, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None, **kwargs)[source]
    mpld3.show()  # show the plot














############################################################################################################################
############################################################################################################################
def pd_plot_highcharts(df):
    """
    # Basic line plot
   chart = serialize(df, render_to="my-chart", title="My Chart")
   # Basic column plot
   chart = serialize(df, render_to="my-chart", title="Test", kind="bar")
   # Basic column plot
   chart = serialize(df, render_to="my-chart", title="Test", kind="barh")
   # Plot C on secondary axis
   chart = serialize(df, render_to="my-chart", title="Test", secondary_y = ["C"])
   # Plot on a 1000x700 div
   chart = serialize(df, render_to="my-chart", title="Test", figsize = (1000, 700))


    """
    import pandas_highcharts
    data = pandas_highcharts.serialize(
        df, render_to='my-chart', output_type='json')
    json_data_2 = "new Highcharts.StockChart(%s);" % pandas_highcharts.core.json_encode(
        data)

    html_code = """<div id="{chart_id}"</div>
      <script type="text/javascript">{data}</script>""".format(chart_id="new_brownian", data=json_data_2)
    return html_code



def pd_plot_scatter_highcharts(df0:pd.DataFrame, colx:str=None, coly:str=None, collabel=None,
                               colclass1=None, colclass2=None, colclass3=None, nmax=10000,
                               cfg:dict={}, mode='d3', save_img=False,  verbose=True,  **kw ):
    """ Plot Highcharts X=Y Scatter

    """
    import matplotlib
    from box import Box
    from highcharts import Highchart

    cc = Box(cfg)
    cc.title      = cc.get('title',    'my scatter')
    cc.figsize    = cc.get('figsize', (640, 480) )   ### Dict type default values
    cc.colormap   = cc.get('colormap', 'brg')
    if verbose: print(cc['title'], cc['figsize'])

    nmax = min(nmax, len(df0))
    df   = df0.sample(nmax)

    colx      = 'x'      if colx is None else colx
    coly      = 'y'      if coly is None else coly
    collabel  = 'label'  if collabel is None else collabel    ### label per point
    colclass1 = 'class1' if colclass1 is None else colclass1  ### Color per point class1
    colclass2 = 'class2' if colclass2 is None else colclass2  ### Size per point class2
    colclass3 = 'class3' if colclass3 is None else colclass3  ### Marker per point


    #######################################################################################
    for ci in [ collabel, colclass1, colclass2 ] :
       if ci  not in df.columns : df[ci]  = ''  ### add missing
       df[ci]  = df[ci].fillna('')

    xx         = df[colx].values
    yy         = df[coly].values
    label_list = df[collabel].values

    ### Using Class 1 ---> Color
    color_list   = [ hash(str(x)) for x in df[colclass1].values     ]
    # Normalize the classes value over [0.0, 1.0]
    norm          = matplotlib.colors.Normalize(vmin=min(color_list), vmax=max(color_list))
    c_map         = plt.cm.get_cmap(cc.colormap)
    color_list   = [  matplotlib.colors.rgb2hex(c_map(norm(x))).upper() for x in color_list    ]


    ### Using Class 2  ---> Color
    n_size      = len(df[colclass2].unique())
    smin, smax  = 1.0, 15.0
    size_scheme = np.arange(smin, smax, (smax-smin)/n_size)
    n_colors    = len(size_scheme)
    size_list   = [  size_scheme[ hash(str( x)) % n_colors ] for x in df[colclass2].values     ]


    # Create chart object
    chart = Highchart()
    options = { 'chart': {
            'width': cc.figsize[0],
            'height': cc.figsize[1]
        },   'title': {
        'text': cc.title
    },
    'xAxis': {
        'title': {
            'text': colx
        }
    },
    'yAxis': {
        'title': {
            'text': coly
        }
    },
    'legend': {
        'enabled': False
    },'tooltip': {
        'pointFormat': '{point.label}'
    }}

    chart.set_dict_options(options)

    # Plot each cluster with the correct size and color
    data = [{
        'x' : float(xx[i]),
        'y' : float(yy[i]),
        "label" : str(label_list[i]),
        "marker": { 'radius' : int(size_list[i]) },
        'color' : color_list[i]
        } for i in range(len(df))
    ]

    chart.add_data_set(data, 'scatter')
    chart.buildcontent()
    html_code = chart._htmlcontent.decode('utf-8')
    return html_code





def pd_plot_tseries_highcharts(df,
                              coldate:str=None,
                              cols_axe1=[],
                              cols_axe2=[],
                              figsize=None,

                              title=None,
                              x_label=None,
                              axe1_label=None,
                              axe2_label=None,
                              cfg:dict={}, mode='d3', save_img=False):

    from highcharts import Highchart
    from box import Box

    cc = Box(cfg)
    cc.coldate      = 'date'  if coldate is None else coldate
    cc.x_label      = coldate if x_label is None else x_label
    cc.axe1_label   = "_".join(cols_axe1)      if axe1_label is None else axe1_label
    cc.axe2_label   = "_".join(cols_axe2)      if axe2_label is None else axe2_label
    cc.title        = cc.get('title',    axe1_label + " vs " + coldate ) if title is None else title
    cc.figsize      = cc.get('figsize', (25, 15) )    if figsize is None else figsize
    cc.subtitle     = cc.get('subtitle', '')


    #########################################################
    H = Highchart()
    options = {
      'chart':   { 'zoomType': 'xy'},
        'title': { 'text': cc.title},
        'subtitle': {  'text': cc.subtitle },
        'xAxis': [{
                      'type': 'datetime',
                      'title': { 'text': cc.x_label }
                  }],
        'yAxis': [{
            'labels': {
                'style': {  'color': 'Highcharts.getOptions().colors[2]' }
            },
            'title': {
                'text': cc.axe2_label,
                'style': {   'color': 'Highcharts.getOptions().colors[2]' }
            },
            'opposite': True

        }, {
            'gridLineWidth': 0,
            'title': {
                'text': cc.axe1_label,
                'style': {
                    'color': 'Highcharts.getOptions().colors[0]'
                }
            },
            'labels': {
                'style': {
                    'color': 'Highcharts.getOptions().colors[0]'
                }
            }

        }],

        'tooltip': { 'shared': True,    },
        'legend': {
            'layout': 'vertical',
            'align': 'left',
            'x': 80,
            'verticalAlign': 'top',
            'y': 55,
            'floating': True,
            'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'"
        },
    }
    H.set_dict_options(options)

    for col_name in cc.cols_axe1:
      data = [[df[cc.coldate][i] , float(df[col_name][i]) ] for i in range(df.shape[0])]
      H.add_data_set(data, 'spline', col_name,yAxis=1)

    for col_name in cc.cols_axe2:
      data = [[df[cc.coldate][i] , float(df[col_name][i])] for i in range(df.shape[0])]
      H.add_data_set(data, 'spline', col_name, yAxis=0, )

    ##################################################################
    H.buildcontent()
    html_code = H._htmlcontent.decode('utf-8')
    return html_code


def pd_plot_histogram_highcharts_base(bins, vals, figsize=None,
                                  title=None,
                                  x_label=None, y_label=None, cfg:dict={}, mode='d3', save_img=False):
      from highcharts import Highchart
      from box import Box

      cc = Box(cfg)
      cc.title        = cc.get('title',    'Histogram' ) if title is None else title
      cc.figsize      = cc.get('figsize', (25, 15) )    if figsize is None else figsize
      cc.subtitle     = cc.get('subtitle', '')

      cc.x_label = 'Bins' if x_label is None else x_label
      cc.y_label = 'Frequency' if y_label is None else y_label

      ################################################################
      H = Highchart()
      options = {
        'chart': {
            'zoomType': 'xy',
            'width' :  cc.figsize[0],
            'height' : cc.figsize[1],
        },
        'title': {
            'text': cc.title
        },
        'xAxis': [{
            'categories' : bins
        }],
        'yAxis': [{
            'title': {
                'text': "Frequency",
                'style': {
                    'color': 'Highcharts.getOptions().colors[0]'
                }
            }
        }],
        'tooltip': {
            'shared': True,

        }
      }
      H.set_dict_options(options)

      H.add_data_set(vals, 'bar', cc.x_label)

      #############################################################
      H.buildcontent()
      html_code = H._htmlcontent.decode('utf-8')
      return html_code




def pd_plot_histogram_highcharts(df, col, figsize=None,
                                 title=None,
                                 cfg:dict={}, mode='d3', save_img=False):
    from box import Box

    cc = Box(cfg)
    cc.title        = cc.get('title',    'Histogram' + col ) if title is None else title
    cc.figsize      = cc.get('figsize', (25, 15) )    if figsize is None else figsize
    cc.subtitle     = cc.get('subtitle', '')
    x_label         = col+'-bins'
    y_label         = col+'-frequency'

    # Get data, calculate histogram and bar centers
    hist, bin_edges = np.histogram( df[col].values )
    bin_centers     = [float(bin_edges[i+1] + bin_edges[i]) / 2 for i in range(len(hist))]
    hist_val        = hist.tolist()

    #### Plot
    pd_plot_histogram_highcharts_base(bins    = bin_centers,
                                      vals    = hist_val,
                                      figsize = figsize,
                                      title   = title,
                                      x_label = x_label, y_label=y_label, cfg=cfg, mode=mode, save_img=save_img)






# Function to display highcharts graph
def html_show_chart_highchart(html_code, verbose=True):
    from highcharts import Highchart
    from IPython.core.display import display, HTML
    hc = Highchart()
    hc.buildhtmlheader()
    if verbose: print(html_code)
    display(HTML(hc.htmlheader + html_code))



# Function to display HTML
def html_show(html_code, verbose=True):
    from IPython.core.display import display, HTML
    display(HTML( html_code))




############################################################################################################################
############################################################################################################################
def images_to_html(dir_input="*.png",  title="", verbose=False):
    """
        images_to_html( model_path + "/graph_shop_17_past/*.png" , model_path + "shop_17.html" )
    """
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    import glob
    html = ""
    flist = glob.glob(dir_input)
    flist.sorted()
    for fp in flist:
        if verbose:
            print(fp, end=",")
        with open(fp, mode="rb") as fp2:
            tmpfile = fp2.read()
        encoded = base64.b64encode(tmpfile) .decode('utf-8')
        html = html + \
            f'<p><img src=\'data:image/png;base64,{encoded}\'> </p>\n'

    return html


############################################################################################################################
############################################################################################################################
def pd_plot_network(df):
    def pandas_plot_network_graph(df):
        """
           Plot network graph with pyviz, networks from pandas dataframe.
        """
        import pandas as pd
        import pyviz_app
        from pyviz_app.pyviz_app import Network
        from pyviz_app.pyviz_app import Node
        from pyviz_app.pyviz_app import Edge
        from pyviz_app.pyviz_app import Graph
        from pyviz_app.pyviz_app import Document

        g = Graph()
        for index, row in df.iterrows():
            node = Node(str(index), label=row['name'])
            g.add_node(node)
            for column in row.index:
                if row[column] != 0:
                    if column == 'name':
                        continue
                    edge = Edge(str(index), str(index), label=column)
                    g.add_edge(edge)
        return g

    def draw_graph3(networkx_graph, notebook=True, output_filename='graph.html', show_buttons=True, only_physics_buttons=False):
        """
        This function accepts a networkx graph object,
        converts it to a pyvis network object preserving its node and edge attributes,
        and both returns and saves a dynamic network visualization.

        Valid node attributes include:
            "size", "value", "title", "x", "y", "label", "color".

            (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

        Valid edge attributes include:
            "arrowStrikethrough", "hidden", "physics", "title", "value", "width"
            (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)

        Args:
            networkx_graph: The graph to convert and display
            notebook: Display in Jupyter?
            output_filename: Where to save the converted network
            show_buttons: Show buttons in saved version of network?
            only_physics_buttons: Show only buttons controlling physics of network?
        """

        # import
        from pyvis import network as net

        # make a pyvis network
        pyvis_graph = net.Network(notebook=notebook)

        # for each node and its attributes in the networkx graph
        for node, node_attrs in networkx_graph.nodes(data=True):
            pyvis_graph.add_node(str(node), **node_attrs)

        # for each edge and its attributes in the networkx graph
        for source, target, edge_attrs in networkx_graph.edges(data=True):
            # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
            if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                # place at key 'value' the weight of the edge
                edge_attrs['value'] = edge_attrs['weight']
            # add the edge
            pyvis_graph.add_edge(str(source), str(target), **edge_attrs)

        # turn buttons on
        if show_buttons:
            if only_physics_buttons:
                pyvis_graph.show_buttons(filter_=['physics'])
            else:
                pyvis_graph.show_buttons()

        # return and also save
        return pyvis_graph.show(output_filename)

    ##
    # For example:
    ##

    # make a new neworkx network
    import networkx as nx
    G = nx.Graph()

    # add nodes and edges (color can be html color name or hex code)
    G.add_node('a', color='red', size=4)
    G.add_node('b', color='#30a1a5', size=3)
    G.add_node('c', color='green', size=1)
    G.add_edge('a', 'b', weight=1023)
    G.add_edge('a', 'c', weight=435)
    G.add_edge('b', 'c', weight=100)

    # draw
    draw_graph3(G)


###################################################################################################
###################################################################################################
js_code = Box({})  # List of javascript code
js_code.js_hidden = """<SCRIPT>
function ShowAndHide() {
    var x = document.getElementById('hidden_section_id');
    if (x.style.display == 'none') {
        x.style.display = 'block';
    } else {
        x.style.display = 'none';
    }
}
</SCRIPT>
"""


###################################################################################################
if __name__ == "__main__":
    # python
    # fire.Fire()
    test_usage()
