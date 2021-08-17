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
    for i in range(100):
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
    doc.table(df, format='orange_light')


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
           Generate HTML Code to display graph


        """
        self.cc = Box(cfg)  # Config dict
        self.dir_out = dir_out

        self.cc.use_datatable = self.cc.get('use_datatable', False)  # Default val

        self.head = "<html>\n    <head>"
        self.html = """    <body>"""

        if self.cc.use_datatable:
            self.head = self.head + """
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.js"></script>"""
            # https://datatables.net/manual/installation
            # add $(document).ready( function () {    $('#table_id').DataTable(); } );

    def get_html(self):
        return self.html

    def tag(self, x):  self.html += "\n" + x
    def h1(self, x): self.html += "\n" + f"<h1>{x}</h1>"
    def h2(self, x): self.html += "\n" + f"<h2>{x}</h2>"
    def h3(self, x): self.html += "\n" + f"<h3>{x}</h3>"
    def h4(self, x): self.html += "\n" + f"<h4>{x}</h4>"
    def hr(self): self.html += "\n" + f"</hr>"
    def sep(self): self.html += "\n" + f"</hr>"
    def br(self): self.html += "\n" + f"</br>"
    def p(self, x): self.html += "\n" + f"<p>{x}</p>"


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


    def table(self, df, format='blue_light', use_datatable=False, table_id:int=1, **kw):
        """
        ## show table in HTML : https://pypi.org/project/pretty-html-table/
        """
        import pretty_html_table
        html_code = pretty_html_table.build_table(df, format)
        if use_datatable:
            html_code += """\n<script>$(document).ready( function () {    $('#{mytable_id}').DataTable(); } );</script>\n""".replace(
                'mytable_id', table_id)
        self.html += "\n\n" + html_code


    def plot_tseries(self, df,  cfg: dict = None, mode='mpld3', save_img="",  **kw):
        if mode == 'mpld3':
            fig = pd_plot_tseries_matplot(df)
            html_code = mpld3.fig_to_html(fig)
        elif mode == 'highcharts':
            fig = pd_plot_highcharts(df)
            html_code = mpld3.fig_to_html(fig)
        self.html += "\n\n" + html_code


    def plot_histogram(self, df,  cfg: dict = None, mode='mpld3', save_img="",  **kw):
        if mode == 'mpld3':
            fig = pd_plot_histogram_matplot(df)
            html_code = mpld3.fig_to_html(fig)
        self.html += "\n\n" + html_code


    def plot_scatter(self, df,  cfg: dict = None, mode='mpld3', save_img=False,  **kw):
        if mode == 'mpld3':
            html_code = pd_plot_scatter_matplot(df,  cfg, mode, save_img,)
        else:
            html_code = ''
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


def pd_plot_scatter_matplot(df,  cfg: dict = None, mode='d3', save_img=False,  **kw):
    """
    """
    cc = Box(cfg)
    cc.name = cc.get('name',    'my scatter')
    cc.figsize = cc.get('figsize', (25, 15))  # Dict type default values
    cc.title = ' my graph title'
    cc.save_name = 'myfile'

    #######################################################################################
    # create data frame that has the result of the MDS plus the cluster numbers and titles
    cols = ['x', 'y',
            'label',  # label per point
            'class1', 'class1_color',  # Color per point
            'class2', 'class2_size'  # Size per point
            ]
    df = df[cols]
    print(df)
    # df[cols]

    df['class1'] = df['class1'].fillna('NA1')
    df['class1_color'] = df['class1'].fillna(1)

    df['class2'] = df['class2'].fillna('NA2')
    df['class2_size'] = df['class2'].fillna(2)

    # group by cluster
    groups_clusters = df.groupby('class1')

    # set up plot
    fig, ax = plt.subplots(figsize=(25, 15))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return
    # the appropriate color/label
    for name, group in groups_clusters:
        ax.plot(group['x'], group['y'], marker='o', linestyle='', ms=2, label=group['class1'],
                color=group['class1_color'],
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

    # add label in x,y position with the label as the
    for i in range(len(df)):
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['label'], size=8)

    # uncomment the below to save the plot if need be
    if save_img:
        plt.savefig(
            f'{cc.dir_out}/{cc.save_name}-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.png', dpi=200)

    # Plot
    fig, ax = plt.subplots(figsize=cc.figsize)  # set plot size
    ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    for name, group in groups_clusters:
        points = ax.plot(group.x, group.y, marker='o', linestyle='',
                         ms=df['class2_size'].values,
                         label=df['class1'].values, mec='none',
                         color=df['class1_color'].values)
        ax.set_aspect('auto')
        labels = [i for i in group['label']]

        # set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        # set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        mlpd3_add_tooltip(fig, points, labels)
        # set tooltip using points, labels and the already defined 'css'
        # tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=mpld3_CSS)
        # connect tooltip to fig
        # mpld3.plugins.connect(fig, tooltip, mpld3_TopToolbar())

    ax.legend(numpoints=1)  # show legend with only one dot

    ##### Export ############################################################
    #mpld3.fig_to_html(fig, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False, **kwargs)[source]
    html_code = mpld3.fig_to_html(fig,  **kw)
    return html_code


def pd_matplotlib_histogram2(df, config):
    """
      return matplotlib figure histogram from pandas dataframe
        """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

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


def pd_plot_histogram_matplot(dfi, path_save=None, nbin=20.0, q5=0.005, q95=0.995, nsample=-1, show=False, clear=True):
    # Plot histogram
    from matplotlib import pyplot as plt
    import numpy as np
    import os
    import time
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


def pd_plot_tseries_matplot(df, plot_type=None, cols_axe1: list = [], cols_axe2: list = [], figsize=(8, 4), spacing=0.1, **kwargs):
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
        ax = df.plot(subplots=True, figsize=figsize, **kwargs)
        plt.show()
        return

    # First axis
    ax = df.loc[:, cols_axe1[0]].plot(
        label=cols_axe1[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols_axe1[0])
    ##  lines, labels = ax.get_legend_handles_labels()
    lines, labels = [], []

    i1 = len(cols_axe1)
    for n in range(1, len(cols_axe1)):
        df.loc[:, cols_axe1[n]].plot(
            ax=ax, label=cols_axe1[n], color=colors[(n) % len(colors)], **kwargs)
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label

    for n in range(0, len(cols_axe2)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        df.loc[:, cols_axe2[n]].plot(
            ax=ax_new, label=cols_axe2[n], color=colors[(i1 + n) % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols_axe2[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    plt.show()
    return ax


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