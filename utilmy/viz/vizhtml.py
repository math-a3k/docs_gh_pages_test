""" Converter python ---> HTML
https://try2explore.com/questions/10109123
https://mpld3.github.io/examples/index.html
https://notebook.community/johnnycakes79/pyops/dashboard/pandas-highcharts-examples


https://datatables.net/


"""
import random, os, sys, numpy as np, pandas as pd
from datetime import datetime ; from typing import List
from tqdm import tqdm

from box import Box

#### Converting python --> HTML
import matplotlib.pyplot as plt
import mpld3
import pandas_highcharts
import pretty_html_table



##################################################################################################################
def log(*s):
    print(*s, flush=True)


###################################################################################
#### Example usage ################################################################
def test_usage():
  cfg = Box({})    #### pip install box-python    can use .key or ["mykey"]  for dict

  cfg.tseries = { "title": 'ok'}
  cfg.scatter = { "title": 'ok'}
  cfg.histo   = { "title": 'ok'}
  cfg.use_datatable = True


  df       = pd.DataFrame([[1,2]])
  df2_list = [ df, df, df]


  doc = htmlDoc(dir_out="", title="hrllo")

  doc.h1('My title')  ## h1
  doc.sep()
  doc.br()  ### <br>

  doc.tag('<h2> My graph title </h2>')
  doc.plot_scatter(df, cfg.scatter, mode='highcharts', save_img=False)
  doc.hr()   ###   doc.sep() line separator


  for df2_i in df2_list :
      doc.h3( f" plot title: {df2_i['category'].values[0]}" )
      doc.plot_tseries(df2_i, cfg.tseries, mode='highcharts', save_img=False)


  doc.tag('<h2> My histo title </h2>')
  doc.plot_histogram(df[['col1', 'col2' ]], cfg.histo, mode='mpld3', save_img=False)

  doc.table(df, format='blue_light')


  doc.tag("""<p>    My mutilines whatever I want to write
      ok
    """)

  doc.save(dir_out= "myfile.html")

  doc.open_browser()  #### Open myfile.html




#####################################################################################
#### Class ##########################################################################
class htmlDoc(object):
    def __init__(self, dir_out=None, mode="", title="", cfg:dict=None):

        self.cc      = Box(cfg)   #### Config dict
        self.dir_out = dir_out

        self.cc.use_datatable = self.cc.get('use_datatable', False)  ### Default val

        self.head = "<body>"
        self.html = """<body>        """


        if self.cc.use_datatable:
            self.head = self.head + """\n
              <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.css">
              <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.js"></script>
            """
            #https://datatables.net/manual/installation

            ### add $(document).ready( function () {    $('#table_id').DataTable(); } );



    def tag(self, x):  self.html += "\n" + x

    def h1(self, x)  : self.html += "\n" + f"<h1>{x}</h1>"
    def h2(self, x)  : self.html += "\n" + f"<h2>{x}</h2>"
    def h3(self, x)  : self.html += "\n" + f"<h3>{x}</h3>"
    def hr(self)     : self.html += "\n" + f"</hr>"
    def sep(self   ) : self.html += "\n" + f"</hr>"
    def br(self, x)  : self.html += "\n" + f"</br>"



    def save(self, dir_out=None):
        self.dir_out = dir_out if dir_out is not None else self.dir_out

        full  = self.head + "</head>" + self.html + "</body></html>"

        with open(self.dir_out, mode='w') as fp :
           fp.write(full)


    def open_browser(self):
        if os.name == 'nt': os.system(f'start chrome "{self.dir_out}/embeds.html" ')



    def table(self, df,  cfg:dict=None, mode='d3', use_datatable=False,  **kw ):
        ## show table in HTML : https://pypi.org/project/pretty-html-table/
        ## pretty_html_table
        html_code = pretty_html_table.build_table(df, mode)

        table_id = '1'

        if use_datatable :
            html_code += """$(document).ready( function () {    $('#{mytable_id}').DataTable(); } );""".replace('mytable_id', table_id)
        return html_code


    def plot_tseries(self, df,  cfg:dict=None, mode='d3', save_img=False,  **kw ):
     pass


    def plot_histogram(self, df,  cfg:dict=None, mode='d3', save_img=False,  **kw ):
     pass


    def plot_scatter(self, df,  cfg:dict=None, mode='d3', save_img=False,  **kw ):
     if mode == 'mpld3' :
        html_code  = pd_plot_scatter_mlpd3(df,  cfg, mode, save_img,  )

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



def pd_plot_highcharts(df ):
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
   # df = ... # create your dataframe here

   data = pandas_highcharts.serialize(df, render_to='my-chart', output_type='json')
   json_data_2 = "new Highcharts.StockChart(%s);" % pandas_highcharts.core.json_encode(data)

   html_code = """<div id="{chart_id}"</div>
      <script type="text/javascript">{data}</script>""".format(chart_id="new_brownian", data=json_data_2)
   return html_code




def pd_plot_scatter_mlpd3(df,  cfg:dict=None, mode='d3', save_img=False,  **kw ):
    """
    """
    cc = Box(cfg)

    cc.name      = cc.get('name',    'my scatter')
    cc.figsize   = cc.get('figsize', (25, 15) )   ### Dict type default values
    cc.title     = ' my graph title'
    cc.save_name = 'myfile'


    #######################################################################################
    # create data frame that has the result of the MDS plus the cluster numbers and titles
    cols = ['x', 'y',
            'label',                     ### label per point
            'class1', 'class1_color',    ### Color per point
            'class2', 'class2_size'      ### Size per point
           ]
    df   = df[cols]
    ##  df[cols]


    # group by cluster
    groups_clusters = df.groupby('class1')

    # set up plot
    fig, ax = plt.subplots(figsize=(25, 15))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return
    # the appropriate color/label
    for name, group in groups_clusters:
        ax.plot(group['x'], group['y'], marker='o', linestyle='', ms=2, label= group['class1'],
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
       plt.savefig(f'{cc.dir_out}/{cc.save_name}-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.png', dpi=200)

    # Plot
    fig, ax = plt.subplots(figsize=cc.figsize)  # set plot size
    ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    for name, group in groups_clusters:
        points = ax.plot(group.x, group.y, marker='o', linestyle='',
                         ms   = df['class2_size'].values,
                         label= df['class1'].values, mec='none',
                         color= df['class1_color'].values)
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


def mlpd3_add_tooltip(fig, points, labels):
        # set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=mpld3_CSS)
        # connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, mpld3_TopToolbar())




def pd_plot_histogram(dfi, path_save=None, nbin=20.0, q5=0.005, q95=0.995, nsample= -1, show=False, clear=True) :
    ### Plot histogram
    from matplotlib import pyplot as plt
    import numpy as np, os, time
    q0 = dfi.quantile(q5)
    q1 = dfi.quantile(q95)

    if nsample < 0 :
        dfi.hist( bins=np.arange( q0, q1,  (  q1 - q0 ) /nbin  ) )
    else :
        dfi.sample(n=nsample, replace=True ).hist( bins=np.arange( q0, q1,  (  q1 - q0 ) /nbin  ) )
    plt.title( path_save.split("/")[-1] )

    if show :
      plt.show()

    if path_save is not None :
      os.makedirs(os.path.dirname(path_save), exist_ok=True)
      plt.savefig( path_save )
      print(path_save )
    if clear :
        # time.sleep(5)
        plt.close()


def pd_plot_multi(df, plot_type=None, cols_axe1:list=[], cols_axe2:list=[],figsize=(8,4), spacing=0.1, **kwargs):
    from pandas import plotting
    from pandas.plotting import _matplotlib
    from matplotlib import pyplot as plt


    plt.figure(figsize= figsize )
    # Get default color style from pandas - can be changed to any other color list
    if cols_axe1 is None: cols_axe1 = df.columns
    if len(cols_axe1) == 0: return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols_axe1 + cols_axe2))

    # Displays subplot's pair in case of plot_type defined as `pair`
    if plot_type=='pair':
        ax = df.plot(subplots=True, figsize=figsize, **kwargs)
        plt.show()
        return

    # First axis
    ax = df.loc[:, cols_axe1[0]].plot(label=cols_axe1[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols_axe1[0])
    ##  lines, labels = ax.get_legend_handles_labels()
    lines, labels = [], []

    i1 = len(cols_axe1)
    for n in range(1, len(cols_axe1)):
        df.loc[:, cols_axe1[n]].plot(ax=ax, label=cols_axe1[n], color=colors[(n) % len(colors)], **kwargs)
        line, label = ax.get_legend_handles_labels()
        lines  += line
        labels += label

    for n in range(0, len(cols_axe2)):
        ######### Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        df.loc[:, cols_axe2[n]].plot(ax=ax_new, label=cols_axe2[n], color=colors[(i1 + n) % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols_axe2[n])

        ######### Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    plt.show()
    return ax



def mpld3_server_start():
    ### Windows specifc
    # if os.name == 'nt': os.system(f'start chrome "{dir_out}/embeds.html" ')
    # mpld3.show(fig=None, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None, **kwargs)[source]
    mpld3.show()  # show the plot



###################################################################################################
if __name__ == "__main__":
    ### python
    import fire
    fire.Fire()
