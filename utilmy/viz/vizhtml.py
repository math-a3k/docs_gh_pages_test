
"""

https://try2explore.com/questions/10109123

https://mpld3.github.io/examples/index.html

https://notebook.community/johnnycakes79/pyops/dashboard/pandas-highcharts-examples

"""
import random, os, sys, numpy as np, pandas as pd, mpld3
from datetime import datetime ; from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm

from box import Box 

##################################################################################################################
def log(*s):
    print(*s, exist_ok=True)


###################################################################################
#### Example usage ################################################################
cfg = Box({})    #### pip install box-python    can use .key or ["mykey"]  for dict

cfg.tseries = { "title": 'ok'}
cfg.scatter = { "title": 'ok'}
cfg.histo   = { "title": 'ok'}


doc = htmlDoc(dir_out="")

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
doc.plot_histogram(df3, cfg.histo, mode='mpld3', save_img=False)


doc.tag("""<p>    My mutilplin whatever I want to write
    ok
  """)

doc.save(dir_out= "myfile.html")





#####################################################################################
#### Prototype Class ################################################################
class htmlDoc(object):
   def __init__()
      self.html= """
      <head>

      <body>


      """
   

  def h1(self, x):
    self.html += "\n" + f"<h1>{x}</h1>" 



  def plot_scatter(self, df,  cfg:dict={}, mode='d3', save_img=False,  **kw ):

     if mode == 'mpld3' :
        html_code  = plot_scatter_mlpd3(df,  cfg, mode, save_img,  )

     self.html += "\n\n" + html_code 



    
    
    

##################################################################################################################
CSS = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
"""



class TopToolbar(mpld3.plugins.PluginBase):
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

    

def plot_scatter_mlpd3(df,  cfg:dict={}, mode='d3', save_img=False,  **kw ):
    """

    """
    cc = Box(cfg)

    cc.name      = cc.get('name',    'my scatter')  
    cc.figsize   = cc.get('figsize', (25, 15) )   ### Dict type default values
    cc.title     = ' my graph title'
    cc.save_name = 'myfile'


    #######################################################################################
    # create data frame that has the result of the MDS plus the cluster numbers and titles
    cols = ['x', 'y', 'label', 'class1', 'class1_color', 'class2', 'class2_size']
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

        # set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=CSS)
        # connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())

        # set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        # set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    ax.legend(numpoints=1)  # show legend with only one dot



    ##### Export ############################################################
    #mpld3.fig_to_html(fig, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False, **kwargs)[source]

    html_code = mpld3.fig_to_html(fig,  **cfg.html_opts)
    return html_code







def html_add():
    ### Windows specifc
    if os.name == 'nt': os.system(f'start chrome "{dir_out}/embeds.html" ')


    if show_server :
       # mpld3.show(fig=None, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None, **kwargs)[source] 
       mpld3.show()  # show the plot



###################################################################################################
if __name__ == "__main__":
    ### python 
    import fire
    fire.Fire()





