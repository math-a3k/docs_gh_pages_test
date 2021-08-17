# -*- coding: utf-8 -*-
'''    Data Analysis Utilities   '''
import sys, os
import numpy as np, pandas as pd, copy, scipy as sci, matplotlib.pyplot as plt, math as mth
import requests, re;
from bs4 import BeautifulSoup

from tabulate import tabulate;
from datetime import datetime;
from datetime import timedelta;
from calendar import isleap
from dateutil.parser import parse

import pylab as pl, itertools

from collections import OrderedDict

import fast, util




######################  DASHBOARD #################################################
'''
https://dashboards.ly/create
Plot LY details:
AI_data
ciYkluvrEP4Qneni833i
API Token :  wkcwcn8m6t

'''


################ Show Map

def map_show():
   pass
 '''
# Convert to interactive Leaflet map
>>> import mplleaflet
>>> mplleaflet.show()
https://github.com/jwass/mplleaflet

https://blog.modeanalytics.com/python-data-visualization-libraries/



'''





################################# Excel  Report    #####################################
'''
Sample report generation script from pbpython.com

This program takes an input Excel file, reads it and turns it into a
pivot table.

The output is saved in multiple tabs in a new Excel file.
'''


def xl_create_pivot(infile, index_list=["Manager", "Rep", "Product"], value_list=["Price", "Quantity"]):
   ''' Read in the Excel file, create a pivot table and return it as a DataFrame '''
   df=pd.read_excel(infile)
   table=pd.pivot_table(df, index=index_list, values=value_list, aggfunc=[np.sum, np.mean], fill_value=0)
   return table


def xl_save_report(report, outfile):
   '''  Take a report and save it to a single Excel file
       sales_report = create_pivot(args.infile.name)
       save_report(sales_report, args.outfile.name)
   '''
   writer=pd.ExcelWriter(outfile)
   for manager in report.index.get_level_values(0).unique():
      temp_df=report.xs(manager, level=0)
      temp_df.to_excel(writer, manager)
   writer.save()


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to generate sales report')
    parser.add_argument('infile', type=argparse.FileType('r'),help="report source file in Excel")
    parser.add_argument('outfile', type=argparse.FileType('w'),  help="output file in Excel")
    args = parser.parse_args()
    # We need to pass the full file name instead of the file object
    sales_report = create_pivot(args.infile.name)
    save_report(sales_report, args.outfile.name)
'''



def xl_create_pdf() :
   from __future__ import print_function
   from jinja2 import Environment, FileSystemLoader
   from weasyprint import HTML

   df=pd.read_excel(args.infile.name)
   sales_report=create_pivot(df, args.infile.name)

   # Get some national summary to include as well
   manager_df=[]
   for manager in sales_report.index.get_level_values(0).unique():
      manager_df.append([manager, sales_report.xs(manager, level=0).to_html()])

   # Do our templating now
   # We can specify any directory for the loader but for this example, use current director
   env=Environment(loader=FileSystemLoader('.'))
   template=env.get_template("myreport.html")
   template_vars={"title": "National Sales Funnel Report", "CPU": get_summary_stats(df, "CPU"), "Software": get_summary_stats(df, "Software"), "national_pivot_table": sales_report.to_html(), "Manager_Detail": manager_df}
   # Render our file and create the PDF using our css style file
   html_out=template.render(template_vars)
   HTML(string=html_out).write_pdf(args.outfile.name, stylesheets=["style.css"])

