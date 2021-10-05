# coding=utf-8
HELP = """
DataPrep — the most comprehensive auto EDA [GitHub, Documentation]
AutoViz — the fastest auto EDA [GitHub]
PandasProfiling — the earliest and one of the best auto EDA tools [GitHub, Documentation]
Lux — the most user-friendly and luxurious EDA [GitHub, Documentation]



### Data performance
https://github.com/evidentlyai/evidently



"""
import os,sys,  pandas as pd, numpy as np



#################################################################################################
def log(*s):
    print(*s, flush=True)


def help():
    ss  = ""
    ss += HELP
    print(ss)






#################################################################################################

















#################################################################################################
if __name__ == '__main__':
    import fire; fire.Fire()









