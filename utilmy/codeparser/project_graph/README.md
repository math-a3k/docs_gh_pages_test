


https://www.statworx.com/en/blog/how-to-automatically-create-project-graphs-with-call-graph/


```
How to Use Project Graphs
Installation
Within your project’s environment, do the following:

brew install graphviz

pip install git+https://github.com/fior-di-latte/project_graph.git
Usage
Within your project’s environment, change your current working directory to the project’s root (this is important!) and then enter for standard usage:

project_graph myscript.py
If your script includes an argparser, use:

project_graph "myscript.py <arg1> <arg2> (...)"
If you want to see the entire graph, including all external packages, use:

project_graph -a myscript.py
If you want to use a visibility threshold other than 1%, use:

project_graph -m <percent_value> myscript.py
Finally, if you want to include external packages into the graph, you can specify them as follows:

project_graph -x <package1> -x <package2> (...) myscript.py

```

