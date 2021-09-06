# -*- coding: utf-8 -*-
HELP ="""
########  Usage 
    docs markdown --repo_url  https://github.com/arita37/spacefusion.git   --out_dir ./

    docs  markdown   --repo_dir utilmy/      --out_dir docs/
    docs  callgraph  --repo_dir utilmy/      --out_dir docs/
    docs  csv        --repo_dir utilmy/      --out_dir docs/
    docs  txt        --repo_dir utilmy/      --out_dir docs/
    
    



###########
    python utilmy/docs/cli.py markdown --repo_url https://github.com/CompVis/taming-transformers.git

    python utilmy/docs/cli.py markdown --repo_dir utilmy/      --out_dir docs/

    python utilmy/docs/cli.py callgraph  --repo_url https://github.com/CompVis/taming-transformers.git

    python utilmy/docs/cli.py callgraph  --repo_dir utilmy/      --out_dir docs/

    python utilmy/docs/cli.py csv  --repo_url https://github.com/CompVis/taming-transformers.git

    python utilmy/docs/cli.py csv  --repo_dir utilmy/      --out_dir docs/

    python utilmy/docs/cli.py txt  --repo_url https://github.com/CompVis/taming-transformers.git

    python utilmy/docs/cli.py csv  --repo_dir utilmy/      --out_dir docs/


###########
      'type': export_stats_pertype,
      'file': export_stats_perfile,
      'repo': export_stats_perrepo,
      
      'repo_url': export_stats_repolink,
      
      'export_call_graph': export_call_graph,
      'export_call_graph_url': export_call_graph_url,


"""
import argparse, os
import  stdlib_list

###  !!!!!!!! Do NOT Change this import structure
from utilmy.docs import generate_doc as gdoc
from utilmy.docs import code_parser as cp
###################################################


def run_cli():
    """ Usage
    cd myutil
    pip install -e  .

    docs  help
    docs markdown --repo_url  https://github.com/arita37/spacefusion.git   --out_dir ./

    docs  callgraph  --repo_dir utilmy/      --out_dir docs/
    docs  csv        --repo_dir utilmy/      --out_dir docs/
    docs  txt        --repo_dir utilmy/      --out_dir docs/


    """
    p   = argparse.ArgumentParser()
    add = p.add_argument

    add('task', metavar='task', type=str, nargs=1, help='markdown/index/callgraph/csv/help')

    add("--repo_url", type=str, default=None,     help = "repo_url")
    add("--repo_dir", type=str, default="./",     help = "repo_dir")
    add("--out_dir",  type=str, default="docs/",  help = "doc_dir")
    add("--prefix",   type=str, default=None,     help = "https://github.com/user/repo/tree/a")
    args = p.parse_args()

    doc_dir            = args.out_dir
    prefix             = args.prefix if args.prefix is not None else "./"

    repo_stat_csv_file = doc_dir + "/output_repo.csv"
    repo_sta_txt_file  = doc_dir + "/output_repo.py"
    repo_graph_file    = doc_dir + "/output_repo_graph.csv"

    if args.task[0] == 'help':
        print(HELP)

    ###############################################################################################        
    os.makedirs(os.path.abs(doc_dir), exists_ok=True)            
    if args.task[0] == 'markdown':
        if args.repo_url is not None :
            cp.export_stats_repolink(args.repo_url,  repo_stat_csv_file)

        elif args.repo_dir is not None :
            cp.export_stats_perrepo(args.repo_dir,  repo_stat_csv_file)
        else:
            raise Exception(" Needs repo_url or repo_dir")

        gdoc.run_markdown(repo_stat_csv_file, output= doc_dir + '/doc_main.md',   prefix= prefix)
        gdoc.run_table(repo_stat_csv_file,    output= doc_dir + '/doc_table.md',  prefix= prefix)


    if args.task[0] == 'callgraph':
        if args.repo_url is not None :
            cp.export_call_graph_url(args.repo_url,  repo_graph_file)

        elif args.repo_dir is not None :
            cp.export_call_graph(args.repo_dir,  repo_graph_file)
        else:
            raise Exception(" Needs repo_url or repo_dir")

    if args.task[0] == 'csv':
        if args.repo_url is not None :
            cp.export_stats_repolink(args.repo_url,  repo_stat_csv_file)

        elif args.repo_dir is not None :
            cp.export_stats_perrepo(args.repo_dir,  repo_stat_csv_file)
        else:
            raise Exception(" Needs repo_url or repo_dir")

    if args.task[0] == 'txt':
        if args.repo_url is not None :
            cp.export_stats_repolink_txt(args.repo_url,  repo_sta_txt_file)

        elif args.repo_dir is not None :
            cp.export_stats_perrepo_txt(args.repo_dir,  repo_sta_txt_file)
        else:
            raise Exception(" Needs repo_url or repo_dir")


#############################################################################
if __name__ == "__main__":
    run_cli()

