import argparse
import os

from utilmy.docs import generate_doc as gdoc
from utilmy.docs import code_parser as cp


def run_cli():
    """ USage
    
    docs  markdown  --repo_dir utilmy/      --doc_dir docs/"

    docs  index      --repo_dir utilmy/      --doc_dir docs/"
    docs  callgraph  --repo_dir utilmy/      --doc_dir docs/"


    """
    p   = argparse.ArgumentParser()
    add = p.add_argument


    add('task', metavar='task', type=str, nargs=1, help='task to do')

    add("--repo_dir", type=str, default="./",     help = "repo_dir")
    add("--doc_dir",  type=str, default="docs/",  help = "doc_dir")
    add("--prefix",   type=str, default=None,     help = "https://github.com/user/repo/tree/a")
    args = p.parse_args()

    doc_dir        = args.doc_dir
    repo_stat_file = doc_dir + "/output_repo.csv"
    prefix         = args.prefix if args.prefix is not None else "./"

    if args.task == 'markdown':
        os.makedirs(doc_dir, exist_ok=True)
        cp.export_stats_perrepo(args.repo_dir,  repo_stat_file)
        gdoc.run_markdown(repo_stat_file, output= doc_dir + '/doc_main.md',   prefix= prefix)
        gdoc.run_table(repo_stat_file,    output= doc_dir + '/doc_table.md',  prefix= prefix)


    if args.task == 'index':
        os.makedirs(doc_dir, exist_ok=True)
        cp.export_stats_perrepo(args.repo_dir,  repo_stat_file)
        gdoc.run_markdown(repo_stat_file, output= doc_dir + '/doc_main.md',   prefix= prefix)
        gdoc.run_table(repo_stat_file,    output= doc_dir + '/doc_table.md',  prefix= prefix)


    if args.task == 'callgraph':
        os.makedirs(doc_dir, exist_ok=True)
        cp.export_stats_perrepo(args.repo_dir,  repo_stat_file)
        gdoc.run_markdown(repo_stat_file, output= doc_dir + '/doc_main.md',   prefix= prefix)
        gdoc.run_table(repo_stat_file,    output= doc_dir + '/doc_table.md',  prefix= prefix)


    if args.task == 'csv':
        os.makedirs(doc_dir, exist_ok=True)
        cp.export_stats_perrepo(args.repo_dir,  repo_stat_file)
        gdoc.run_markdown(repo_stat_file, output= doc_dir + '/doc_main.md',   prefix= prefix)
        gdoc.run_table(repo_stat_file,    output= doc_dir + '/doc_table.md',  prefix= prefix)

#############################################################################
if __name__ == "__main__":
    run_cli()

