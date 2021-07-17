import argparse
import os

from utilmy.docs.generate_doc import run_markdown, run_table
from utilmy.docs.code_parser import  export_stats_perfile, export_stats_perrepo, export_stats_pertype


def run_cli():
    """ USage
    
    doc-gen  doc-gen  --repo_dir utilmy/      --doc_dir docs/"
    """
    p   = argparse.ArgumentParser()
    add = p.add_argument

    add("--repo_dir", type=str, default="./",     help = "repo_dir")
    add("--doc_dir",  type=str, default="docs/",  help = "doc_dir")
    add("--prefix",   type=str, default=None,     help = "https://github.com/user/repo/tree/a")
    args = p.parse_args()

    doc_dir        = args.doc_dir
    repo_stat_file = doc_dir + "/output_repo.csv"
    prefix         = args.prefix if args.prefix is not None else "./"

    os.makedirs(doc_dir, exist_ok=True)

    export_stats_perrepo(args.repo_dir,  repo_stat_file)
    run_markdown(repo_stat_file, output= doc_dir + '/doc_main.md',   prefix= prefix)
    run_table(repo_stat_file,    output= doc_dir + '/doc_table.md',  prefix= prefix)



#############################################################################
if __name__ == "__main__":
    run_cli()

