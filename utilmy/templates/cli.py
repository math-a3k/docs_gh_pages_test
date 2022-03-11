import argparse
import os


def run_cli():
    """ USage
    
    template  copy  --repo_dir utilmy/
    """
    p   = argparse.ArgumentParser()
    add = p.add_argument

    add("--action",   type=str,  default="show",           help = "repo_dir")

    add("--name",     type=str,  default="pypi_package",   help = "")
    add("--out_dir",  type=str,  default="mytemplate/",    help = "")
    args = p.parse_args()

    if args.action == 'show':
       template_show()

    if args.action == 'copy':
       template_copy(args.name, args.out_dir)


def template_show():
    """function template_show
    Args:
    Returns:
        
    """
    import glob
    this_repo = os.path.abspath(__file__).replace("\\", "/")
    flist     = os.walk(this_repo +"/templist/")
    print(flist)


def template_copy(name, out_dir):
    """function template_copy
    Args:
        name:   
        out_dir:   
    Returns:
        
    """
    from utilmy import os_copy
    import glob
    from pathlib import Path

    this_file = os.path.abspath(__file__).replace("\\", "/")
    this_repo = os.path.abspath(this_file + "/../")
    src       = this_repo + "/templates/tempfile/" + name
    os_copy(src, out_dir)
    print( Path(out_dir) /  Path(name) )


#############################################################################
if __name__ == "__main__":
    run_cli()

