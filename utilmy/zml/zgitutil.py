import os
import subprocess
import datetime
import ntpath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
files = list()
# files = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR)]  # list of the files in BASE directory
for root, directories, filenames in os.walk(BASE_DIR):
        if '.' in root:
                continue 
        for filename in filenames:
            files.append(os.path.join(root,filename))
# print("total files: ",len(files))

def path_leaf(path):
    ''' return filename from path '''
    return ntpath.basename(path)


def _run(*args):
    '''core function'''
    # print(list(args))
    try:
        return subprocess.check_call(['git'] + list(args))
    except:
        return 0

def commit(mylist):
    ''' commit function '''
    file_names = [path_leaf(l) for l in mylist]
    message = str( ",".join(file_names))[:100]
    commit_message = f'{message}'

    _run("commit", "-am", commit_message)
    _run("push", "-u", "origin", "master")


def _filter_on_size(size=0, f=files):
    """core function to filter files to be added, take size in bytes"""
    # files_list = [file for file in f if os.path.getsize(file) > size]
    file_list = list()
    datetime.datetime.fromtimestamp(1618829835.0847054)
    date_limit = datetime.datetime.now() - datetime.timedelta(weeks=1)
    for file in f :
        try:
            last_mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(file))
            if last_mod_date > date_limit and os.path.getsize(file) < size:
                # _run("add",file)
                file_list.append(file)
        except:
            continue


    # print(files_list)

    return file_list


def add(size=10000000):
    """function add
    Args:
        size:   
    Returns:
        
    """
    if size == 0:
        _run("add", ".")
    else:
        files = _filter_on_size(size)
        _run("add", *files)
        return files

def main():
    """function main
    Args:
    Returns:
        
    """
    print("adding files")
    files = add()  # change the number to filter files on size , size in bytes
    print('committing files')
    commit(files)
    print('done')


if __name__ == '__main__':
    main()
