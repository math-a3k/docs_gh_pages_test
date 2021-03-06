![test_fast_linux](https://github.com/arita37/cli_code/workflows/test_fast_linux/badge.svg)

# Command line utilities

## Install

    Stable : pip install -e git://github.com/arita37/cli_code.git@0.1#egg=cli_codde

    Latest : pip install -e git+git://github.com/arita37/cli_code.git#egg=clI_code

    Or git clone, pip install -e .

## Utilities available

- [Convert Notebook](#1-convert-notebook)
- [Search Github](#2-search-github)
- [Auto Create Conda Environment](#3-auto-create-conda-environment)
- [Auto-generate Package Docs](#4-auto-generate-package-docs)
- [Parse Python Modules](#5-parse-python-modules)
- [Easy Merge Conda Environmetns](#6-easy-merge-conda-environmetns)
- [Checkout any Github Repo](#7-get-github-repository-and-check-it-in-a-new-conda-environment)
- [Automate Downloading from Dropbox, Google Drive, and Github](#8-automate-downloading-from-dropbox-google-drive-and-github)
- [JSON and Code Parsing Utilities](#9-json-and-code-parsing-utilities)
- [Python Code Formatter](#10-python-code-formatter)

---

### 1. Convert Notebook

convert all IPython notebooks inside a directory to python scripts and save them to another directory. This also tests python scripts for potential syntax errors.

`cli_convert_jupyter -i /path/to/notebooks -o path/to/python-scripts`

[More Documentation](cli_code/cli_convert_ipynb.py)

---

### 2. Search Github

Search on github for keyword(s) with optional parameters to refine your search and get all results in a CSV file.

Usage:

`cli_github_search amazon scraper`

[More Documentation](cli_code/cli_github_search.py)

---

### 3. Auto Create Conda Environment

Automatically create conda virtual environment for a specified repository. It also autodetects all required packages and install them into the newly created environment.

Usage:

`cli_repo_install -i test -n notebook_cvt`

`cli_repo_install -i test -n notebook_cvt -py 3.6 -p tensorflow pandas`

[More Documentation](cli_code/cli_repo_install.py)

---

### 4. Auto-generate Package Docs

This script parses the python files matching the given pattern inside a directory and generates documentaion in csv which includes functions and classes with information about their arguments, their scope etc.

usage:

`cli_doc -i test -vvv --tab 4 --out docs.txt`

`cli_doc -i test -vvv --tab 4 --out test_out/docs.txt --filter ".*?api.py"`

[More Documentation](cli_code/cli_doc_auto/main.py)

---

### 5. Parse Python Modules

This scripts parses the source and generates its signature, source may be given either as a .py filepath or as a directory path containing multiple .py source files.

Usage:

`cli_env_module_parser -i /path/to/module(s) or package(s) -o module_parsed.csv`

[More Documentation](cli_code/cli_module_parser.py)

### 6. Easy Merge Conda Environmetns

This is a very simple piece of script that allows user to merge multiple YAML files generated by anaconda into a single unified YAML.

Usage:

`cli_conda_merge /path/to/env1.yaml /path/to/env2.yaml`

[More Documentation](cli_code/cli_conda_merge.py)

### 7. Get Github Repository and Check it in a New Conda Environment

This script automates code checking using a python and also checks files and provide signature of modules in a nicely formatted csv.

Usage:

`cli_repo_check https://www.github.com/{username}/{reponame}.git -n testing_env`

[More Documentation](cli_code/cli_repo_check.py)

### 8. Automate Downloading from Dropbox, Google Drive, and Github

This script automates downloading of bulk files from github, google drive and dr0pbox. You can either provide a single url or a file containg multiple urls.

Usage:

`cli_download -u a_valid_url`

`cli_download -f /path/to/a_valid_urls_file -o my_download_dir`

[More Documentation](cli_code/cli_download.py)

### 9. JSON and Code Parsing Utilities

Parse python source code and put into json format.

Usage:

`cli_json --path `

[More Documentation](cli_code/cli_json.py)

### 10. Python Code Formatter

Foramt a python module or a package based on some rules to make everyting pretty and more maintainable.

Usage:

`cli_format --dir_in /path/to/py_module --dir_out /path/to/output_dir`

[More Documentation](cli_code/cli_format2.py)
