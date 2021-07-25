'''
  Generate markdown documentation from code source.

    python code_parser.py  repo       --in_path  parser/test3    --out_path  parser/output/output_repo.csv
    python generate_doc.py markdown   --repo_stat_file parser/output/output_repo.csv      --output  parser/output/doc_repo.md  --prefix "https://github.com/arita37/zz936/tree/a"
    python generate_doc.py table      --repo_stat_file parser/output/output_repo.csv      --output  parser/output/doc_repo_table.md  --prefix "https://github.com/arita37/zz936/tree/a"


    --prefix: link to your repo
    ###   Note: the repo_stat_file csv file is the file was create python code_parser.py


'''
    
import pandas as pd
from ast import literal_eval
import fire




def markdown_create_function(uri, name, type, args_name, args_type, args_value, start_line, list_docs, prefix=""):
    rsp = '''
    <details>
        <summary>
        {} | <a name='{}' href='{}#L{}'>{}</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>{}</ul>
        <li>Docs:<br>{}</li>
        </ul>
    </details>'''

    function_name = uri.split(':', 1)[1]
    file          = uri.split(':', 1)[0]
    args_name     = literal_eval(args_name)
    args_type     = literal_eval(args_type)
    args_value    = literal_eval(args_value)
    list_docs     = literal_eval(list_docs)
    # print(args_name)
    docs_str = ''
    for doc in list_docs:
        docs_str += '{}<br>\n'.format(doc)

    args_str = ''
    for arg_name, arg_type, arg_value in zip(args_name, args_type, args_value):
        arg_type   = f': {arg_type}'   if arg_type  != None else ''
        arg_value  = f' = {arg_value}' if arg_value != None else ''
        args_str  += f'{arg_name}{arg_type}{arg_value},<br>'
    return {'file': file, 'info': rsp.format(type, uri, f'{prefix}/{file}',
                                             start_line, function_name, args_str, docs_str)}


def markdown_create_file(list_info, prefix=''):
    rsp = '''
<details>
<summary>
<a name='{}' href='{}'>{}</a>
</summary>
<ul>{}</ul>
</details>
'''
    output_str = ''
    list_files = set([data['file'] for data in list_info])
    for file in list_files:
        eles = ''
        for data in list_info:
            if data['file'] == file:
                eles += '{}'.format(data['info'])
        output_str += rsp.format(file, f'{prefix}/{file}', file, eles)
    # print(output_str)
    return(output_str)


def markdown_createall(dfi, prefix=""):
    result = [markdown_create_function(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], prefix) for row in zip(
               dfi['uri'], dfi['name'], dfi['type'], dfi['arg_name'], 
               dfi['arg_type'], dfi['arg_value'], dfi['line'], 
               dfi['docs'])]
    # print(result)
    list_markdown = markdown_create_file(result, prefix)
    return list_markdown

  
#############################################################################
############ TABLE
def table_create_row(uri, name, type, start_line, list_funtions, prefix):
    rsp  = "| <a name='{}' href='{}'>{}</a> | {} | <a href='{}#L{}'>{}</a> | {} |"
    list_funtions = literal_eval(list_funtions)
    print(list_funtions)
    funcs_str = ""
    for function in list_funtions:
        funcs_str += f'{function}<br>'
    file = uri.split(':', 1)[0]
    return rsp.format(file, f'{prefix}/{file}', file, type, f'{prefix}/{file}', start_line, name, funcs_str)


def table_all_row(list_rows):
    rsp = '''
| file | type | name  | List functions |
| ------- | --- | --- | -------------- |
'''
    for row in list_rows:
        rsp += f'{row}\n'
    return rsp


def table_create(dfi, prefix):
    list_rows = [table_create_row(row[0], row[1], row[2], row[3], row[4], prefix)
                 for row in zip(dfi['uri'], dfi['name'], dfi['type'], dfi['line'], dfi['list_functions'])]
    data = table_all_row(list_rows)
    return data


################################################################################
def run_markdown(repo_stat_file, output='docs/doc_main.md', prefix="https://github.com/user/repo/tree/a"):
    """ 
        python generate_doc.py run_all <in_file> <out_file> <prefix>
    Returns:
    """
    print(f"Start generate readme file {output} ... ")
    print(f'Prefix: {prefix}')
    dfi          = pd.read_csv(repo_stat_file)
    str_markdown = markdown_createall(dfi, prefix)
    with open(output, 'w+', encoding='utf-8') as f:
        # edit here to write
        f.write('# All files\n')
        f.write(str_markdown)
    print(f"Done.")


def run_table(repo_stat_file, output='docs/doc_table.md', prefix="https://github.com/user/repo/tree/a"):
    """ 
        python generate_doc.py run_table <in_file> <out_file> <prefix>
    Returns:
    """
    print(f"Start generate readme file {output} ... ")
    print(f'Prefix: {prefix}')
    dfi          = pd.read_csv(repo_stat_file)
    str_markdown = table_create(dfi, prefix)
    with open(output, 'w+', encoding='utf-8') as f:
        # edit here to write
        f.write('# Table\n')
        f.write(f'Repo: {prefix}\n')
        f.write(str_markdown)
    print(f"Done.")


################################################################################    
def test():
    input_file = 'parser/output/output_repo.csv'
    output_file = "test.md"
    prefix_repo = "https://github.com/arita37/zz936/tree/a"

    dfi = pd.read_csv(input_file)

    str_markdown = table_create(dfi, prefix_repo)
    with open(output_file, 'w+', encoding='utf-8') as f:
        f.write('# All files\n')
        f.write(f'Repo: {prefix_repo}\n')
        f.write(str_markdown)


################################################################################    
################################################################################    
if __name__ == '__main__':
    fire.Fire({
        'markdown'  : run_markdown,
        'table'     : run_table,
    })
    # test()

