"""Automates Python scripts formatting, linting and Mkdocs documentation."""

import ast
import importlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Union, get_type_hints
import sys

def docstring_from_type_hints(repo_dir: Path, dirout:str, overwrite_script: bool = False, test: bool = True) -> str:
    """Automate docstring argument variable-type from type-hints.

    Args:
        repo_dir (< nothing >): textual directory to search for Python functions in
        overwrite_script (< wrong variable type>): enables automatic overwriting of Python scripts in repo_dir
        test (Something completely different): whether to write script content to a test_it.py file

    Returns:
        str: feedback message

    """
    p = repo_dir.glob("**/*.py")
    scripts = [x for x in p if x.is_file()]

    print(scripts)

    functions = defaultdict(list)
    for script in scripts:
        # print(script)

        with open(script, "r") as source:
            tree = ast.parse(source.read())

        function_docs = []
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if child.name not in ["main"]:

                    docstring_node = child.body[0]

                    try:
                        sys.path.append(script.parent)
                        module = importlib.import_module(script.stem)
                    except Exception as e:
                        return str(e)
                    f_ = getattr(module, child.name)


                    type_hints = get_type_hints(f_)  # the same as f_.__annotations__
                    return_hint = type_hints.pop("return", None)
                    function = f_.__name__
                    functions[script].append(function)
                    
                    print('----------------------')
                    print(type_hints)
                    if type_hints:

                        docstring = f'"""{ast.get_docstring(child, clean=True)}\n"""'
                        docstring_lines = docstring.split("\n")

                        if docstring:

                            args = re.search(
                                r'Args:(.*?)(Example[s]?:|Return[s]?:|""")',
                                docstring,
                                re.DOTALL,
                            )

                            new_arguments = {}
                            if args:

                                arguments = args.group()
                                argument_lines = arguments.split("\n")

                                exclude = [
                                    "Args:",
                                    "Example:",
                                    "Examples:",
                                    "Returns:",
                                    '"""',
                                ]

                                argument_lines = [arg for arg in argument_lines if arg]
                                argument_lines = [arg for arg in argument_lines if not any(x in arg for x in exclude)]

                                for argument in argument_lines:
                                    arg_name = argument.split()[0]
                                    if arg_name in argument:

                                        if argument.split(":"):
                                            if "(" and ")" in argument.split(":")[0]:

                                                variable_type = str(type_hints[arg_name])
                                                class_type = re.search(r"(<class ')(.*)('>)", variable_type)
                                                if class_type:
                                                    variable_type = class_type.group(2)

                                                new_argument_docstring = re.sub(
                                                    r"\(.*?\)",
                                                    f"({variable_type})",
                                                    argument,
                                                )

                                                idx = docstring_lines.index(f"{argument}")
                                                new_arguments[idx] = f"{new_argument_docstring}"

                                            else:
                                                print(f"no variable type in the argument: {arg_name}")
                                        else:
                                            print(f"no 'arg : description'-format for this argument: {arg_name}")
                                    else:
                                        print(f"no docstring for this argument: {arg_name}")
                            else:
                                print(f"there are no arguments in this docstring: {function}")

                            if return_hint:

                                raw_return = re.search(
                                    # r'(?<=Returns:\n).*',
                                    r"Return[s]?:\n(.*)",
                                    docstring,
                                    re.DOTALL,
                                )

                                if raw_return:

                                    return_argument = raw_return.group(1)
                                    return_lines = return_argument.split("\n")

                                    exclude = ["Returns:", '"""']

                                    return_lines = [return_arg for return_arg in return_lines if return_arg]
                                    return_lines = [
                                        return_arg
                                        for return_arg in return_lines
                                        if not any(x in return_arg for x in exclude)
                                    ]

                                    if return_lines and len(return_lines) == 1:

                                        return_arg = return_lines[0]
                                        if return_arg.split(":"):

                                            variable_type = str(return_hint)
                                            class_type = re.search(r"(<class ')(.*)('>)", variable_type)
                                            if class_type:
                                                variable_type = class_type.group(2)

                                            new_return_docstring = re.sub(
                                                r"\S(.*:)",
                                                f"{variable_type}:",
                                                return_arg,
                                            )

                                            idx = docstring_lines.index(f"{return_arg}")
                                            new_arguments[idx] = f"{new_return_docstring}\n"

                                        else:
                                            print(f"no variable-type in return statement docstring: {function}")
                                    else:
                                        print(f"no return statement docstring argument: {function}")
                                else:
                                    print(f"no return argument in docstring for function: {function}")
                            else:
                                print(f"no return type-hint for function: {function}")

                            sorted_arguments = sorted(new_arguments.items(), reverse=True)
                            for (idx, new_arg) in sorted_arguments:
                                docstring_lines[idx] = new_arg

                            docstring_lines = [f"{' '*docstring_node.col_offset}{line}" for line in docstring_lines]
                            new_docstring = "\n".join(docstring_lines)

                            function_docs.append(
                                (
                                    docstring_node.lineno,
                                    {
                                        "function_name": function,
                                        "col_offset": docstring_node.col_offset,
                                        "begin_lineno": docstring_node.lineno,
                                        "end_lineno": docstring_node.end_lineno,
                                        "value": new_docstring,
                                    },
                                )
                            )

                            # print(ast.unparse(child))
                            # you would be able to use ast.unparse(child), however this does not include # comments.
                            # https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c

                        else:
                            print(f"no docstring for function: {function}")
                    else:
                        print(f"no type-hints for function: {function}")

        with open(script, "r") as file:
            script_lines = file.readlines()

        function_docs.sort(key=lambda x: x[0], reverse=True)
        for (idx, docstring_attr) in function_docs:
            script_lines = (
                script_lines[: docstring_attr["begin_lineno"] - 1]
                + [f'{docstring_attr["value"]}\n']
                + script_lines[docstring_attr["end_lineno"] :]
            )

        if overwrite_script:
            if test:
                script = f"{dirout}/test_{script.stem}.py"
            else:
               script = script.replace( str(repo_dir), dirout)

            with open(script, "w") as script_file:
                script_file.writelines(script_lines)

            print(f"Automated docstring generation from type hints: {script}")

    return "Docstring generation from type-hints complete!"


##########################################################################################################
def main():
    """Execute when running this script."""
    python_tips_dir = Path.cwd().joinpath("utilmy/docs")

    docstring_from_type_hints(python_tips_dir, python_tips_dir, overwrite_script=True, test=True)



if __name__ == "__main__":
    # import fire
    # fire.Fire()
    main()


