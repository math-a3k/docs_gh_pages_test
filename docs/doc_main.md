# All files

<details>
<summary>
<a name='./utilmy/images.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/images.py'>./utilmy/images.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/images.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/images.py#L13'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/images.py:deps' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/images.py#L17'>deps</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/images.py:read_image' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/images.py#L24'>read_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filepath_or_buffer:  typing.Union[str,<br>io.BytesIO],<br></ul>
        <li>Docs:<br>    """Read a file into an image object
<br>
    Args:
<br>
        filepath_or_buffer: The path to the file, a URL, or any object
<br>
            with a `read` method (such as `io.BytesIO`)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/images.py:visualize_in_row' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/images.py#L53'>visualize_in_row</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>**images,<br></ul>
        <li>Docs:<br>    """Plot images in one row."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/images.py:maintain_aspect_ratio_resize' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/images.py#L68'>maintain_aspect_ratio_resize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>image,<br>width = None,<br>height = None,<br>inter = cv2.INTER_AREA,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/configs/util_config.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py'>./utilmy/configs/util_config.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L23'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:loge' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L27'>loge</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:config_load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L32'>config_load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config_path:  str  =  None,<br>path_default:  str  =  None,<br>config_default:  dict  =  None,<br>save_default:  bool  =  False,<br>to_dataclass:  bool  =  True,<br>,<br></ul>
        <li>Docs:<br>    """Load Config file into a dict
<br>
    1) load config_path
<br>
    2) If not, load in USER/.myconfig/.config.yaml
<br>
    3) If not, create default save in USER/.myconfig/.config.yaml
<br>
    Args:
<br>
        config_path:   path of config or 'default' tag value
<br>
        path_default : path of default config
<br>
        config_default: dict value of default config
<br>
        save_default: save default config on disk
<br>
    Returns: dict config
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:config_isvalid_yamlschema' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L105'>config_isvalid_yamlschema</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config_dict:  dict,<br>schema_path:  str  =  'config_val.yaml',<br>silent:  bool  =  False,<br></ul>
        <li>Docs:<br>    """Validate using a  yaml file
<br>
    Args:
<br>
        config_dict:
<br>
        schema_path:
<br>
        silent:
<br>
    Returns: True/False
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:config_isvalid_pydantic' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L129'>config_isvalid_pydantic</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config_dict:  dict,<br>pydanctic_schema:  str  =  'config_py.yaml',<br>silent:  bool  =  False,<br></ul>
        <li>Docs:<br>    """Validate using a pydantic files
<br>
    Args:
<br>
        config_dict:
<br>
        pydanctic_schema:
<br>
        silent:
<br>
    Returns: True/False
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:convert_yaml_to_box' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L147'>convert_yaml_to_box</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>yaml_path:  str,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:convert_dict_to_pydantic' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L153'>convert_dict_to_pydantic</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config_dict:  dict,<br>schema_name:  str,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:pydantic_model_generator' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L166'>pydantic_model_generator</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input_file:  Union[Path,<br>str],<br>input_file_type,<br>output_file:  Path,<br>**kwargs,<br>,<br></ul>
        <li>Docs:<br>    """
<br>
    Args:
<br>
        input_file:
<br>
        input_file_type:
<br>
        output_file:
<br>
        **kwargs:
<br>

<br>
    Returns:
<br>
    # https://github.com/koxudaxi/datamodel-code-generator
<br>
    # pip install datamodel-code-generator
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:test_yamlschema' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L200'>test_yamlschema</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:test_pydanticgenrator' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L206'>test_pydanticgenrator</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:test4' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L221'>test4</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/configs/util_config.py:test_example' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/configs/util_config.py#L227'>test_example</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/templates/cli.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/cli.py'>./utilmy/templates/cli.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/templates/cli.py:run_cli' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/cli.py#L5'>run_cli</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """ USage
<br>
    
<br>
    template  copy  --repo_dir utilmy/
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/cli.py:template_show' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/cli.py#L26'>template_show</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/cli.py:template_copy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/cli.py#L33'>template_copy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name,<br>out_dir,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/tabular.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py'>./utilmy/tabular.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L12'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:test_anova' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L19'>test_anova</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col1,<br>col2,<br></ul>
        <li>Docs:<br>    """
<br>
    ANOVA test two categorical features
<br>
    Input dfframe, 1st feature and 2nd feature
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:test_normality2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L48'>test_normality2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>column,<br>test_type,<br></ul>
        <li>Docs:<br>    """
<br>
    Function to check Normal Distribution of a Feature by 3 methods
<br>
    Input dfframe, feature name, and a test type
<br>
    Three types of test
<br>
    1)'Shapiro'
<br>
    2)'Normal'
<br>
    3)'Anderson'
<br>

<br>
    output the statistical test score and result whether accept or reject
<br>
    Accept mean the feature is Gaussain
<br>
    Reject mean the feature is not Gaussain
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:test_plot_qqplot' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L95'>test_plot_qqplot</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col_name,<br></ul>
        <li>Docs:<br>    """
<br>
    Function to plot boxplot, histplot and qqplot for numerical feature analyze
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:test_heteroscedacity' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L112'>test_heteroscedacity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y,<br>y_pred,<br>pred_value_only = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:pd_stat_distribution_colnum' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L289'>pd_stat_distribution_colnum</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>nrows = 2000,<br>verbose = False,<br></ul>
        <li>Docs:<br>    """ Stats the tables
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:pd_stat_histogram' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L329'>pd_stat_histogram</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>bins = 50,<br>coltarget = "diff",<br></ul>
        <li>Docs:<br>    """
<br>
    :param df:
<br>
    :param bins:
<br>
    :param coltarget:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:np_col_extractname' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L344'>np_col_extractname</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>col_onehot,<br></ul>
        <li>Docs:<br>    """
<br>
    Column extraction from onehot name
<br>
    :param col_onehotp
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:np_list_remove' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L367'>np_list_remove</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cols,<br>colsremove,<br>mode = "exact",<br></ul>
        <li>Docs:<br>    """
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:pd_stat_shift_trend_changes' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L392'>pd_stat_shift_trend_changes</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>feature,<br>target_col,<br>threshold = 0.03,<br></ul>
        <li>Docs:<br>    """
<br>
    Calculates number of times the trend of feature wrt target changed direction.
<br>
    :param df: df_grouped dataset
<br>
    :param feature: feature column name
<br>
    :param target_col: target column
<br>
    :param threshold: minimum % difference required to count as trend change
<br>
    :return: number of trend chagnes for the feature
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:pd_stat_shift_trend_correlation' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L416'>pd_stat_shift_trend_correlation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>df_test,<br>colname,<br>target_col,<br></ul>
        <li>Docs:<br>    """
<br>
    Calculates correlation between train and test trend of colname wrt target.
<br>
    :param df: train df data
<br>
    :param df_test: test df data
<br>
    :param colname: colname column name
<br>
    :param target_col: target column name
<br>
    :return: trend correlation between train and test
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:pd_stat_shift_changes' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L447'>pd_stat_shift_changes</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>target_col,<br>features_list = 0,<br>bins = 10,<br>df_test = 0,<br></ul>
        <li>Docs:<br>    """
<br>
    Calculates trend changes and correlation between train/test for list of features
<br>
    :param df: dfframe containing features and target columns
<br>
    :param target_col: target column name
<br>
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
<br>
    :param bins: number of bins to be created from continuous colname
<br>
    :param df_test: test df which has to be compared with input df for correlation
<br>
    :return: dfframe with trend changes and trend correlation (if test df passed)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/tabular.py:np_conv_to_one_col' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/tabular.py#L492'>np_conv_to_one_col</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>np_array,<br>sep_char = "_",<br></ul>
        <li>Docs:<br>    """
<br>
    converts string/numeric columns to one string column
<br>
    :param np_array: the numpy array with more than one column
<br>
    :param sep_char: the separator character
<br>
    """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/docs/code_parser.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py'>./utilmy/docs/code_parser.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_function_name' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L26'>get_list_function_name</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get all functions of the python file
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: list_functions   - List all python functions in the input file
<br>
    Example Output:
<br>
        ['func1', 'func2']
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_class_name' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L48'>get_list_class_name</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get all classes of the python file
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: list_classes     - List all python classes in the input file
<br>
    Example Output:
<br>
        ['Class1', 'Class1']
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_class_methods' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L73'>get_list_class_methods</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get all classes and all methods in this class of the python file
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: An array of class info [{dict}, {dict}, ...]
<br>
    Example Output:
<br>
    [
<br>
        {"class_name": "Class1", "listMethods": ["method1", "method2", "method3"]},
<br>
        {"class_name": "Class2", "listMethods": ["method4", "method5", "method6"]},
<br>
    ]
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_variable_global' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L105'>get_list_variable_global</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get all global variable of the python file
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: list_var         - Array of all global variable
<br>
    Example Output:
<br>
        ['Var1', 'Var2']
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_docs' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L127'>_get_docs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>all_lines,<br>index_1,<br>func_lines,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_function_info' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L167'>get_list_function_info</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get functions stars
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Array of functions, lines of the function, and variable in function
<br>
    Example Output:
<br>
        [
<br>
            {"name": "function_name1", "lines": 20, "variables": ["a", "b", "c"]},
<br>
            {"name": "function_name2", "lines": 30, "variables": []},
<br>
        ]
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_class_info' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L205'>get_list_class_info</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The class use to get functions stars
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Array of functions, lines of the function, and variable in function
<br>
    Example Output:
<br>
        [
<br>
            {"function": "function_name1", "lines": 20, "variables": ["a", "b", "c"]},
<br>
            {"function": "function_name2", "lines": 30, "variables": []},
<br>
        ]
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_method_info' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L241'>get_list_method_info</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """get_list_method_info
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Array of methods in class
<br>
    Example Output:
<br>
        [
<br>
            {"function": "function_name1", "lines": 20, "variables": ["a", "b", "c"]},
<br>
            {"function": "function_name2", "lines": 30, "variables": []},
<br>
        ]
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_method_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L283'>get_list_method_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get methods stars
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Dataframe with bellow fields
<br>
            uri:   path1/path2/filename.py:function1
<br>
            name: function1
<br>
            n_lines
<br>
            n_words
<br>
            n_words_unqiue
<br>
            n_characters
<br>
            avg_char_per_word = n_charaecter / n_words
<br>
            n_loop  : nb of for, while loop
<br>
            n_ifthen  : nb of if_then
<br>
        
<br>
        **** return None if no function in file
<br>
    Example Output:
<br>
                                                    uri                                               name    type  n_variable  n_words  n_words_unique  n_characters  avg_char_per_word  n_loop  n_ifthen
<br>
    0   d:/Project/job/test2/zz936/parser/test/keys.py...                              VerifyingKey:__init__  method           2       11              11           100           9.090909       0         1      
<br>
    1   d:/Project/job/test2/zz936/parser/test/keys.py...                     VerifyingKey:from_public_point  method          10       13              12           185          14.230769       0         0      
<br>
    2   d:/Project/job/test2/zz936/parser/test/keys.py...                           VerifyingKey:from_string  method          17       45              39           504          11.200000       0         1      
<br>
    3   d:/Project/job/test2/zz936/parser/test/keys.py...                              VerifyingKey:from_pem  method           2        2               2            39          19.500000       0         0      
<br>
    4   d:/Project/job/test2/zz936/parser/test/keys.py...                              VerifyingKey:from_der  method          19       64              38           683          10.671875       0         3      
<br>
    5   d:/Project/job/test2/zz936/parser/test/keys.py...              VerifyingKey:from_public_key_recovery  method           4        8               8           137          17.125000       0         0      
<br>
    6   d:/Project/job/test2/zz936/parser/test/keys.py...  VerifyingKey:from_public_key_recovery_with_digest  method          13       24              23           288          12.000000       0         0      
<br>
    7   d:/Project/job/test2/zz936/parser/test/keys.py...                             VerifyingKey:to_string  method           6       11               8           145          13.181818       0         0      
<br>
    8   d:/Project/job/test2/zz936/parser/test/keys.py...                                VerifyingKey:to_pem  method           2        4               4            42          10.500000       0         0  
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_class_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L319'>get_list_class_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The class use to get functions stars
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Dataframe with bellow fields
<br>
            uri:   path1/path2/filename.py:function1
<br>
            name: function1
<br>
            n_lines
<br>
            n_words
<br>
            n_words_unqiue
<br>
            n_characters
<br>
            avg_char_per_word = n_charaecter / n_words
<br>
            n_loop  : nb of for, while loop
<br>
            n_ifthen  : nb of if_then
<br>
        
<br>
        **** return None if no function in file
<br>
    Example Output:
<br>
                                                    uri               name   type  n_variable  n_words  n_words_unique  n_characters  avg_char_per_word  n_loop  n_ifthen
<br>
    0  d:/Project/job/test2/zz936/parser/test/keys.py...  BadSignatureError  class           0        1               1             4           4.000000       0         0
<br>
    1  d:/Project/job/test2/zz936/parser/test/keys.py...     BadDigestError  class           0        1               1             4           4.000000       0         0
<br>
    2  d:/Project/job/test2/zz936/parser/test/keys.py...       VerifyingKey  class          84      301             189          3584          11.906977       0         7
<br>
    3  d:/Project/job/test2/zz936/parser/test/keys.py...         SigningKey  class         138      482             310          4615           9.574689       3         9
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_list_function_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L350'>get_list_function_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get functions stars
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Dataframe with bellow fields
<br>
            uri:   path1/path2/filename.py:function1
<br>
            name: function1
<br>
            n_lines
<br>
            n_words
<br>
            n_words_unqiue
<br>
            n_characters
<br>
            avg_char_per_word = n_charaecter / n_words
<br>
            n_loop  : nb of for, while loop
<br>
            n_ifthen  : nb of if_then
<br>
        
<br>
        **** return None if no function in file
<br>
    Example Output:
<br>
            uri                                 name  n_variable  n_words  n_words_unique  n_characters  avg_char_per_word  n_loop  n_ifthen
<br>
        0   d:\Project\job\test2\zz936\parser/test/test2.p...     prepare_target_and_clean_up_test           8       92              32           535           5.815217       0         0
<br>
        1   d:\Project\job\test2\zz936\parser/test/test2.p...                 clean_up_config_test           6       55              19           241           4.381818       0         1
<br>
        2   d:\Project\job\test2\zz936\parser/test/test2.p...         check_default_network_config          22      388              74           955           2.461340       1         5
<br>
        3   d:\Project\job\test2\zz936\parser/test/test2.p...                     check_module_env           9      250              54           553           2.212000       1         1
<br>
        4   d:\Project\job\test2\zz936\parser/test/test2.p...     provision_certificates_to_target           7      101              29           384           3.801980       0         3
<br>
        5   d:\Project\job\test2\zz936\parser/test/test2.p...            config_session_connection           2       14               8            97           6.928571       0         0
<br>
        6   d:\Project\job\test2\zz936\parser/test/test2.p...  config_cipher_suite_and_tcps_action           8      101              30           335           3.316832       0         3
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L386'>get_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df: pd.DataFrame,<br>file_path: str,<br></ul>
        <li>Docs:<br>    """ Calculate stats from datafaframe
<br>
    Args:
<br>
        df: pandas DataFrame
<br>

<br>
    Returns:
<br>
        pandas DataFrame
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:get_file_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L415'>get_file_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """The function use to get file stars
<br>
    Args:
<br>
        IN: file_path         - the file path input
<br>
        OUT: Dict of file stars
<br>
    Example Output:
<br>
        {
<br>
            "total_functions": 22,
<br>
            "avg_lines" : 110.2,
<br>
            "total_class": 3
<br>
        }
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_words' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L441'>_get_words</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>row,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_avg_char_per_word' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L451'>_get_avg_char_per_word</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>row,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_validate_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L455'>_validate_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """Check if the file is existed and it's a python file
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_clean_data' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L470'>_clean_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>array,<br></ul>
        <li>Docs:<br>    """Remove empty lines and comment lines start with #
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_remove_empty_line' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L513'>_remove_empty_line</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>line,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_remmove_commemt_line' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L517'>_remmove_commemt_line</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>line,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_and_clean_all_lines' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L523'>_get_and_clean_all_lines</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """Prepare all lines of the file
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_all_line' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L533'>_get_all_line</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_all_lines_in_function' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L539'>_get_all_lines_in_function</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>function_name,<br>array,<br>indentMethod = '',<br></ul>
        <li>Docs:<br>    """The function use to get all lines of the function
<br>
    Args:
<br>
        IN: function_name - name of the function will be used to get all line
<br>
        IN: array         - list all lines of the file have this input function
<br>
        OUT: list_lines   - Array of all line of this function
<br>
        OUT: indent       - The indent of this function (this will be used for another calculation)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_all_lines_in_class' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L584'>_get_all_lines_in_class</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>class_name,<br>array,<br></ul>
        <li>Docs:<br>    """The function use to get all lines of the class
<br>
    Args:
<br>
        IN: class_name    - name of the class will be used to get all line
<br>
        IN: array         - list all lines of the file have this input class
<br>
        OUT: list_lines   - Array of all line of this class
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_all_lines_define_function' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L623'>_get_all_lines_define_function</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>function_name,<br>array,<br>indentMethod = '',<br></ul>
        <li>Docs:<br>    """The function use to get all lines define_function
<br>
    Args:
<br>
        IN: function_name - name of the function will be used to get all line
<br>
        IN: array         - list all lines of the file have this input function
<br>
        OUT: list_lines   - Array of all line used to define the function
<br>
        OUT: indent       - The indent of this function (this will be used for another calculation)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_define_function_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L660'>_get_define_function_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>array,<br></ul>
        <li>Docs:<br>    """The function use to get define function stats: arg_name, arg_type, arg_value
<br>
    Args:
<br>
        IN: array         - list all lines of function to get variables
<br>
        OUT: function stats: arg_name, arg_type, arg_value
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:_get_function_stats' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L721'>_get_function_stats</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>array,<br>indent,<br></ul>
        <li>Docs:<br>    """The function use to get all lines of the function
<br>
    Args:
<br>
        IN: indent        - indent string
<br>
        IN: array         - list all lines of function to get variables
<br>
        OUT: list_var     - Array of all variables
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:export_stats_pertype' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L810'>export_stats_pertype</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>in_path: str = None,<br>type: str = None,<br>out_path: str = None,<br></ul>
        <li>Docs:<br>    """
<br>
        python code_parser.py type <in_path> <type> <out_path>
<br>
    Returns:
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:export_stats_perfile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L836'>export_stats_perfile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>in_path: str = None,<br>out_path: str = None,<br></ul>
        <li>Docs:<br>    """
<br>
        python code_parser.py  export_stats_perfile <in_path> <out_path>
<br>

<br>
    Returns:
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:export_stats_perrepo' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L860'>export_stats_perrepo</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>in_path: str = None,<br>out_path: str = None,<br></ul>
        <li>Docs:<br>    """ 
<br>
        python code_parser.py  export_stats_perfile <in_path> <out_path>
<br>

<br>
    Returns:
<br>
        1  repo   --->  a single file stats for all sub-diractory
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/code_parser.py:test_example' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/code_parser.py#L896'>test_example</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/text.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/text.py'>./utilmy/text.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/text.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/text.py#L17'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/text.py:pd_text_hash_create_lsh' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/text.py#L24'>pd_text_hash_create_lsh</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col,<br>sep = " ",<br>threshold = 0.7,<br>num_perm = 10,<br></ul>
        <li>Docs:<br>    '''
<br>
    For each of the entry create a hash function
<br>
    '''
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/text.py:pd_text_getcluster' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/text.py#L53'>pd_text_getcluster</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col,<br>threshold,<br>num_perm,<br></ul>
        <li>Docs:<br>    '''
<br>
    For each of the hash function find a cluster and assign unique id to the dataframe cluster_id
<br>
    '''
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/text.py:pd_similarity' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/text.py#L81'>pd_similarity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df:  pd.DataFrame,<br>cols = [],<br>algo = '',<br></ul>
        <li>Docs:<br>    '''
<br>
        Return similarities between two columns with 
<br>
        python's SequenceMatcher algorithm
<br>

<br>
        Args:
<br>
            df (pd.DataFrame): Pandas Dataframe.
<br>
            algo (String)    : rapidfuzz | editdistance 
<br>
            cols (list[str]) : List of of columns name (2 columns)
<br>

<br>
        Returns:
<br>
            pd.DataFrame
<br>

<br>
    '''
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/text.py:test_lsh' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/text.py#L120'>test_lsh</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/io.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/io.py'>./utilmy/io.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/io.py:screenshot' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/io.py#L4'>screenshot</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>output = 'fullscreen.png',<br>monitors = -1,<br></ul>
        <li>Docs:<br>  """
<br>
  with mss() as sct:
<br>
    for _ in range(100):
<br>
        sct.shot()
<br>
  # MacOS X
<br>
  from mss.darwin import MSS as mss
<br>
  
<br>
  
<br>
  """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/utilmy.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py'>./utilmy/utilmy.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L6'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_merge' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L11'>pd_merge</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df1,<br>df2,<br>on = None,<br>colkeep = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_plot_multi' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L17'>pd_plot_multi</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>plot_type = None,<br>cols_axe1: list = [],<br>cols_axe2: list = [],<br>figsize = (8,<br>4,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_filter' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L66'>pd_filter</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>filter_dict = "shop_id=11, l1_genre_id>600, l2_genre_id<80311,",<br>verbose = False,<br></ul>
        <li>Docs:<br>    """
<br>
     dfi = pd_filter2(dfa, "shop_id=11, l1_genre_id>600, l2_genre_id<80311," )
<br>
     dfi2 = pd_filter(dfa, {"shop_id" : 11} )
<br>
     ### Dilter dataframe with basic expr
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_to_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L104'>pd_to_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>filei,<br>check = "check",<br>verbose = True,<br>**kw,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_read_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L130'>pd_read_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path_glob = "*.pkl",<br>ignore_index = True,<br>cols = None,<br>verbose = False,<br>nrows = -1,<br>concat_sort = True,<br>n_pool = 1,<br>drop_duplicates = None,<br>col_filter = None,<br>col_filter_val = None,<br>dtype_reduce = None,<br>**kw,<br></ul>
        <li>Docs:<br>  """  Read file in parallel from disk : very Fast
<br>
  :param path_glob: list of pattern, or sep by ";"
<br>
  :return:
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_sample_strat' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L213'>pd_sample_strat</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col,<br>n,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_cartesian' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L221'>pd_cartesian</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df1,<br>df2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_plot_histogram' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L233'>pd_plot_histogram</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dfi,<br>path_save = None,<br>nbin = 20.0,<br>q5 = 0.005,<br>q95 = 0.995,<br>nsample =  -1,<br>show = False,<br>clear = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_col_bins' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L258'>pd_col_bins</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col,<br>nbins = 5,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_dtype_reduce' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L265'>pd_dtype_reduce</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dfm,<br>int0  = 'int32',<br>float0  =  'float32',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_dtype_count_unique' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L274'>pd_dtype_count_unique</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col_continuous = [],<br></ul>
        <li>Docs:<br>    """Learns the number of categories in each variable and standardizes the data.
<br>
        ----------
<br>
        data: pd.DataFrame
<br>
        continuous_ids: list of ints
<br>
            List containing the indices of known continuous variables. Useful for
<br>
            discrete data like age, which is better modeled as continuous.
<br>
        Returns
<br>
        -------
<br>
        ncat:  number of categories of each variable. -1 if the variable is  continuous.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_dtype_to_category' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L313'>pd_dtype_to_category</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col_exclude,<br>treshold = 0.5,<br></ul>
        <li>Docs:<br>  """
<br>
    Convert string to category
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_dtype_getcontinuous' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L332'>pd_dtype_getcontinuous</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>cols_exclude: list = [],<br>nsample = -1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_del' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L348'>pd_del</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>cols: list,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_add_noise' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L357'>pd_add_noise</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>level = 0.05,<br>cols_exclude: list = [],<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_cols_unique_count' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L371'>pd_cols_unique_count</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>cols_exclude: list = [],<br>nsample = -1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:pd_show' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L389'>pd_show</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>nrows = 100,<br>reader = 'notepad.exe',<br>**kw,<br></ul>
        <li>Docs:<br>    """ Show from Dataframe
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:to_dict' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L409'>to_dict</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>**kw,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:to_timeunix' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L414'>to_timeunix</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>datex = "2018-01-16",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:to_datetime' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L422'>to_datetime</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:np_list_intersection' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L427'>np_list_intersection</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>l1,<br>l2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:np_add_remove' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L431'>np_add_remove</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>set_,<br>to_remove,<br>to_add,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:to_float' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L440'>to_float</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:to_int' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L447'>to_int</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:config_load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L458'>config_load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config_path: str  =  None,<br>path_default: str = None,<br>config_default: dict = None,<br></ul>
        <li>Docs:<br>    """Load Config file into a dict  from .json or .yaml file
<br>
    TODO .cfg file
<br>
    1) load config_path
<br>
    2) If not, load default from HOME USER
<br>
    3) If not, create default on in python code
<br>
    Args:
<br>
        config_path: path of config or 'default' tag value
<br>
    Returns: dict config
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_path_split' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L520'>os_path_split</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fpath: str = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_file_replacestring' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L537'>os_file_replacestring</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>findstr,<br>replacestr,<br>some_dir,<br>pattern = "*.*",<br>dirlevel = 1,<br></ul>
        <li>Docs:<br>    """ #fil_replacestring_files("logo.png", "logonew.png", r"D:/__Alpaca__details/aiportfolio",
<br>
        pattern="*.html", dirlevel=5  )
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_walk' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L559'>os_walk</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dir1,<br>pattern = "*.*",<br>dirlevel = 20,<br>path_only = False,<br></ul>
        <li>Docs:<br>    """
<br>
            matches["dirpath"]  = []
<br>
            matches["filename"] = []
<br>
            matches["fullpath"] = []
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:z_os_search_fast' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L603'>z_os_search_fast</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fname,<br>texts = None,<br>mode = "regex/str",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_search_content' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L647'>os_search_content</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>srch_pattern = None,<br>mode = "str",<br>dir1 = "",<br>file_pattern = "*.*",<br>dirlevel = 1,<br></ul>
        <li>Docs:<br>    """  search inside the files
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_get_function_name' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L663'>os_get_function_name</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_variable_init' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L676'>os_variable_init</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ll,<br>globs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_import' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L684'>os_import</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>mod_name = "myfile.config.model",<br>globs = None,<br>verbose = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_variable_exist' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L711'>os_variable_exist</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>globs,<br>msg = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_variable_check' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L721'>os_variable_check</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ll,<br>globs = None,<br>do_terminate = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_clean_memory' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L733'>os_clean_memory</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>varlist,<br>globx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_system_list' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L741'>os_system_list</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ll,<br>logfile = None,<br>sleep_sec = 10,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_file_check' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L763'>os_file_check</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fp,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_to_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L771'>os_to_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>txt = "",<br>filename = "ztmp.txt",<br>mode = 'a',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_platform_os' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L776'>os_platform_os</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_cpu' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L781'>os_cpu</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_platform_ip' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L786'>os_platform_ip</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_memory' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L791'>os_memory</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """ Get node total memory and memory usage in linux
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_sleep_cpu' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L808'>os_sleep_cpu</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>priority = 300,<br>cpu_min = 50,<br>sleep = 10,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_ram_object' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L822'>os_ram_object</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>o,<br>ids,<br>hint = " deep_getsizeof(df_pd,<br>set(,<br></ul>
        <li>Docs:<br>    """ deep_getsizeof(df_pd, set())
<br>
    Find the memory footprint of a Python object
<br>
    The sys.getsizeof function does a shallow size of only. It counts each
<br>
    object inside a container as pointer only regardless of how big it
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_copy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L853'>os_copy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>src,<br>dst,<br>overwrite = False,<br>exclude = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_removedirs' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L865'>os_removedirs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br>    """  issues with no empty Folder
<br>
    # Delete everything reachable from the directory named in 'top',
<br>
    # assuming there are no symbolic links.
<br>
    # CAUTION:  This is dangerous!  For example, if top == '/', it could delete all your disk files.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_getcwd' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L892'>os_getcwd</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_system' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L898'>os_system</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ll,<br>logfile = None,<br>sleep_sec = 10,<br></ul>
        <li>Docs:<br>  """ get values
<br>
       os_system( f"   ztmp ",  doprint=True)
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:os_makedirs' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L915'>os_makedirs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dir_or_file,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:global_verbosity' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L924'>global_verbosity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cur_path,<br>path_relative = "/../../config.json",<br>default = 5,<br>key = 'verbosity',<br>,<br></ul>
        <li>Docs:<br>    """ Get global verbosity
<br>
    verbosity = global_verbosity(__file__, "/../../config.json", default=5 )
<br>

<br>
    verbosity = global_verbosity("repo_root", "config/config.json", default=5 )
<br>

<br>
    :param cur_path:
<br>
    :param path_relative:
<br>
    :param key:
<br>
    :param default:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:git_repo_root' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L962'>git_repo_root</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:git_current_hash' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L972'>git_current_hash</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>mode = 'full',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:plot_to_html' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L984'>plot_to_html</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dir_input = "*.png",<br>out_file = "graph.html",<br>title = "",<br>verbose = False,<br></ul>
        <li>Docs:<br>    """
<br>
      plot_to_html( model_path + "/graph_shop_17_past/*.png" , model_path + "shop_17.html" )
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:save' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1083'>save</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dd,<br>to_file = "",<br>verbose = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1090'>load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>to_file = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:print_everywhere' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1100'>print_everywhere</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>
    https://github.com/alexmojaki/snoop
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:log5' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1130'>log5</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br>    """    ### Equivalent of print, but more :  https://github.com/gruns/icecream
<br>
    pip install icrecream
<br>
    ic()  --->  ic| example.py:4 in foo()
<br>
    ic(var)  -->   ic| d['key'][1]: 'one'
<br>
    
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:log_trace' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1141'>log_trace</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>msg = "",<br>dump_path = "",<br>globs = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:profiler_start' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1147'>profiler_start</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/utilmy.py:profiler_stop' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1155'>profiler_stop</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./utilmy/utilmy.py:dict_to_namespace' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L403'>dict_to_namespace</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./utilmy/utilmy.py:Session' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1014'>Session</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:dict_to_namespace:__init__' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L405'>dict_to_namespace:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:Session:__init__' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1020'>Session:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dir_session = "ztmp/session/",<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:Session:show' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1026'>Session:show</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:Session:save' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1031'>Session:save</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>name,<br>glob = None,<br>tag = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:Session:load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1037'>Session:load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>name,<br>glob: dict = None,<br>tag = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:Session:save_session' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1044'>Session:save_session</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>folder,<br>globs,<br>tag = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/utilmy.py:Session:load_session' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/utilmy.py#L1067'>Session:load_session</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>folder,<br>globs = None,<br></ul>
        <li>Docs:<br>      """
<br>
      """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/docs/generate_doc.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/generate_doc.py'>./utilmy/docs/generate_doc.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/docs/generate_doc.py:markdown_create_function' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/generate_doc.py#L22'>markdown_create_function</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>uri,<br>name,<br>type,<br>args_name,<br>args_type,<br>args_value,<br>start_line,<br>list_docs,<br>prefix = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/docs/generate_doc.py:markdown_create_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/generate_doc.py#L55'>markdown_create_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>list_info,<br>prefix = '',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/decorators.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py'>./utilmy/decorators.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:timeout' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L22'>timeout</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>seconds = 10,<br>error_message = os.strerror(errno.ETIME,<br></ul>
        <li>Docs:<br>    """Decorator to throw timeout error, if function doesnt complete in certain time
<br>
    Args:
<br>
        seconds:``int``
<br>
            No of seconds to wait
<br>
        error_message:``str``
<br>
            Error message
<br>
            
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:timer' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L49'>timer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>func,<br></ul>
        <li>Docs:<br>    """
<br>
    Decorator to show the execution time of a function or a method in a class.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:profiler_context' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L67'>profiler_context</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>
    Context Manager the will profile code inside it's bloc.
<br>
    And print the result of profiler.
<br>
    Example:
<br>
        with profiler_context():
<br>
            # code to profile here
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:profiler_deco' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L87'>profiler_deco</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>func,<br></ul>
        <li>Docs:<br>    """
<br>
    A decorator that will profile a function
<br>
    And print the result of profiler.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:profiler_decorator_base' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L105'>profiler_decorator_base</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fnc,<br></ul>
        <li>Docs:<br>    """
<br>
    A decorator that uses cProfile to profile a function
<br>
    And print the result
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:os_multithread' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L127'>os_multithread</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>**kwargs,<br></ul>
        <li>Docs:<br>    """
<br>
    Creating n number of threads:  1 thread per function,
<br>
    starting them and waiting for their subsequent completion
<br>
   Example:
<br>
        os_multithread(function1=(function_name1, (arg1, arg2, ...)),
<br>
                       ...)
<br>

<br>
    def test_print(*args):
<br>
        print(*args)
<br>

<br>
    os_multithread(function1=(test_print, ("some text",)),
<br>
                          function2=(test_print, ("bbbbb",)),
<br>
                          function3=(test_print, ("ccccc",)))
<br>

<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/decorators.py:threading_deco' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L166'>threading_deco</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>func,<br></ul>
        <li>Docs:<br>    """ A decorator to run function in background on thread
<br>
	Args:
<br>
		func:``function``
<br>
			Function with args
<br>

<br>
	Return:
<br>
		background_thread: ``Thread``
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./utilmy/decorators.py:_TimeoutError' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/decorators.py#L11'>_TimeoutError</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/templates/templist/pypi_package/run_pipy.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py'>./utilmy/templates/templist/pypi_package/run_pipy.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:get_current_githash' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L39'>get_current_githash</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:update_version' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L46'>update_version</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br>n = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:git_commit' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L65'>git_commit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>message,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:ask' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L77'>ask</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>question,<br>ans = 'yes',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:pypi_upload' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L81'>pypi_upload</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>
      It requires credential in .pypirc  files
<br>
      __token__
<br>
      or in github SECRETS
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:main' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L102'>main</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*args,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L10'>Version</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version:__init__' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L13'>Version:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>major,<br>minor,<br>patch,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version:__str__' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L18'>Version:__str__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version:__repr__' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L21'>Version:__repr__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version:stringify' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L24'>Version:stringify</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version:new_version' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L27'>Version:new_version</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>orig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/templates/templist/pypi_package/run_pipy.py:Version:parse' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/run_pipy.py#L31'>Version:parse</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cls,<br>string,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/logs/util_log.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py'>./utilmy/logs/util_log.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:logger_setup' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L34'>logger_setup</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>log_config_path:  str  =  None,<br>log_template:  str  =  "default",<br>**kwargs,<br></ul>
        <li>Docs:<br>    """ Generic Logging setup
<br>
      Overide logging using loguru setup
<br>
      1) Custom config from log_config_path .yaml file
<br>
      2) Use shortname log, log2, logw, loge for logging output
<br>

<br>
    Args:
<br>
        log_config_path:
<br>
        template_name:
<br>
        **kwargs:
<br>
    Returns:None
<br>

<br>
    TODO:
<br>

<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L115'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>log_config_path:  str  =  None,<br>log_template:  str  =  "default",<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:log2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L119'>log2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:log3' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L123'>log3</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s):   ### Debuggine level 2depth = 1, lazy=True).log("DEBUG_2", ",".join([str(t) for t in s]))*s):,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:logw' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L128'>logw</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:logc' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L132'>logc</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:loge' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L136'>loge</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:logr' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L140'>logr</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:test' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L145'>test</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:z_logger_stdout_override' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L162'>z_logger_stdout_override</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """ Redirect stdout --> logger
<br>
    Returns:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/util_log.py:z_logger_custom_1' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/util_log.py#L187'>z_logger_custom_1</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./test.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py'>./test.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./test.py:pd_random' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L8'>pd_random</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ncols = 7,<br>nrows = 100,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_utilmy_plot' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L19'>test_utilmy_plot</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_utilmy_pd_os_session' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L25'>test_utilmy_pd_os_session</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_utilmy_session' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L118'>test_utilmy_session</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_docs_cli' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L142'>test_docs_cli</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>
      from utilmy.docs.generate_doc import run_markdown, run_table
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_decorators_os' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L155'>test_decorators_os</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*args,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:pd_generate_data' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L209'>pd_generate_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ncols = 7,<br>nrows = 100,<br></ul>
        <li>Docs:<br>    """
<br>
    Generate sample data for function testing
<br>
    categorical features for anova test
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_tabular_test' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L227'>test_tabular_test</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>        """
<br>
        ANOVA test
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_text_similarity' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L242'>test_text_similarity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./test.py:test_text_pdcluster' href='https://github.com/arita37/myutil/blob/main/utilmy//./test.py#L262'>test_text_pdcluster</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./run_pipy.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py'>./run_pipy.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./run_pipy.py:get_current_githash' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L62'>get_current_githash</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./run_pipy.py:update_version' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L69'>update_version</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br>n = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./run_pipy.py:git_commit' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L90'>git_commit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>message,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./run_pipy.py:ask' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L101'>ask</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>question,<br>ans = 'yes',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./run_pipy.py:pypi_upload' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L105'>pypi_upload</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>
      It requires credential in .pypirc  files
<br>
      __token__
<br>
      or in github SECRETS
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./run_pipy.py:main' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L125'>main</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*args,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./run_pipy.py:Version' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L33'>Version</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./run_pipy.py:Version:__init__' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L36'>Version:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>major,<br>minor,<br>patch,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./run_pipy.py:Version:__str__' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L41'>Version:__str__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./run_pipy.py:Version:__repr__' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L44'>Version:__repr__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./run_pipy.py:Version:stringify' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L47'>Version:stringify</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./run_pipy.py:Version:new_version' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L50'>Version:new_version</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>orig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./run_pipy.py:Version:parse' href='https://github.com/arita37/myutil/blob/main/utilmy//./run_pipy.py#L54'>Version:parse</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cls,<br>string,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/docs/cli.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/cli.py'>./utilmy/docs/cli.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/docs/cli.py:run_cli' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/docs/cli.py#L8'>run_cli</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """ USage
<br>
    
<br>
    doc-gen  doc-gen  --repo_dir utilmy/      --doc_dir docs/"
<br>
    """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./setup.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./setup.py'>./setup.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./setup.py:get_current_githash' href='https://github.com/arita37/myutil/blob/main/utilmy//./setup.py#L33'>get_current_githash</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/templates/templist/pypi_package/setup.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/setup.py'>./utilmy/templates/templist/pypi_package/setup.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/templates/templist/pypi_package/setup.py:get_current_githash' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/templates/templist/pypi_package/setup.py#L17'>get_current_githash</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/zutil.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py'>./utilmy/zutil.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:session_load_function' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L66'>session_load_function</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name = "test_20160815",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:session_save_function' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L75'>session_save_function</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name = "test",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:py_save_obj_dill' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L83'>py_save_obj_dill</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>obj1,<br>keyname = "",<br>otherfolder = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:aa_unicode_ascii_utf8_issue' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L111'>aa_unicode_ascii_utf8_issue</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """Take All csv in a folder and provide Table, Column Schema, type
<br>

<br>
 METHOD FOR Unicode / ASCII issue
<br>
1. Decode early
<br>
Decode to <type 'unicode'> ASAP
<br>
df['PREF_NAME']=       df['PREF_NAME'].apply(to_unicode)
<br>

<br>
2. Unicode everywhere
<br>

<br>

<br>
3. Encode late
<br>
# >>> f = open('/tmp/ivan_out.txt','w')
<br>
# >>> f.write(ivan_uni.encode('utf-8'))
<br>

<br>
Important methods
<br>
s.decode(encoding)  <type 'str'> to <type 'unicode'>
<br>
u.encode(encoding)  <type 'unicode'> to <type 'str'>
<br>

<br>
http://farmdev.com/talks/unicode/
<br>

<br>
   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:isfloat' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L135'>isfloat</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:isint' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L145'>isint</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:isanaconda' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L149'>isanaconda</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:a_run_ipython' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L157'>a_run_ipython</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cmd1,<br></ul>
        <li>Docs:<br>    """ Execute Ipython Command in python code
<br>
     run -i :  run including current interprete variable
<br>
 """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:py_autoreload' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L164'>py_autoreload</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_platform' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L169'>os_platform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:a_start_log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L173'>a_start_log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>id1 = "",<br>folder = "aaserialize/log/",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:a_cleanmemory' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L187'>a_cleanmemory</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:a_info_conda_jupyter' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L193'>a_info_conda_jupyter</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:print_object_tofile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L355'>print_object_tofile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>vv,<br>txt,<br>file1="d:  = "d:/regression_output.py",<br></ul>
        <li>Docs:<br>    """ #Print to file Object   Table   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:print_progressbar' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L372'>print_progressbar</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>iteration,<br>total,<br>prefix = "",<br>suffix = "",<br>decimals = 1,<br>bar_length = 100,<br></ul>
        <li>Docs:<br>    """# Print iterations progress
<br>
     Call in a loop to create terminal progress bar
<br>
    @params:
<br>
        iteration   - Required  : current iteration (Int)
<br>
        total       - Required  : total iterations (Int)
<br>
        prefix      - Optional  : prefix string (Str)
<br>
        suffix      - Optional  : suffix string (Str)
<br>
        decimals    - Optional  : positive number of decimals in percent complete (Int)
<br>
        bar_length   - Optional  : character length of bar (Int)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_zip_checkintegrity' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L401'>os_zip_checkintegrity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filezip1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_zipfile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L414'>os_zipfile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>folderin,<br>folderzipname,<br>iscompress = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_zipfolder' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L432'>os_zipfolder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dir_tozip = "/zdisks3/output",<br>zipname = "/zdisk3/output.zip",<br>dir_prefix = True,<br>iscompress=Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[:  = Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[:-1]if dir_prefix:,<br></ul>
        <li>Docs:<br>    """
<br>
 shutil.make_archive('/zdisks3/results/output', 'zip',
<br>
                     root_dir=/zdisks3/results/',
<br>
                     base_dir='output')
<br>
 os_zipfolder('zdisk/test/aapackage', 'zdisk/test/aapackage.zip', 'zdisk/test')"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_zipextractall' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L484'>os_zipextractall</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filezip_or_dir = "folder1/*.zip",<br>tofolderextract = "zdisk/test",<br>isprint = 1,<br></ul>
        <li>Docs:<br>    """os_zipextractall( 'aapackage.zip','zdisk/test/'      )  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_folder_copy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L518'>os_folder_copy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>src,<br>dst,<br>symlinks = False,<br>pattern1 = "*.py",<br>fun_file_toignore = None,<br></ul>
        <li>Docs:<br>    """
<br>
       callable(src, names) -> ignored_names
<br>
       'src' parameter, which is the directory being visited by copytree(), and
<br>
       'names' which is the list of `src` contents, as returned by os.listdir():
<br>

<br>
    Since copytree() is called recursively, the callable will be called once for each
<br>
    directory that is copied.
<br>
    It returns a  list of names relative to the `src` directory that should not be copied.
<br>
   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_folder_create' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L548'>os_folder_create</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>directory,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_folder_robocopy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L555'>os_folder_robocopy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>from_folder = "",<br>to_folder = "",<br>my_log="H:  = "H:/robocopy_log.txt",<br></ul>
        <li>Docs:<br>    """
<br>
    Copy files to working directory
<br>
    robocopy <Source> <Destination> [<File>[ ...]] [<Options>]
<br>
    We want to copy the files to a fast SSD drive
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_replace' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L568'>os_file_replace</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>source_file_path,<br>pattern,<br>substring,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_replacestring1' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L582'>os_file_replacestring1</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>find_str,<br>rep_str,<br>file_path,<br></ul>
        <li>Docs:<br>    """replaces all find_str by rep_str in file file_path"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_replacestring2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L594'>os_file_replacestring2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>findstr,<br>replacestr,<br>some_dir,<br>pattern = "*.*",<br>dirlevel = 1,<br></ul>
        <li>Docs:<br>    """ #fil_replacestring_files("logo.png", "logonew.png", r"D:/__Alpaca__details/aiportfolio",
<br>
    pattern="*.html", dirlevel=5  )
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_getname' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L604'>os_file_getname</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_getpath' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L611'>os_file_getpath</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_gettext' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L618'>os_file_gettext</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_listall' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L625'>os_file_listall</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dir1,<br>pattern = "*.*",<br>dirlevel = 1,<br>onlyfolder = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:_os_file_search_fast' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L695'>_os_file_search_fast</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fname,<br>texts = None,<br>mode = "regex/str",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_search_content' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L740'>os_file_search_content</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>srch_pattern = None,<br>mode = "str",<br>dir1 = "",<br>file_pattern = "*.*",<br>dirlevel = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_rename' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L758'>os_file_rename</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>some_dir,<br>pattern = "*.*",<br>pattern2 = "",<br>dirlevel = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_gui_popup_show' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L788'>os_gui_popup_show</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>txt,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_print_tofile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L808'>os_print_tofile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>vv,<br>file1,<br>mode1="a"):  = "a"):  # print into a file='afile1,<br>mode1) as text_file: ,<br></ul>
        <li>Docs:<br>    """
<br>
    Here is a list of the different modes of opening a file:
<br>
r
<br>
Opens a file for reading only. The file pointer is placed at the beginning of the file.
<br>
This is the default mode.
<br>

<br>
rb
<br>

<br>
Opens a file for reading only in binary format. The file pointer is placed at the
<br>
beginning of the file. This is the default mode.
<br>
r+
<br>

<br>
Opens a file for both reading and writing. The file pointer will be at the beginning of the file.
<br>
rb+
<br>

<br>
Opens a file for both reading and writing in binary format. The file pointer will be at the
<br>
beginning of the file.
<br>
w
<br>

<br>
Opens a file for writing only. Overwrites the file if the file exists. If the file does not exist,
<br>
creates a new file for writing.
<br>
wb
<br>

<br>
Opens a file for writing only in binary format. Overwrites the file if the file exists.
<br>
If the file does not exist, creates a new file for writing.
<br>
w+
<br>

<br>
Opens a file for both writing and reading. Overwrites the existing file if the file exists.
<br>
If the file does not exist, creates a new file for reading and writing.
<br>
wb+
<br>

<br>
Opens a file for both writing and reading in binary format. Overwrites the existing file if
<br>
the file exists. If the file does not exist, creates a new file for reading and writing.
<br>
a
<br>

<br>
Opens a file for appending. The file pointer is at the end of the file if the file exists. That is,
<br>
the file is in the append mode. If the file does not exist, it creates a new file for writing.
<br>
ab
<br>

<br>
Opens a file for appending in binary format. The file pointer is at the end of the file if the file
<br>
exists. That is, the file is in the append mode. If the file does not exist, it creates a new file
<br>
for writing.
<br>
a+
<br>

<br>
Opens a file for both appending and reading. The file pointer is at the end of the file if the
<br>
file exists. The file opens in the append mode. If the file does not exist, it creates a new file
<br>
for reading and writing.
<br>
ab+
<br>

<br>
Opens a file for both appending and reading in binary format. The file pointer is at the end of
<br>
the file if the file exists. The file opens in the append mode. If the file does not exist,
<br>
it creates a new file for reading and writing.
<br>
To open a text file, use:
<br>
fh = open("hello.txt", "r")
<br>

<br>
To read a text file, use:
<br>
print fh.read()
<br>

<br>
To read one line at a time, use:
<br>
print fh.readline()
<br>

<br>
To read a list of lines use:
<br>
print fh.readlines()
<br>

<br>
To write to a file, use:
<br>
fh = open("hello.txt", "w")
<br>
lines_of_text = ["a line of text", "another line of text", "a third line"]
<br>
fh.writelines(lines_of_text)
<br>
fh.close()
<br>

<br>
To append to file, use:
<br>
fh = open("Hello.txt", "a")
<br>
fh.close()
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_path_norm' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L888'>os_path_norm</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>pth):   # Normalize path for Python directoryr""" #r"D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples" """)  = = 2:,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_path_change' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L902'>os_path_change</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_path_current' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L907'>os_path_current</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_exist' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L911'>os_file_exist</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_size' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L915'>os_file_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_read' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L919'>os_file_read</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_isame' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L924'>os_file_isame</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br>file2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_get_extension' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L930'>os_file_get_extension</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br></ul>
        <li>Docs:<br>    """
<br>
    # >>> get_file_extension("/a/b/c")
<br>
    ''
<br>
    # >>> get_file_extension("/a/b/c.tar.xz")
<br>
    'xz'
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_normpath' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L944'>os_file_normpath</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br>    """Normalize path.
<br>
    - eliminating double slashes, etc. (os.path.normpath)
<br>
    - ensure paths contain ~[user]/ expanded.
<br>

<br>
    :param path: Path string :: str
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_folder_is_path' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L954'>os_folder_is_path</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path_or_stream,<br></ul>
        <li>Docs:<br>    """
<br>
    Is given object `path_or_stream` a file path?
<br>
    :param path_or_stream: file path or stream, file/file-like object
<br>
    :return: True if `path_or_stream` is a file path
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_get_path_from_stream' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L963'>os_file_get_path_from_stream</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>maybe_stream,<br></ul>
        <li>Docs:<br>    """
<br>
    Try to get file path from given stream `stream`.
<br>

<br>
    :param maybe_stream: A file or file-like object
<br>
    :return: Path of given file or file-like object or None
<br>

<br>
    # >>> __file__ == get_path_from_stream(__file__)
<br>
    True
<br>
    # >>> __file__ == get_path_from_stream(open(__file__, 'r'))
<br>
    True
<br>
    # >>> strm = anyconfig.compat.StringIO()
<br>
    # >>> get_path_from_stream(strm) is None
<br>
    True
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_try_to_get_extension' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L988'>os_file_try_to_get_extension</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path_or_strm,<br></ul>
        <li>Docs:<br>    """
<br>
    Try to get file extension from given path or file object.
<br>
    :return: File extension or None
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_are_same_file_types' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1000'>os_file_are_same_file_types</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>paths,<br></ul>
        <li>Docs:<br>    """
<br>
    Are given (maybe) file paths same type (extension) ?
<br>
    :param paths: A list of file path or file(-like) objects
<br>

<br>
    # >>> are_same_file_types([])
<br>
    False
<br>
    # >>> are_same_file_types(["a.conf"])
<br>
    True
<br>
    # >>> are_same_file_types(["a.yml", "b.json"])
<br>
    False
<br>
    # >>> strm = anyconfig.compat.StringIO()
<br>
    # >>> are_same_file_types(["a.yml", "b.yml", strm])
<br>
    False
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_norm_paths' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1024'>os_file_norm_paths</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>paths,<br>marker = "*",<br></ul>
        <li>Docs:<br>    """
<br>
    :param paths:
<br>
        A glob path pattern string, or a list consists of path strings or glob
<br>
        path pattern strings or file objects
<br>
    :param marker: Glob marker character or string, e.g. '*'
<br>
    :return: List of path strings
<br>
    # >>> norm_paths([])
<br>
    []
<br>
    # >>> norm_paths("/usr/lib/a/b.conf /etc/a/b.conf /run/a/b.conf".split())
<br>
    ['/usr/lib/a/b.conf', '/etc/a/b.conf', '/run/a/b.conf']
<br>
    # >>> paths_s = os.path.join(os.path.dirname(__file__), "u*.py")
<br>
    # >>> ref = sglob(paths_s)
<br>
    # >>> ref = ["/etc/a.conf"] + ref
<br>
    # >>> assert norm_paths(["/etc/a.conf", paths_s]) == ref
<br>
    # >>> strm = anyconfig.compat.StringIO()
<br>
    # >>> assert norm_paths(["/etc/a.conf", strm]) == ["/etc/a.conf", strm]
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_mergeall' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1068'>os_file_mergeall</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>nfile,<br>dir1,<br>pattern1,<br>deepness = 2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_file_extracttext' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1077'>os_file_extracttext</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>output_file,<br>dir1,<br>pattern1 = "*.html",<br>htmltag = "p",<br>deepness = 2,<br></ul>
        <li>Docs:<br>    """ Extract text from html """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_path_append' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1093'>os_path_append</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>p1,<br>p2 = None,<br>p3 = None,<br>p4 = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_wait_cpu' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1104'>os_wait_cpu</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>priority = 300,<br>cpu_min = 50,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_split_dir_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1116'>os_split_dir_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dirfile,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_process_run' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1126'>os_process_run</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cmd_list,<br>capture_output = False,<br></ul>
        <li>Docs:<br>    """os_process_run
<br>
    
<br>
    Args:
<br>
         cmd_list: list ["program", "arg1", "arg2"]
<br>
         capture_output: bool
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_process_2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1149'>os_process_2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:py_importfromfile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1183'>py_importfromfile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>modulename,<br>dir1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:py_memorysize' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1199'>py_memorysize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>o,<br>ids,<br>hint = " deep_getsizeof(df_pd,<br>set(,<br></ul>
        <li>Docs:<br>    """ deep_getsizeof(df_pd, set())
<br>
    Find the memory footprint of a Python object
<br>
    The sys.getsizeof function does a shallow size of only. It counts each
<br>
    object inside a container as pointer only regardless of how big it
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:save' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1229'>save</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>obj,<br>folder = "/folder1/keyname",<br>isabsolutpath = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1233'>load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>folder = "/folder1/keyname",<br>isabsolutpath = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:save_test' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1237'>save_test</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>folder = "/folder1/keyname",<br>isabsolutpath = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:py_save_obj' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1244'>py_save_obj</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>obj1,<br>keyname = "",<br>otherfolder = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:py_load_obj' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1257'>py_load_obj</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>folder = "/folder1/keyname",<br>isabsolutpath = 0,<br>encoding1 = "utf-8",<br></ul>
        <li>Docs:<br>    """def load_obj(name, encoding1='utf-8' ):
<br>
         with open('D:/_devs/Python01/aaserialize/' + name + '.pkl', 'rb') as f:
<br>
            return pickle.load(f, encoding=encoding1)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:z_key_splitinto_dir_name' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1277'>z_key_splitinto_dir_name</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>keyname,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_config_setfile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1287'>os_config_setfile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dict_params,<br>outfile,<br>mode1 = "w+",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_config_getfile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1299'>os_config_getfile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:os_csv_process' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1307'>os_csv_process</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_toexcel' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1427'>pd_toexcel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>outfile = "file.xlsx",<br>sheet_name = "sheet1",<br>append = 1,<br>returnfile = 1,<br></ul>
        <li>Docs:<br>    """
<br>
# Create a Pandas Excel writer using XlsxWriter as the engine.
<br>
writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
<br>
df.to_excel(writer, sheet_name='Sheet1')
<br>
writer.save()
<br>

<br>
# Get the xlsxwriter objects from the dataframe writer object.
<br>
workbook  = writer.book
<br>
worksheet = writer.sheets['Sheet1']
<br>

<br>
# Add some cell formats.
<br>
format1 = workbook.add_format({'num_format': '#,##0.00'})
<br>
format2 = workbook.add_format({'num_format': '0%'})
<br>
format3 = workbook.add_format({'num_format': 'h:mm:ss AM/PM'})
<br>

<br>
# Set the column width and format.
<br>
worksheet.set_column('B:B', 18, format1)
<br>

<br>
# Set the format but not the column width.
<br>
worksheet.set_column('C:C', None, format2)
<br>

<br>
worksheet.set_column('D:D', 16, format3)
<br>

<br>
# Close the Pandas Excel writer and output the Excel file.
<br>
writer.save()
<br>

<br>
from openpyxl import load_workbook
<br>
wb = load_workbook(outfile)
<br>
ws = wb.active
<br>
ws.title = 'Table 1'
<br>

<br>
tableshape = np.shape(table)
<br>
alph = list(string.ascii_uppercase)
<br>

<br>
for i in range(tableshape[0]):
<br>
    for j in range(tableshape[1]):
<br>
        ws[alph[i]+str(j+1)] = table[i, j]
<br>

<br>
for cell in ws['A'] + ws[1]:
<br>
    cell.style = 'Pandas'
<br>

<br>
wb.save('Scores.xlsx')
<br>

<br>
   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_toexcel_many' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1492'>pd_toexcel_many</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>outfile = "file1.xlsx",<br>df1 = None,<br>df2 = None,<br>df3 = None,<br>df4 = None,<br>df5 = None,<br>df6 = Nonedf1,<br>outfile,<br>sheet_name="df1")if df2 is not None:  = "df1")if df2 is not None:,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:find_fuzzy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1509'>find_fuzzy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>xstring,<br>list_string,<br></ul>
        <li>Docs:<br>    """ if xstring matches partially, add to the list   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_match_fuzzy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1514'>str_match_fuzzy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>xstring,<br>list_string,<br></ul>
        <li>Docs:<br>    """ if any of list_strinf elt matches partially xstring """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_parse_stringcalendar' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1522'>str_parse_stringcalendar</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cal,<br></ul>
        <li>Docs:<br>    """----------Parse Calendar  --------"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_make_unicode' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1539'>str_make_unicode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input_str,<br>errors = "replace",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_empty_string_array' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1548'>str_empty_string_array</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>y = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_empty_string_array_numpy' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1555'>str_empty_string_array_numpy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>nx,<br>ny = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_isfloat' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1561'>str_isfloat</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>value,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_is_azchar' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1569'>str_is_azchar</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_is_az09char' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1577'>str_is_az09char</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_reindent' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1585'>str_reindent</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>s,<br>num_spaces):   # change indentation of multine string"\n")num_spaces * " ") + line.lstrip() for line in s]s)return sdelimiters,<br>string,<br>maxsplit=0):  = 0):  # Split into Sub-Sentenceimport remap(re.escape,<br>delimiters))regex_pattern,<br>string,<br>maxsplit)sep2,<br>ll,<br>maxsplit=0):  = 0):  # Find Sentence Patternimport re_ = maxsplitsep2)"(" + regex_pat + r")|(?:(?!" + regex_pat + ").)*",<br>re.S)lambda m:  m.group(1) if m.group(1) else "P",<br>ll)return llx): ,<br></ul>
        <li>Docs:<br>    """
<br>
   if args:
<br>
       aux= name1+'.'+obj.__name__ +'('+ str(args) +')  \n' + str(inspect.getdoc(obj))
<br>
       aux= aux.replace('\n', '\n       ')
<br>
       aux= aux.rstrip()
<br>
       aux= aux + ' \n'
<br>
       wi( aux)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_split2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1600'>str_split2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>delimiters,<br>string,<br>maxsplit=0):  = 0):  # Split into Sub-Sentenceimport remap(re.escape,<br>delimiters))regex_pattern,<br>string,<br>maxsplit)sep2,<br>ll,<br>maxsplit=0):  = 0):  # Find Sentence Patternimport re_ = maxsplitsep2)"(" + regex_pat + r")|(?:(?!" + regex_pat + ").)*",<br>re.S)lambda m:  m.group(1) if m.group(1) else "P",<br>ll)return llx): ,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_split_pattern' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1607'>str_split_pattern</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>sep2,<br>ll,<br>maxsplit=0):  = 0):  # Find Sentence Patternimport re_ = maxsplitsep2)"(" + regex_pat + r")|(?:(?!" + regex_pat + ").)*",<br>re.S)lambda m:  m.group(1) if m.group(1) else "P",<br>ll)return llx): ,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_str_isascii' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1619'>pd_str_isascii</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_to_utf8' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1627'>str_to_utf8</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br>    """ Do it before saving/output to external printer """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:str_to_unicode' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1632'>str_to_unicode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>encoding = "utf-8",<br></ul>
        <li>Docs:<br>    """ Do it First after Loading some text """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_minimize' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1642'>np_minimize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fun_obj,<br>x0 = None,<br>argext = (0,<br>0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_minimize_de' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1662'>np_minimize_de</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fun_obj,<br>bounds,<br>name1,<br>maxiter = 10,<br>popsize = 5,<br>solver = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_remove_na_inf_2d' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1686'>np_remove_na_inf_2d</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_addcolumn' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1695'>np_addcolumn</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>nbcol,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_addrow' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1702'>np_addrow</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>nbrow,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_int_tostr' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1712'>np_int_tostr</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_dictordered_create' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1721'>np_dictordered_create</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_list_unique' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1727'>np_list_unique</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>seq,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_list_tofreqdict' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1731'>np_list_tofreqdict</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>l1,<br>wweight = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_list_flatten' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1754'>np_list_flatten</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>seq,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_dict_tolist' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1766'>np_dict_tolist</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dd,<br>withkey = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_dict_tostr_val' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1773'>np_dict_tostr_val</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dd,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_dict_tostr_key' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1777'>np_dict_tostr_key</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dd,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_removelist' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1781'>np_removelist</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x0,<br>xremove = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_transform2d_int_1d' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1792'>np_transform2d_int_1d</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>m2d,<br>onlyhalf = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_mergelist' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1809'>np_mergelist</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x0,<br>x1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_enumerate2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1816'>np_enumerate2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>vec_1d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_pivottable_count' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1825'>np_pivottable_count</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>mylist,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_nan_helper' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1834'>np_nan_helper</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y,<br></ul>
        <li>Docs:<br>    """ Input:  - y, 1d numpy array with possible NaNs
<br>
        Output - nans, logical indices of NaNs - index, a function, with signature
<br>
              indices= index(logical_indices),
<br>
              to convert logical indices of NaNs to 'equivalent' indices
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_interpolate_nan' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1843'>np_interpolate_nan</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_and1' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1849'>np_and1</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>y,<br>x3 = None,<br>x4 = None,<br>x5 = None,<br>x6 = None,<br>x7 = None,<br>x8 = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_sortcol' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1864'>np_sortcol</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>colid,<br>asc = 1,<br></ul>
        <li>Docs:<br>    """ df.sort(['A', 'B'], ascending=[1, 0])  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_ma' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1871'>np_ma</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>vv,<br>n,<br></ul>
        <li>Docs:<br>    """Moving average """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_cleanmatrix' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1877'>np_cleanmatrix</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>m,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_torecarray' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1887'>np_torecarray</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>colname,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_sortbycolumn' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1893'>np_sortbycolumn</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>colid,<br>asc = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_sortbycol' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1899'>np_sortbycol</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>colid,<br>asc = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_min_kpos' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1912'>np_min_kpos</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>kth,<br></ul>
        <li>Docs:<br>    """ return kth mininimun """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_max_kpos' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1917'>np_max_kpos</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>kth,<br></ul>
        <li>Docs:<br>    """ return kth mininimun """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_findfirst' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1924'>np_findfirst</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>item,<br>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_find' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1933'>np_find</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>item,<br>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:find' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1941'>find</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>xstring,<br>list_string,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:findnone' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1949'>findnone</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:findx' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1957'>findx</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>item,<br>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:finds' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1970'>finds</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>itemlist,<br>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:findhigher' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1987'>findhigher</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:findlower' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L1995'>findlower</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>vec,<br></ul>
        <li>Docs:<br>    """return the index of the first occurence of item in vec"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_find_minpos' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2003'>np_find_minpos</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>values,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_find_maxpos' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2008'>np_find_maxpos</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>values,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_find_maxpos_2nd' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2013'>np_find_maxpos_2nd</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>numbers,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_findlocalmax2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2028'>np_findlocalmax2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>v,<br>trig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_findlocalmin2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2064'>np_findlocalmin2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>v,<br>trig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_findlocalmax' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2100'>np_findlocalmax</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>v,<br>trig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_findlocalmin' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2116'>np_findlocalmin</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>v,<br>trig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_stack' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2134'>np_stack</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>v1,<br>v2 = None,<br>v3 = None,<br>v4 = None,<br>v5 = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_uniquerows' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2156'>np_uniquerows</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>a,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_remove_zeros' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2162'>np_remove_zeros</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>vv,<br>axis1 = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_sort' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2166'>np_sort</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br>colid,<br>asc = 1,<br></ul>
        <li>Docs:<br>    """
<br>
    Creates a cross-tab or pivot table from a normalised input table. Use this
<br>
    function to 'denormalize' a table of normalized records.
<br>

<br>
    * The table argument can be a list of dictionaries or a Table object.
<br>
    (http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/334621)
<br>
    * The left argument is a tuple of headings which are displayed down the
<br>
    left side of the new table.
<br>
    * The top argument is a tuple of headings which are displayed across the
<br>
    top of the new table.
<br>
    Tuples are used so that multiple element headings and columns can be used.
<br>

<br>
    E.g. To transform the list (listOfDicts):
<br>

<br>
    Name,   Year,  Value
<br>
    -----------------------
<br>
    'Simon', 2004, 32
<br>
    'Simon', 2005, 128
<br>
    'Russel', 2004, 64
<br>
    'Eric', 2004, 52
<br>
    'Russel', 2005, 32
<br>

<br>
    into the new list:
<br>

<br>
    'Name',   2004, 2005
<br>
    ------------------------
<br>
    'Simon',  32,     128
<br>
    'Russel',  64,     32
<br>
    'Eric',   52,     NA
<br>

<br>
    you would call pivot with the arguments:
<br>

<br>
    newList = pivot(listOfDicts, ('Name',), ('Year',), 'Value')
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_memory_array_adress' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2170'>np_memory_array_adress</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_pivotable_create' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2175'>np_pivotable_create</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>table,<br>left,<br>top,<br>value,<br></ul>
        <li>Docs:<br>    """
<br>
    Creates a cross-tab or pivot table from a normalised input table. Use this
<br>
    function to 'denormalize' a table of normalized records.
<br>

<br>
    * The table argument can be a list of dictionaries or a Table object.
<br>
    (http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/334621)
<br>
    * The left argument is a tuple of headings which are displayed down the
<br>
    left side of the new table.
<br>
    * The top argument is a tuple of headings which are displayed across the
<br>
    top of the new table.
<br>
    Tuples are used so that multiple element headings and columns can be used.
<br>

<br>
    E.g. To transform the list (listOfDicts):
<br>

<br>
    Name,   Year,  Value
<br>
    -----------------------
<br>
    'Simon', 2004, 32
<br>
    'Simon', 2005, 128
<br>
    'Russel', 2004, 64
<br>
    'Eric', 2004, 52
<br>
    'Russel', 2005, 32
<br>

<br>
    into the new list:
<br>

<br>
    'Name',   2004, 2005
<br>
    ------------------------
<br>
    'Simon',  32,     128
<br>
    'Russel',  64,     32
<br>
    'Eric',   52,     NA
<br>

<br>
    you would call pivot with the arguments:
<br>

<br>
    newList = pivot(listOfDicts, ('Name',), ('Year',), 'Value')
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_info' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2259'>pd_info</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>doreturn = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_info_memsize' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2268'>pd_info_memsize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>memusage = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_row_findlast' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2275'>pd_row_findlast</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colid = 0,<br>emptyrowid = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_row_select' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2282'>pd_row_select</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>**conditions,<br></ul>
        <li>Docs:<br>    """Select rows from a df according to conditions
<br>
    pdselect(data, a=2, b__lt=3) __gt __ge __lte  __in  __not_in
<br>
    will select all rows where 'a' is 2 and 'b' is less than 3
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_csv_randomread' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2328'>pd_csv_randomread</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filename,<br>nsample = 10000,<br>filemaxline = -1,<br>dtype = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_array_todataframe' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2341'>pd_array_todataframe</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>array,<br>colname = None,<br>index1 = None,<br>dotranspose = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_dataframe_toarray' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2352'>pd_dataframe_toarray</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_createdf' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2359'>pd_createdf</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>array1,<br>col1 = None,<br>idx1 = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_create_colmapdict_nametoint' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2363'>pd_create_colmapdict_nametoint</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br></ul>
        <li>Docs:<br>    """ 'close' ---> 5    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_extract_col_idx_val' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2373'>pd_extract_col_idx_val</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_extract_col_uniquevalue_tocsv' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2377'>pd_extract_col_uniquevalue_tocsv</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colname = "",<br>csvfile = "",<br></ul>
        <li>Docs:<br>    """ Write one column into a file   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_split_col_idx_val' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2385'>pd_split_col_idx_val</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_splitdf_inlist' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2389'>pd_splitdf_inlist</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colid,<br>type1 = "dict",<br></ul>
        <li>Docs:<br>    """ Split df into dictionnary of dict/list """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_find' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2405'>pd_find</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>regex_pattern = "*",<br>col_restrict = None,<br>isnumeric = False,<br>doreturnposition = False,<br></ul>
        <li>Docs:<br>    """ Find string / numeric values inside df columns, return position where found
<br>
     col_restrict : restrict to these columns """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_dtypes_totype2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2466'>pd_dtypes_totype2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>columns = (,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_dtypes' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2472'>pd_dtypes</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>columns = (,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_df_todict2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2492'>pd_df_todict2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colkey = "table",<br>excludekey = ("",<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_df_todict' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2507'>pd_df_todict</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colkey = "table",<br>excludekey = ("",<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_col_addfrom_dfmap' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2518'>pd_col_addfrom_dfmap</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>dfmap,<br>colkey,<br>colval,<br>df_colused,<br>df_colnew,<br>exceptval = -1,<br>inplace = Truedfmap,<br>colkey = colkey,<br>colval=colval)rowi):  = colval)rowi):,<br></ul>
        <li>Docs:<br>    """ Add new columns based on df_map:  In Place Modification of df
<br>
    df:     Dataframe of transactions.
<br>
    dfmap:  FSMaster Dataframe
<br>
      colkey: colum used for dict key.  machine_code
<br>
      colval: colum used for dict val.  adress
<br>

<br>
    df_colused  :     "machine_code"
<br>
    exception val:  -1 or ''
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_applyfun_col' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2578'>pd_applyfun_col</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>newcol,<br>ff,<br>use_colname = "all/[colname]",<br></ul>
        <li>Docs:<br>    """ use all Columns to compute values """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_date_intersection' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2599'>pd_date_intersection</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>qlist,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_is_categorical' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2609'>pd_is_categorical</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>z,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_str_encoding_change' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2618'>pd_str_encoding_change</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>cols,<br>fromenc = "iso-8859-1",<br>toenc = "utf-8",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_str_unicode_tostr' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2625'>pd_str_unicode_tostr</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>targetype = str,<br></ul>
        <li>Docs:<br>    """
<br>
 https://www.azavea.com/blog/2014/03/24/solving-unicode-problems-in-python-2-7/
<br>
 Nearly every Unicode problem can be solved by the proper application of these tools;
<br>
 they will help you build an airlock to keep the inside of your code nice and clean:
<br>

<br>
encode(): Gets you from Unicode -> bytes
<br>
decode(): Gets you from bytes -> Unicode
<br>
codecs.open(encoding="utf-8"): Read and write files directly to/from Unicode (you can use any
<br>
encoding,
<br>
 not just utf-8, but utf-8 is most common).
<br>
u": Makes your string literals into Unicode objects rather than byte sequences.
<br>
Warning: Don't use encode() on bytes or decode() on Unicode objects
<br>

<br>
# >>> uni_greeting % utf8_name
<br>
Traceback (most recent call last):
<br>
 File "<stdin>", line 1, in <module>
<br>
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 3: ordinal not in range(128)
<br>
# Solution:
<br>
# >>> uni_greeting % utf8_name.decode('utf-8')
<br>
u'Hi, my name is Josxe9.'
<br>

<br>
 """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_dtypes_type1_totype2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2653'>pd_dtypes_type1_totype2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>fromtype = str,<br>targetype = str,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_resetindex' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2661'>pd_resetindex</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_insertdatecol' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2666'>pd_insertdatecol</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col,<br>format1="%Y-%m-%d %H:  = "%Y-%m-%d %H:%M:%S:%f",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_replacevalues' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2671'>pd_replacevalues</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>matrix,<br></ul>
        <li>Docs:<br>    """ Matrix replaces df.values  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_removerow' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2681'>pd_removerow</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>row_list_index = (23,<br>45,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_removecol' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2685'>pd_removecol</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df1,<br>name1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_insertrow' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2689'>pd_insertrow</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>rowval,<br>index1 = None,<br>isreset = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_cleanbeforesave' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2699'>pd_h5_cleanbeforesave</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br></ul>
        <li>Docs:<br>    """Clean Column type before Saving in HDFS: Unicode, Datetime  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_addtable' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2716'>pd_h5_addtable</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>tablename,<br>dbfile="F:  = "F:\temp_pandas.h5",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_tableinfo' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2726'>pd_h5_tableinfo</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filenameh5,<br>table,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_dumpinfo' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2731'>pd_h5_dumpinfo</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dbfile=r"E:  = r"E:\_data\stock\intraday_google.h5",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_save' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2753'>pd_h5_save</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>filenameh5="E:  = "E:/_data/_data_outlier.h5",<br>key = "data",<br></ul>
        <li>Docs:<br>    """ File is release after saving it"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2760'>pd_h5_load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filenameh5="E:  = "E:/_data/_data_outlier.h5",<br>table_id = "data",<br>exportype = "pandas",<br>rowstart = -1,<br>rowend = -1,<br>),<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_h5_fromcsv_tohdfs' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2780'>pd_h5_fromcsv_tohdfs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dircsv = "dir1/dir2/",<br>filepattern = "*.csv",<br>tofilehdfs = "file1.h5",<br>tablename = "df",<br>),<br>dtype0 = None,<br>encoding = "utf-8",<br>chunksize = 2000000,<br>mode = "a",<br>form = "table",<br>complib = None,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:pd_np_toh5file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2837'>pd_np_toh5file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>numpyarr,<br>fileout = "file.h5",<br>table1 = "data",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_allinfo' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2844'>date_allinfo</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>

<br>
https://aboutsimon.com/blog/2016/08/04/datetime-vs-Arrow-vs-Pendulum-vs-Delorean-vs-udatetime.html
<br>

<br>

<br>
   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:datetime_tostring' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2853'>datetime_tostring</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>datelist1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_remove_bdays' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2866'>date_remove_bdays</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>from_date,<br>add_days,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_add_bdays' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2884'>date_add_bdays</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>from_date,<br>add_days,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:datenumpy_todatetime' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2902'>datenumpy_todatetime</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>tt,<br>islocaltime = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:datetime_tonumpydate' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2919'>datetime_tonumpydate</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>t,<br>islocaltime = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:datestring_todatetime' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2925'>datestring_todatetime</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>datelist1,<br>format1 = "%Y%m%d",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:datetime_toint' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2937'>datetime_toint</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>datelist1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_holiday' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2946'>date_holiday</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """
<br>
   https://jakevdp.github.io/blog/2015/07/23/learning-seattles-work-habits-from-bicycle-counts/
<br>

<br>
from pandas.tseries.holiday import USFederalHolidayCalendar
<br>
cal = USFederalHolidayCalendar()
<br>
holidays = cal.holidays('2012', '2016', return_name=True)
<br>
holidays.head()
<br>

<br>
holidays_all = pd.concat([holidays, "Day Before " + holidays.shift(-1, 'D'),  "Day After "
<br>
+ holidays.shift(1, 'D')])
<br>
holidays_all = holidays_all.sort_index()
<br>
holidays_all.head()
<br>

<br>
from pandas.tseries.offsets import CustomBusinessDay
<br>
from pandas.tseries.holiday import USFederalHolidayCalendar
<br>
bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
<br>
dateref[-1] - bday_us- bday_us
<br>

<br>
   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_add_bday' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2968'>date_add_bday</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>from_date,<br>add_days,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:dateint_todatetime' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2983'>dateint_todatetime</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>datelist1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_diffinday' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2993'>date_diffinday</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>intdate1,<br>intdate2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_diffinbday' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L2998'>date_diffinbday</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>intd2,<br>intd1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_gencalendar' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3007'>date_gencalendar</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>start = "2010-01-01",<br>end = "2010-01-15",<br>country = "us",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_finddateid' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3017'>date_finddateid</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>date1,<br>dateref,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:datestring_toint' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3042'>datestring_toint</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>datelist1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_now' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3051'>date_now</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>i = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_nowtime' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3062'>date_nowtime</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>type1 = "str",<br>format1="%Y-%m-%d %H:  = "%Y-%m-%d %H:%M:%S:%f",<br></ul>
        <li>Docs:<br>    """ str / stamp /  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:date_generatedatetime' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3076'>date_generatedatetime</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>start = "20100101",<br>nbday = 10,<br>end = "",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil.py:np_numexpr_vec_calc' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil.py#L3088'>np_numexpr_vec_calc</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/util_default.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py'>./utilmy/util_default.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L18'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:log2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L22'>log2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:logw' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L26'>logw</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:loge' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L30'>loge</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:logger_setup' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L34'>logger_setup</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:config_load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L51'>config_load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config_path:  Optional[Union[str,<br>pathlib.Path]]  =  None,<br></ul>
        <li>Docs:<br>    """Load Config file into a dict
<br>
    1) load config_path
<br>
    2) If not, load in HOME USER
<br>
    3) If not, create default one
<br>
    # config_default = yaml.load(os.path.join(os.path.dirname(__file__), 'config', 'config.yaml'))
<br>

<br>
    Args:
<br>
        config_path: path of config or 'default' tag value
<br>
    Returns: dict config
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:dataset_donwload' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L99'>dataset_donwload</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>url,<br>path_target,<br></ul>
        <li>Docs:<br>    """Donwload on disk the tar.gz file
<br>
    Args:
<br>
        url:
<br>
        path_target:
<br>
    Returns:
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:os_extract_archive' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L117'>os_extract_archive</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_path,<br>path = ".",<br>archive_format = "auto",<br></ul>
        <li>Docs:<br>    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
<br>
    Args:
<br>
        file_path: path to the archive file
<br>
        path: path to extract the archive file
<br>
        archive_format: Archive format to try for extracting the file.
<br>
            Options are 'auto', 'tar', 'zip', and None.
<br>
            'tar' includes tar, tar.gz, and tar.bz files.
<br>
            The default 'auto' is ['tar', 'zip'].
<br>
            None or an empty list will return no matches found.
<br>
    Returns:
<br>
        True if a match was found and an archive extraction was completed,
<br>
        False otherwise.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/util_default.py:to_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/util_default.py#L164'>to_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>s,<br>filep,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/zutil_features.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py'>./utilmy/zutil_features.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L12'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br>n = 0,<br>m = 1,<br>**kw,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:log2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L18'>log2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br>**kw,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:log3' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L21'>log3</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br>**kw,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:os_get_function_name' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L32'>os_get_function_name</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:os_getcwd' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L37'>os_getcwd</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pa_read_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L43'>pa_read_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path =   'folder_parquet/',<br>cols = None,<br>n_rows = 1000,<br>file_start = 0,<br>file_end = 100000,<br>verbose = 1,<br>,<br></ul>
        <li>Docs:<br>    """Requied HDFS connection
<br>
       http://arrow.apache.org/docs/python/parquet.html
<br>

<br>
       conda install libhdfs3 pyarrow
<br>
       in your script.py:
<br>
        import os
<br>
        os.environ['ARROW_LIBHDFS_DIR'] = '/opt/cloudera/parcels/CDH/lib64/'
<br>

<br>
       https://stackoverflow.com/questions/18123144/missing-server-jvm-java-jre7-bin-server-jvm-dll
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pa_write_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L97'>pa_write_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>path =   'folder_parquet/',<br>cols = None,<br>n_rows = 1000,<br>partition_cols = None,<br>overwrite = True,<br>verbose = 1,<br>filesystem  =  'hdfs',<br></ul>
        <li>Docs:<br>    """ Pandas to HDFS
<br>
      pyarrow.parquet.write_table(table, where, row_group_size=None, version='1.0',
<br>
      use_dictionary=True, compression='snappy', write_statistics=True, use_deprecated_int96_timestamps=None,
<br>
      coerce_timestamps=None, allow_truncated_timestamps=False, data_page_size=None,
<br>
      flavor=None, filesystem=None, compression_level=None, use_byte_stream_split=False, data_page_version='1.0', **kwargs)
<br>

<br>
      https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_to_dataset.html#pyarrow.parquet.write_to_dataset
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:test_get_classification_data' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L143'>test_get_classification_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:params_check' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L160'>params_check</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>pars,<br>check_list,<br>name = "",<br></ul>
        <li>Docs:<br>    """
<br>
      Validate a dict parans
<br>
    :param pars:
<br>
    :param check_list:
<br>
    :param name:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:save_features' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L186'>save_features</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>name,<br>path = None,<br></ul>
        <li>Docs:<br>    """ Save dataframe on disk
<br>
    :param df:
<br>
    :param name:
<br>
    :param path:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:load_features' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L206'>load_features</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:save_list' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L214'>save_list</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br>name_list,<br>glob,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:save' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L221'>save</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>name,<br>path = None,<br></ul>
        <li>Docs:<br>  """
<br>
      Read file in parallel from disk : very Fast
<br>
  :param path_glob:
<br>
  :param ignore_index:
<br>
  :param cols:
<br>
  :param verbose:
<br>
  :param nrows:
<br>
  :param concat_sort:
<br>
  :param n_pool:
<br>
  :param drop_duplicates:
<br>
  :param shop_id:
<br>
  :param kw:
<br>
  :return:
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:load' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L228'>load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name,<br>path,<br></ul>
        <li>Docs:<br>  """
<br>
      Read file in parallel from disk : very Fast
<br>
  :param path_glob:
<br>
  :param ignore_index:
<br>
  :param cols:
<br>
  :param verbose:
<br>
  :param nrows:
<br>
  :param concat_sort:
<br>
  :param n_pool:
<br>
  :param drop_duplicates:
<br>
  :param shop_id:
<br>
  :param kw:
<br>
  :return:
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_read_file' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L233'>pd_read_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path_glob = "*.pkl",<br>ignore_index = True,<br>cols = None,<br>verbose = False,<br>nrows = -1,<br>concat_sort = True,<br>n_pool = 1,<br>drop_duplicates = None,<br>col_filter = None,<br>col_filter_val = None,<br>**kw,<br></ul>
        <li>Docs:<br>  """
<br>
      Read file in parallel from disk : very Fast
<br>
  :param path_glob:
<br>
  :param ignore_index:
<br>
  :param cols:
<br>
  :param verbose:
<br>
  :param nrows:
<br>
  :param concat_sort:
<br>
  :param n_pool:
<br>
  :param drop_duplicates:
<br>
  :param shop_id:
<br>
  :param kw:
<br>
  :return:
<br>
  """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:load_dataset' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L300'>load_dataset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path_data_x,<br>path_data_y = '',<br>colid = "jobId",<br>n_sample = -1,<br></ul>
        <li>Docs:<br>    """
<br>
      return a datraframe
<br>
      https://raw.github.com/someguy/brilliant/master/somefile.txt
<br>

<br>
    :param path_data_x:
<br>
    :param path_data_y:
<br>
    :param colid:
<br>
    :param n_sample:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:fetch_spark_koalas' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L378'>fetch_spark_koalas</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path_data_x,<br>path_data_y = '',<br>colid = "jobId",<br>n_sample = -1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:fetch_dataset' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L386'>fetch_dataset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>url_dataset,<br>path_target = None,<br>file_target = None,<br></ul>
        <li>Docs:<br>    """Fetch dataset from a given URL and save it.
<br>

<br>
    Currently `github`, `gdrive` and `dropbox` are the only supported sources of
<br>
    data. Also only zip files are supported.
<br>

<br>
    :param url_dataset:   URL to send
<br>
    :param path_target:   Path to save dataset
<br>
    :param file_target:   File to save dataset
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:load_function_uri' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L491'>load_function_uri</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>uri_name="myfolder/myfile.py:  = "myfolder/myfile.py::myFunction",<br></ul>
        <li>Docs:<br>    """
<br>
    #load dynamically function from URI pattern
<br>
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"
<br>
    ###### External File processor :
<br>
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:metrics_eval' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L531'>metrics_eval</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>metric_list = ["mean_squared_error"],<br>ytrue = None,<br>ypred = None,<br>ypred_proba = None,<br>return_dict = False,<br></ul>
        <li>Docs:<br>    """
<br>
      Generic metrics calculation, using sklearn naming pattern
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_dataset_shift' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L572'>pd_stat_dataset_shift</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dftrain,<br>dftest,<br>colused,<br>nsample = 10000,<br>buckets = 5,<br>axis = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_datashift_psi' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L588'>pd_stat_datashift_psi</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>expected,<br>actual,<br>buckettype = 'bins',<br>buckets = 10,<br>axis = 0,<br></ul>
        <li>Docs:<br>    '''Calculate the PSI (population stability index) across all variables
<br>
    Args:
<br>
       expected: numpy matrix of original values
<br>
       actual: numpy matrix of new values, same size as expected
<br>
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
<br>
       buckets: number of quantiles to use in bucketing variables
<br>
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
<br>
    Returns:
<br>
       psi_values: ndarray of psi values for each variable
<br>
    '''
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:feature_importance_perm' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L660'>feature_importance_perm</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>clf,<br>Xtrain,<br>ytrain,<br>cols,<br>n_repeats = 8,<br>scoring = 'neg_root_mean_squared_error',<br>show_graph = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:feature_selection_multicolinear' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L694'>feature_selection_multicolinear</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>threshold = 1.0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:feature_correlation_cat' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L712'>feature_correlation_cat</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colused,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_feature_generate_cross' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L735'>pd_feature_generate_cross</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>cols,<br>cols_cross_input = None,<br>pct_threshold = 0.2,<br>m_combination = 2,<br></ul>
        <li>Docs:<br>    """
<br>
       Generate Xi.Xj features and filter based on stats threshold
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_col_to_onehot' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L770'>pd_col_to_onehot</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dfref,<br>colname = None,<br>colonehot = None,<br>return_val = "dataframe,column",<br></ul>
        <li>Docs:<br>    """
<br>
    :param df:
<br>
    :param colname:
<br>
    :param colonehot: previous one hot columns
<br>
    :param returncol:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colcat_mergecol' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L813'>pd_colcat_mergecol</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>col_list,<br>x0,<br>colid = "easy_id",<br></ul>
        <li>Docs:<br>    """
<br>
       Merge category onehot column
<br>
    :param df:
<br>
    :param l:
<br>
    :param x0:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colcat_tonum' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L837'>pd_colcat_tonum</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colcat = "all",<br>drop_single_label = False,<br>drop_fact_dict = True,<br></ul>
        <li>Docs:<br>    """
<br>
    Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
<br>
    using the following logic:
<br>
    * categorical with only a single value will be marked as zero (or dropped, if requested)
<br>
    * categorical with two values will be replaced with the result of Pandas `factorize`
<br>
    * categorical with more than two values will be replaced with the result of Pandas `get_dummies`
<br>
    * numerical columns will not be modified
<br>
    **Returns:** DataFrame or (DataFrame, dict). If `drop_fact_dict` is True, returns the encoded DataFrame.
<br>
    else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
<br>
    value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
<br>
    present in the data-set
<br>
    Parameters
<br>
    ----------
<br>
    df : NumPy ndarray / Pandas DataFrame
<br>
        The data-set to encode
<br>
    colcat : sequence / string
<br>
        A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
<br>
        all columns are nominal. If None, nothing happens. Default: 'all'
<br>
    drop_single_label : Boolean, default = False
<br>
        If True, nominal columns with a only a single value will be dropped.
<br>
    drop_fact_dict : Boolean, default = True
<br>
        If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
<br>
        the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colcat_mapping' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L889'>pd_colcat_mapping</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colname,<br></ul>
        <li>Docs:<br>    """
<br>
       map category to integers
<br>
    :param df:
<br>
    :param colname:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colcat_toint' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L909'>pd_colcat_toint</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dfref,<br>colname,<br>colcat_map = None,<br>suffix = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colnum_tocat' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L948'>pd_colnum_tocat</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colname = None,<br>colexclude = None,<br>colbinmap = None,<br>bins = 5,<br>suffix = "_bin",<br>method = "uniform",<br>na_value = -1,<br>return_val = "dataframe,param",<br>params={"KMeans_n_clusters":  = {"KMeans_n_clusters": 8,<br>"KMeans_init":  'k-means++',<br>"KMeans_n_init":  10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'},<br></ul>
        <li>Docs:<br>    """
<br>
    colbinmap = for each column, definition of bins
<br>
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
<br>
       :param df:
<br>
       :param method:
<br>
       :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colnum_normalize' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1039'>pd_colnum_normalize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df0,<br>colname,<br>pars,<br>suffix = "_norm",<br>return_val = 'dataframe,param',<br></ul>
        <li>Docs:<br>    """
<br>
    :param df:
<br>
    :param colnum_log:
<br>
    :param colproba:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_col_merge_onehot' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1085'>pd_col_merge_onehot</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colname,<br></ul>
        <li>Docs:<br>    """
<br>
      Merge columns into single (hotn
<br>
    :param df:
<br>
    :param colname:
<br>
    :return :
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_col_to_num' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1102'>pd_col_to_num</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>colname = None,<br>default = np.nan,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_col_filter' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1115'>pd_col_filter</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>filter_val = None,<br>iscol = 1,<br></ul>
        <li>Docs:<br>    """
<br>
   # Remove Columns where Index Value is not in the filter_value
<br>
   # filter1= X_client['client_id'].values
<br>
   :param df:
<br>
   :param filter_val:
<br>
   :param iscol:
<br>
   :return:
<br>
   """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_col_fillna' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1134'>pd_col_fillna</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dfref,<br>colname = None,<br>method = "frequent",<br>value = None,<br>colgroupby = None,<br>return_val = "dataframe,param",<br>,<br></ul>
        <li>Docs:<br>    """
<br>
    Function to fill NaNs with a specific value in certain columns
<br>
    Arguments:
<br>
        df:            dataframe
<br>
        colname:      list of columns to remove text
<br>
        value:         value to replace NaNs with
<br>
    Returns:
<br>
        df:            new dataframe with filled values
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_pipeline_apply' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1180'>pd_pipeline_apply</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>pipeline,<br></ul>
        <li>Docs:<br>    """
<br>
      pipe_preprocess_colnum = [
<br>
      (pd_col_to_num, {"val": "?", })
<br>
    , (pd_colnum_tocat, {"colname": None, "colbinmap": colnum_binmap, 'bins': 5,
<br>
                         "method": "uniform", "suffix": "_bin",
<br>
                         "return_val": "dataframe"})
<br>
    , (pd_col_to_onehot, {"colname": None, "colonehot": colnum_onehot,
<br>
                          "return_val": "dataframe"})
<br>
      ]
<br>
    :param df:
<br>
    :param pipeline:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_correl_pair' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1204'>pd_stat_correl_pair</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>coltarget = None,<br>colname = None,<br></ul>
        <li>Docs:<br>    """
<br>
      Genearte correletion between the column and target column
<br>
      df represents the dataframe comprising the column and colname comprising the target column
<br>
    :param df:
<br>
    :param colname: list of columns
<br>
    :param coltarget : target column
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_pandas_profile' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1225'>pd_stat_pandas_profile</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>savefile = "report.html",<br>title = "Pandas Profile",<br></ul>
        <li>Docs:<br>    """ Describe the tables
<br>
        #Pandas-Profiling 2.0.0
<br>
        df.profile_report()
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_distribution_colnum' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1238'>pd_stat_distribution_colnum</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>nrows = 2000,<br>verbose = False,<br></ul>
        <li>Docs:<br>    """ Stats the tables
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_histogram' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1278'>pd_stat_histogram</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>bins = 50,<br>coltarget = "diff",<br></ul>
        <li>Docs:<br>    """
<br>
    :param df:
<br>
    :param bins:
<br>
    :param coltarget:
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:col_extractname' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1293'>col_extractname</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>col_onehot,<br></ul>
        <li>Docs:<br>    """
<br>
    Column extraction from onehot name
<br>
    :param col_onehot
<br>
    :return:
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:col_remove' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1316'>col_remove</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cols,<br>colsremove,<br>mode = "exact",<br></ul>
        <li>Docs:<br>    """
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_colnum_tocat_stat' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1341'>pd_colnum_tocat_stat</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>feature,<br>target_col,<br>bins,<br>cuts = 0,<br></ul>
        <li>Docs:<br>    """
<br>
    Bins continuous features into equal sample size buckets and returns the target mean in each bucket. Separates out
<br>
    nulls into another bucket.
<br>
    :param df: dataframe containg features and target column
<br>
    :param feature: feature column name
<br>
    :param target_col: target column
<br>
    :param bins: Number bins required
<br>
    :param cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.
<br>
    :return: If cuts are passed only df_grouped data is returned, else cuts and df_grouped data is returned
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_shift_trend_changes' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1410'>pd_stat_shift_trend_changes</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>feature,<br>target_col,<br>threshold = 0.03,<br></ul>
        <li>Docs:<br>    """
<br>
    Calculates number of times the trend of feature wrt target changed direction.
<br>
    :param df: df_grouped dataset
<br>
    :param feature: feature column name
<br>
    :param target_col: target column
<br>
    :param threshold: minimum % difference required to count as trend change
<br>
    :return: number of trend chagnes for the feature
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_shift_trend_correlation' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1434'>pd_stat_shift_trend_correlation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>df_test,<br>colname,<br>target_col,<br></ul>
        <li>Docs:<br>    """
<br>
    Calculates correlation between train and test trend of colname wrt target.
<br>
    :param df: train df data
<br>
    :param df_test: test df data
<br>
    :param colname: colname column name
<br>
    :param target_col: target column name
<br>
    :return: trend correlation between train and test
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:pd_stat_shift_changes' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1465'>pd_stat_shift_changes</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>target_col,<br>features_list = 0,<br>bins = 10,<br>df_test = 0,<br></ul>
        <li>Docs:<br>    """
<br>
    Calculates trend changes and correlation between train/test for list of features
<br>
    :param df: dfframe containing features and target columns
<br>
    :param target_col: target column name
<br>
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
<br>
    :param bins: number of bins to be created from continuous colname
<br>
    :param df_test: test df which has to be compared with input df for correlation
<br>
    :return: dfframe with trend changes and trend correlation (if test df passed)
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/zutil_features.py:np_conv_to_one_col' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L1510'>np_conv_to_one_col</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>np_array,<br>sep_char = "_",<br></ul>
        <li>Docs:<br>    """
<br>
    converts string/numeric columns to one string column
<br>
    :param np_array: the numpy array with more than one column
<br>
    :param sep_char: the separator character
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./utilmy/zutil_features.py:dict2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L28'>dict2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/zutil_features.py:dict2:__init__' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/zutil_features.py#L29'>dict2:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/logs/test_log.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py'>./utilmy/logs/test_log.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/logs/test_log.py:test1' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py#L9'>test1</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/test_log.py:test2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py#L28'>test2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/test_log.py:test_launch_server' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py#L81'>test_launch_server</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>	'''
<br>
	Server code from loguru.readthedocs.io
<br>
	Use to test network logging
<br>

<br>
     python   test.py test_launch_server
<br>

<br>

<br>
	'''
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/logs/test_log.py:test_server' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py#L95'>test_server</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='./utilmy/logs/test_log.py:LoggingStreamHandler' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py#L65'>LoggingStreamHandler</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='./utilmy/logs/test_log.py:LoggingStreamHandler:handle' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/logs/test_log.py#L66'>LoggingStreamHandler:handle</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='./utilmy/dates.py' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py'>./utilmy/dates.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:log' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L6'>log</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>*s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:pd_date_split' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L11'>pd_date_split</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>df,<br>coldate  =   'time_key',<br>prefix_col  = "",<br>verbose = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_now' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L41'>date_now</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fmt="%Y-%m-%d %H:  = "%Y-%m-%d %H:%M:%S %Z%z",<br>add_days = 0,<br>timezone = 'Asia/Tokyo',<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_is_holiday' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L53'>date_is_holiday</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>array,<br></ul>
        <li>Docs:<br>    """
<br>
      is_holiday([ pd.to_datetime("2015/1/1") ] * 10)
<br>

<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_weekmonth2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L63'>date_weekmonth2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_weekmonth' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L71'>date_weekmonth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_weekyear2' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L80'>date_weekyear2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dt,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_weekday_excel' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L84'>date_weekday_excel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_weekyear_excel' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L91'>date_weekyear_excel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='./utilmy/dates.py:date_generate' href='https://github.com/arita37/myutil/blob/main/utilmy//./utilmy/dates.py#L110'>date_generate</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>start = '2018-01-01',<br>ndays = 100,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>
