### CODE PARSER
code_parser.py






### Example 
```

python code_parser.py       





```






### Example code
```

    # Example save in csv format
    file = "{}/test/{}".format(CUR_DIR, "keys.py")
    df = get_list_function_stats(file)
    print(df)
    if df is not None:
        df.to_csv('functions_stats1.csv', index=False)

    df = get_list_class_stats(file)
    print(df)
    if df is not None:
        df.to_csv('class_stats1.csv', index=False)

    df = get_list_method_stats(file)
    print(df)
    if df is not None:
        df.to_csv('method_stats1.csv', index=False)



```











### List functions

- get_list_function_name(file_path)
- get_list_class_name(file_path)
- get_list_class_methods(file_path)
- get_list_variable_global(file_path)
- get_list_function_stats(file_path)
- get_file_stats(file_path)



