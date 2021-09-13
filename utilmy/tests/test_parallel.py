from utilmy.tests.conftest import apply_func, fun_async, group_function
from utilmy import parallel


counter = 0
dic = {
    "multithread_run_test": "Incorrect test",
    "multithread_run_list": "Incorrect test",
    "multiproc_run_test": "Incorrect test",
    "pd_groupby_parallel_test": "Incorrect test",
    "pd_apply_parallel_test": "Incorrect test",
}


def test_parallel():
    "testing script for multithread_run"
    li_of_tuples = [
        ("x", "y", "z"),
        ("y", "z", "p"),
        ("yw", "zs", "psd"),
        ("yd", "zf", "pf"),
    ]
    if [["xyz", "ywzspsd"], ["yzp", "ydzfpf"]] == parallel.multithread_run(
        fun_async, li_of_tuples, n_pool=2, start_delay=0.1, verbose=True
    ):
        dic["multithread_run_test"] = "correct test"
    else:
        dic["multithread_run_test"] = "Incorrect test"

    ##### 2nd test for multithread run list

    #####

    """testing the script for checking the list"""
    li_of_tuples = [
        ["x", "y", "z"],
        ["y", "z", "p"],
    ]
    if (
        parallel.multithread_run_list(
            function1=(fun_async, (li_of_tuples[0],)),
            function2=(fun_async, (li_of_tuples[1],)),
        )
        == [("function1", ["x", "y", "z"]), ("function2", ["y", "z", "p"])]
    ):
        dic["multithread_run_list"] = "correct test"

    ### testing
    ####

    "testing script for multiproc_run"
    li_of_tuples = [
        ("x", "y", "z"),
        ("y", "z", "p"),
        ("yw", "zs", "psd"),
        ("yd", "zf", "pf"),
    ]
    if parallel.multiproc_run(
        fun_async, li_of_tuples, n_pool=2, start_delay=0.1, verbose=True
    ) == [["xyz"], ["yzp"], ["ywzspsd"], ["ydzfpf"], []]:
        dic["multiproc_run_test"] = "correct test"

    ####
    #### TEST applying groupby
    """
    test for applying groupby in pandas
    """
    import pandas as pd

    df = pd.DataFrame()
    df["result"] = [5, 8, 1, 7, 0, 3, 2, 9, 4, 6]
    df["user_id"] = [1, 1, 2, 3, 4, 4, 5, 8, 9, 9]
    df["value"] = [27, 14, 26, 19, 28, 9, 11, 1, 26, 18]
    df["data_chunk"] = [1, 1, 2, 3, 4, 4, 5, 8, 9, 9]

    expected_df = df.copy()
    expected_df["inv_sum"] = [14.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 18.0, 0.0]
    result = parallel.pd_groupby_parallel(
        df.groupby("user_id"), func=group_function, int=1
    )
    if expected_df.equals(result):
        dic["pd_groupby_parallel_test"] = "correct test"

    ### next test

    """
    test for applying groupby in pandas
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "A": [0, 1, 2, 3, 4],
            "B": [100, 200, 300, 400, 500],
        }
    )
    expected_df = pd.DataFrame(
        {"A": [0, 1, 4, 9, 16], "B": [10000, 40000, 90000, 160000, 250000]}
    )
    result = parallel.pd_apply_parallel(
        df=df, colsgroup=["A" "B"], fun_apply=apply_func, npool=4
    )
    if expected_df.equals(result):
        dic["pd_apply_parallel_test"] = "correct test"

    return dic


if __name__ == "__main__":
    print(test_parallel())
