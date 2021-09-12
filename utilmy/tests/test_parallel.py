from multiprocessing import cpu_count
from utilmy import parallel
import pandas as pd
import numpy as np


counter = 0


def fun_async(xlist):
    print(counter, " Thread ", "Got: ", xlist)
    list = []
    for x in xlist:
        stdr = ""
        for y in x:
            stdr += y
        list.append(stdr)
    return list


def test_multithread_run():
    "testing script for multithread_run"
    li_of_tuples = [
        ("x", "y", "z"),
        ("y", "z", "p"),
        ("yw", "zs", "psd"),
        ("yd", "zf", "pf"),
    ]
    return parallel.multithread_run(
        fun_async, li_of_tuples, n_pool=2, start_delay=0.1, verbose=True
    )


def test_multithread_run_list():
    """testing the script for checking the list"""
    li_of_tuples = [
        ["x", "y", "z"],
        ["y", "z", "p"],
    ]
    return parallel.multithread_run_list(
        function1=(fun_async, (li_of_tuples[0],)),
        function2=(fun_async, (li_of_tuples[1],)),
    )


def test_multiproc_run():
    "testing script for multiproc_run"
    li_of_tuples = [
        ("x", "y", "z"),
        ("y", "z", "p"),
        ("yw", "zs", "psd"),
        ("yd", "zf", "pf"),
    ]
    return parallel.multiproc_run(
        fun_async, li_of_tuples, n_pool=2, start_delay=0.1, verbose=True
    )


def group_function(name, group):
    # Inverse cumulative sum

    group["inv_sum"] = group.iloc[::-1]["value"].cumsum()[::-1].shift(-1).fillna(0)
    return group


def test_pd_groupby_parallel():
    """
    test for applying groupby in pandas
    """
    import pandas as pd

    df = pd.DataFrame()
    # 5000 users with approx 100 values
    df["user_id"] = np.random.randint(5000, size=500000)
    # Generate 500000 random integer values
    df["value"] = np.random.randint(30, size=500000)
    # Create data_chunk based on modulo of user_id
    df["data_chunk"] = df["user_id"].mod(cpu_count() * 3)

    return parallel.pd_groupby_parallel(
        df.groupby("user_id"),
        func=group_function,
        int=1,
    )


def apply_func(x):
    return x ** 2


def test_pd_apply_parallel():
    """
    test for applying groupby in pandas
    """
    import pandas as pd

    df = pd.DataFrame(
        {"A": [0, 1, 2, 3, 4, 5, 6], "B": [100, 200, 300, 400, 500, 600, 700]}
    )

    return parallel.pd_apply_parallel(
        df=df, colsgroup=["A" "B"], fun_apply=apply_func, npool=4
    )


if __name__ == "__main__":
    print("Testing .. \n")
    for fun in [
        test_multithread_run,
        test_multithread_run_list,
        test_multiproc_run,
        test_pd_apply_parallel,
        test_pd_groupby_parallel,
    ]:
        print("Running:", fun.__name__, "\n\n")
        print(fun())
