def fun_async(xlist):
    list = []
    for x in xlist:
        stdr = ""
        for y in x:
            stdr += y
        list.append(stdr)
    return list


def group_function(name, group):
    # Inverse cumulative sum

    group["inv_sum"] = group.iloc[::-1]["value"].cumsum()[::-1].shift(-1).fillna(0)
    return group


def apply_func(x):
    return x ** 2
