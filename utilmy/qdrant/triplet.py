import itertools
import time
import pandas as pd

def new_algo(df):
    ddict3 = {}
    df = df[['userid', 'genre_id']]
    user_grouped=df.groupby('userid')
    for group_name, df_group in user_grouped:
        geners=df_group["genre_id"].unique()
        geners.sort()
        if len(geners)>2:
            comb=itertools.combinations(geners, 3)
            for i in list(comb):
                key = '_'.join(map(str, i))
                if key not in ddict3:
                    ddict3[key]=[]
                ddict3[key].append(group_name)
    new_ddict3={}
    for key in ddict3.keys():
        if len(ddict3[key])>1:
            new_ddict3[key]=ddict3[key]
    l=[]
    for key,item in new_ddict3.items():
        keys=key.split("_")
        keys.append(','.join(map(str, item)))
        l.append(keys)
    new_df = pd.DataFrame(columns=['g0', 'g1','g2','userlist'], data=l)

    return new_ddict3,new_df

if __name__ == '__main__':
    # read the input
    df = pd.read_csv("test_input.csv")
    # measure the time
    start = time.time()
    # calling the function
    my_dict,new_df=new_algo(df)
    # print the timing
    end = time.time()
    print(end - start)
    # save the output
    with open('test_output.csv', 'w') as f:
        for key in my_dict.keys():
            f.write("%s,%s\n"%(key,my_dict[key]))


            